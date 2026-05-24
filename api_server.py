#!/usr/bin/env python3
"""
Gomoku AI FastAPI Server
========================
Provides REST API + WebSocket endpoints for the Gomoku AI web UI.
Wraps the existing AI engine (board, network, mcts, vct, train).

Run:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import asyncio
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict

import numpy as np
import torch

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Lazy imports for AI modules (loaded on first use) ──────────────────────
_board_mod = None
_network_mod = None
_mcts_mod = None
_vct_mod = None
_train_mod = None
_config_mod = None


def _ensure_imports():
    global _board_mod, _network_mod, _mcts_mod, _vct_mod, _train_mod, _config_mod
    if _board_mod is None:
        import board as _b
        _board_mod = _b
        import network as _n
        _network_mod = _n
        import mcts as _m
        _mcts_mod = _m
        import vct as _v
        _vct_mod = _v
        import train as _t
        _train_mod = _t
        import config as _c
        _config_mod = _c


# ── Global State ───────────────────────────────────────────────────────────

class TrainingState:
    """Thread-safe training state container."""
    def __init__(self):
        self.running = False
        self.iteration = 0
        self.total_steps = 0
        self.best_elo = 0.0
        self.current_stats: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.trainer = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

    def reset(self):
        self.running = False
        self.iteration = 0
        self.total_steps = 0
        self.best_elo = 0.0
        self.current_stats = {}
        self._stop_flag.clear()


training_state = TrainingState()

# Currently loaded model for play
_current_model = None
_current_model_info: Dict[str, Any] = {}


# ── Pydantic Models ────────────────────────────────────────────────────────

class TrainStartRequest(BaseModel):
    resume_path: Optional[str] = None
    iterations: Optional[int] = None


class ModelLoadRequest(BaseModel):
    path: str


class AnalyzeRequest(BaseModel):
    board: List[List[int]]
    current_player: int


class ConfigUpdateRequest(BaseModel):
    # Common config overrides for next training run
    num_simulations: Optional[int] = None
    num_res_blocks: Optional[int] = None
    num_filters: Optional[int] = None
    learning_rate: Optional[float] = None
    total_iterations: Optional[int] = None
    c_puct: Optional[float] = None
    dirichlet_alpha: Optional[float] = None


# ── Lifespan (must be defined before app) ─────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """Lifespan event: initialize on startup, cleanup on shutdown."""
    # Lazy load: only import config at startup, AI modules loaded on first API call
    print("[Server] Starting Gomoku AI API Server...")
    print("[Server] AI modules will be loaded on first request (lazy initialization)")
    print("[Server] Ready.")

    yield  # App is running

    # Cleanup
    if training_state.running:
        training_state._stop_flag.set()
        print("[Server] Training stop flag set on shutdown")
    print("[Server] Shutting down.")


# ── FastAPI App ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gomoku AI Server",
    description="REST API + WebSocket endpoints for Gomoku AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: run sync function in thread ────────────────────────────────────

async def run_in_thread(func, *args):
    """Run a blocking function in a thread pool and return the result."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)


# ── 1. Training Control ───────────────────────────────────────────────────

@app.post("/api/train/start")
async def start_training(req: TrainStartRequest):
    """Start training in a background thread."""
    _ensure_imports()

    with training_state._lock:
        if training_state.running:
            raise HTTPException(status_code=409, detail="Training is already running")

        training_state.reset()

        # Apply iteration override
        if req.iterations is not None:
            _config_mod.TOTAL_ITERATIONS = req.iterations

        def _train_worker():
            try:
                training_state.running = True
                trainer = _train_mod.Trainer(
                    device='cpu',
                    resume_path=req.resume_path,
                )
                training_state.trainer = trainer

                # Monkey-patch trainer.train to check stop flag
                original_train = trainer.train

                def _interruptible_train():
                    # Override the iteration loop to check stop flag
                    for iteration in range(trainer.iteration, _config_mod.TOTAL_ITERATIONS):
                        if training_state._stop_flag.is_set():
                            print("[Training] Stop flag set, stopping training.")
                            break

                        trainer.iteration = iteration
                        iter_start = time.time()

                        # Update global state before each iteration
                        with training_state._lock:
                            training_state.iteration = iteration
                            training_state.total_steps = trainer.total_steps
                            training_state.best_elo = trainer.best_elo

                        # Run one iteration manually (simplified from Trainer.train)
                        _run_one_iteration(trainer, iteration)

                        with training_state._lock:
                            training_state.total_steps = trainer.total_steps
                            training_state.best_elo = trainer.best_elo
                            # Merge history
                            for k, v in trainer.history.items():
                                if v and (not training_state.history[k] or
                                          len(v) > len(training_state.history[k])):
                                    training_state.history[k] = list(v)

                    # SWA finalize
                    if _config_mod.USE_SWA and trainer.swa_model is not None and trainer.swa_started:
                        try:
                            trainer._swa_finalize()
                        except Exception as e:
                            print(f"[Training] SWA finalize error: {e}")

                    trainer._save_history()
                    print(f"[Training] Done. Best ELO: {trainer.best_elo:.0f}")

                _interruptible_train()

            except Exception as e:
                print(f"[Training] Error: {e}")
                traceback.print_exc()
            finally:
                with training_state._lock:
                    training_state.running = False
                    training_state._stop_flag.clear()

        t = threading.Thread(target=_train_worker, daemon=True)
        training_state._thread = t
        t.start()

    return {"status": "started", "message": "Training started in background"}


def _run_one_iteration(trainer, iteration):
    """Run a single training iteration, updating global state."""
    current_sims = _config_mod.NUM_SIMULATIONS
    if _config_mod.USE_PROGRESSIVE_SIMS:
        for thresh, sims in _config_mod.PROGRESSIVE_SIMS_SCHEDULE:
            if iteration >= thresh:
                current_sims = sims

    print(f"\n[Iter {iteration + 1}/{_config_mod.TOTAL_ITERATIONS}] MCTS={current_sims}")

    # Self-play
    print("  [1/3] Self-play...")
    sp_start = time.time()
    opp_pool = trainer.opponent_pool if _config_mod.USE_OPPONENT_POOL else None
    new_buffer = _train_mod.generate_self_play_data(
        trainer.model, num_games=_config_mod.NUM_GAMES_PER_ITER,
        num_actors=_config_mod.NUM_ACTORS, opponent_pool=opp_pool
    )
    sp_time = time.time() - sp_start

    # Merge replay buffer
    if _config_mod.USE_SUMTREE:
        for i in range(len(new_buffer)):
            data = new_buffer._tree.data[i]
            if data is not None:
                trainer.replay_buffer.add_sample(data)
    else:
        trainer.replay_buffer.buffer.extend(new_buffer.buffer)
        trainer.replay_buffer.priorities.extend(new_buffer.priorities)

    print(f"  Self-play: {sp_time:.1f}s, buffer: {len(trainer.replay_buffer)}")

    # Training
    stats = {}
    if len(trainer.replay_buffer) >= _config_mod.REPLAY_MIN_SIZE:
        print("  [2/3] Training...")
        train_start = time.time()
        stats = trainer._train_step(iteration)
        train_time = time.time() - train_start
        print(f"  Training: {train_time:.1f}s, p_loss={stats.get('policy_loss', 0):.4f}, "
              f"v_loss={stats.get('value_loss', 0):.4f}")
        trainer.history['policy_loss'].append(stats.get('policy_loss', 0))
        trainer.history['value_loss'].append(stats.get('value_loss', 0))

        with training_state._lock:
            training_state.current_stats = stats
    else:
        print(f"  Buffer insufficient ({len(trainer.replay_buffer)}/{_config_mod.REPLAY_MIN_SIZE})")

    # Opponent pool
    if _config_mod.USE_OPPONENT_POOL and trainer.opponent_pool is not None:
        import copy
        trainer.opponent_pool.append(copy.deepcopy(trainer.model.state_dict()))
        if len(trainer.opponent_pool) > _config_mod.OPPONENT_POOL_SIZE:
            trainer.opponent_pool.pop(0)

    # Optimizer switch
    if (_config_mod.USE_OPTIMIZER_SWITCH and trainer._use_adamw
            and trainer.total_steps >= _config_mod.OPTIMIZER_SWITCH_STEP):
        trainer.optimizer = torch.optim.SGD(
            trainer.model.parameters(), lr=_config_mod.LEARNING_RATE * 0.1,
            momentum=_config_mod.MOMENTUM, weight_decay=_config_mod.WEIGHT_DECAY, nesterov=True
        )
        trainer.lr_scheduler = _train_mod.create_lr_scheduler(trainer.optimizer)
        trainer._use_adamw = False

    # Eval & Save
    elo = 0.0
    if (iteration + 1) % _config_mod.EVAL_INTERVAL == 0:
        print("  [3/3] Evaluating...")
        if _config_mod.USE_CHAMPION_EVAL and trainer.champion_model is not None:
            win_rate = trainer._champion_eval()
            if win_rate > 0.5:
                elo = max(0, -400 * np.log10(max(0.01, 1 / max(0.01, win_rate) - 1)) + 1000)
            else:
                elo = 0
            if win_rate >= _config_mod.CHAMPION_WIN_RATE:
                import copy
                trainer.champion_model = copy.deepcopy(trainer.model)
                trainer.champion_model.eval()
                trainer.best_elo = max(trainer.best_elo, elo)
                trainer.save_checkpoint(_config_mod.BEST_MODEL_PATH)
        else:
            elo = trainer._simple_eval()
            if elo > trainer.best_elo:
                trainer.best_elo = elo
                trainer.save_checkpoint(_config_mod.BEST_MODEL_PATH)

        trainer.history['elo'].append(elo)

    if (iteration + 1) % _config_mod.SAVE_INTERVAL == 0:
        path = os.path.join(_config_mod.CHECKPOINT_DIR, f"model_iter_{iteration + 1}.pt")
        trainer.save_checkpoint(path)

    iter_time = time.time() - (time.time() - sp_start)  # approximate
    print(f"  Iteration done. Best ELO: {trainer.best_elo:.0f}")


@app.post("/api/train/stop")
async def stop_training():
    """Stop training gracefully."""
    with training_state._lock:
        if not training_state.running:
            raise HTTPException(status_code=400, detail="Training is not running")
        training_state._stop_flag.set()

    return {"status": "stopping", "message": "Stop flag set, training will stop after current iteration"}


@app.get("/api/train/status")
async def get_train_status():
    """Get current training status."""
    with training_state._lock:
        return {
            "running": training_state.running,
            "iteration": training_state.iteration,
            "total_steps": training_state.total_steps,
            "best_elo": training_state.best_elo,
            "current_stats": training_state.current_stats,
        }


@app.get("/api/train/history")
async def get_train_history():
    """Get training history arrays."""
    with training_state._lock:
        return dict(training_state.history)


# ── 2. Model Management ──────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """List available checkpoint models."""
    # Only need config for CHECKPOINT_DIR, not full AI modules
    global _config_mod
    if _config_mod is None:
        import config as _c
        _config_mod = _c
    checkpoint_dir = _config_mod.CHECKPOINT_DIR
    models = []

    if not os.path.exists(checkpoint_dir):
        return models

    for f in sorted(Path(checkpoint_dir).glob("*.pt")):
        stat = f.stat()
        models.append({
            "name": f.name,
            "path": str(f),
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
        })

    return models


@app.post("/api/models/load")
async def load_model(req: ModelLoadRequest):
    """Load a specific model for play."""
    _ensure_imports()
    global _current_model, _current_model_info

    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {req.path}")

    try:
        model = _network_mod.create_model(device='cpu', optimize_for_inference=False)
        ckpt = torch.load(req.path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        _current_model = model
        _current_model_info = {
            "path": req.path,
            "name": os.path.basename(req.path),
            "iteration": ckpt.get('iteration', 0),
            "best_elo": ckpt.get('best_elo', 0),
            "total_steps": ckpt.get('total_steps', 0),
        }

        return {"status": "loaded", "model": _current_model_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/api/models/current")
async def get_current_model():
    """Get currently loaded model info."""
    if _current_model is None:
        return {"loaded": False, "model": None}
    return {"loaded": True, "model": _current_model_info}


# ── 3. Human vs AI Game (WebSocket) ──────────────────────────────────────

@app.websocket("/ws/play")
async def ws_human_vs_ai(websocket: WebSocket):
    """WebSocket for real-time human vs AI game play."""
    await websocket.accept()
    _ensure_imports()

    board = None
    mcts = None
    human_color = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                # Start a new game
                human_color = data.get("human_color", 1)
                model_path = data.get("model_path")

                # Load model if specified
                model = _current_model
                if model_path:
                    if os.path.exists(model_path):
                        model = _network_mod.create_model(device='cpu', optimize_for_inference=False)
                        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                        model.load_state_dict(ckpt['model_state_dict'])
                        model.eval()
                    else:
                        await websocket.send_json({"type": "error", "message": f"Model not found: {model_path}"})
                        continue
                elif model is None:
                    # Use default model or create a fresh one
                    model = _network_mod.create_model(device='cpu', optimize_for_inference=False)
                    model.eval()

                board = _board_mod.Board()
                mcts = _mcts_mod.MCTS(model, num_simulations=_config_mod.NUM_SIMULATIONS,
                                       add_noise=False, temperature=0.0)

                # Send initial state
                await websocket.send_json({
                    "type": "state",
                    "board": board.board.tolist(),
                    "current_player": board.current_player,
                    "game_over": board.game_over,
                    "winner": board.winner,
                    "last_move": None,
                })

                # If AI goes first (human is WHITE), make AI move
                if human_color == 2 and board.current_player == 1:
                    await _make_ai_move(websocket, board, mcts)

            elif msg_type == "move":
                if board is None:
                    await websocket.send_json({"type": "error", "message": "Game not started"})
                    continue

                row = data.get("row")
                col = data.get("col")

                if row is None or col is None:
                    await websocket.send_json({"type": "error", "message": "Missing row/col"})
                    continue

                if board.game_over:
                    await websocket.send_json({"type": "error", "message": "Game is over"})
                    continue

                if board.current_player != human_color:
                    await websocket.send_json({"type": "error", "message": "Not your turn"})
                    continue

                # Place human move
                if not board.is_legal(row, col):
                    await websocket.send_json({"type": "error", "message": "Illegal move"})
                    continue

                move_idx = board.get_move_index(row, col)
                board.place_stone(row, col)
                if mcts:
                    mcts.advance(move_idx)

                # Send updated state
                await websocket.send_json({
                    "type": "state",
                    "board": board.board.tolist(),
                    "current_player": board.current_player,
                    "game_over": board.game_over,
                    "winner": board.winner,
                    "last_move": [row, col],
                })

                if board.game_over:
                    await websocket.send_json({
                        "type": "game_over",
                        "winner": board.winner,
                    })
                    continue

                # AI move
                await _make_ai_move(websocket, board, mcts)

            elif msg_type == "undo":
                if board is None or not board.move_history:
                    await websocket.send_json({"type": "error", "message": "Cannot undo"})
                    continue

                # Undo two moves (human + AI), or one if only one was made
                undone = 0
                while board.move_history and undone < 2:
                    board.undo_stone()
                    undone += 1
                    # If we just undid the human's move, stop
                    if board.current_player == human_color:
                        break

                # Reset MCTS since we can't easily undo in the tree
                if mcts:
                    mcts.root = None

                await websocket.send_json({
                    "type": "state",
                    "board": board.board.tolist(),
                    "current_player": board.current_player,
                    "game_over": board.game_over,
                    "winner": board.winner,
                    "last_move": ([board.move_history[-1][0], board.move_history[-1][1]]
                                  if board.move_history else None),
                })

    except WebSocketDisconnect:
        print("[WS /ws/play] Client disconnected")
    except Exception as e:
        print(f"[WS /ws/play] Error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _make_ai_move(websocket, board, mcts):
    """Run MCTS search and send AI move."""
    if board.game_over:
        return

    start_time = time.time()

    # Run MCTS in a thread to not block the event loop
    probs, value = await run_in_thread(mcts.search, board)

    thinking_time = time.time() - start_time

    action = int(np.argmax(probs))
    ai_row, ai_col = board.index_to_move(action)

    board.place_stone(ai_row, ai_col)
    mcts.advance(action)

    # Send AI move
    await websocket.send_json({
        "type": "ai_move",
        "row": ai_row,
        "col": ai_col,
        "value": float(value),
        "thinking_time": round(thinking_time, 3),
    })

    # Send updated state
    await websocket.send_json({
        "type": "state",
        "board": board.board.tolist(),
        "current_player": board.current_player,
        "game_over": board.game_over,
        "winner": board.winner,
        "last_move": [ai_row, ai_col],
    })

    if board.game_over:
        await websocket.send_json({
            "type": "game_over",
            "winner": board.winner,
        })


# ── 4. AI vs AI Game (WebSocket) ─────────────────────────────────────────

@app.websocket("/ws/battle")
async def ws_ai_vs_ai(websocket: WebSocket):
    """WebSocket for AI vs AI matches."""
    await websocket.accept()
    _ensure_imports()

    try:
        data = await websocket.receive_json()
        if data.get("type") != "start":
            await websocket.send_json({"type": "error", "message": "Must send 'start' first"})
            return

        model1_path = data.get("model1_path")
        model2_path = data.get("model2_path")
        num_simulations = data.get("num_simulations", 200)

        # Load models
        model1 = await _load_model_or_default(model1_path)
        model2 = await _load_model_or_default(model2_path)

        board = _board_mod.Board()
        mcts1 = _mcts_mod.MCTS(model1, num_simulations=num_simulations,
                                add_noise=False, temperature=0.0)
        mcts2 = _mcts_mod.MCTS(model2, num_simulations=num_simulations,
                                add_noise=False, temperature=0.0)

        move_number = 0

        while not board.game_over and board.move_count < _config_mod.MAX_MOVES:
            # Determine which model plays
            if board.current_player == 1:  # BLACK = model1
                mcts = mcts1
            else:  # WHITE = model2
                mcts = mcts2

            # Run search in thread
            probs, value = await run_in_thread(mcts.search, board)

            action = int(np.argmax(probs))
            row, col = board.index_to_move(action)
            board.place_stone(row, col)
            mcts.advance(action)
            # Also advance the other MCTS for subtree reuse
            if mcts is mcts1:
                mcts2.advance(action)
            else:
                mcts1.advance(action)

            move_number += 1

            await websocket.send_json({
                "type": "move",
                "player": 3 - board.current_player,  # the player who just moved
                "row": row,
                "col": col,
                "value": round(float(value), 4),
                "move_number": move_number,
            })

            # Small delay to avoid flooding
            await asyncio.sleep(0.01)

        # Game over
        await websocket.send_json({
            "type": "game_over",
            "winner": board.winner,
            "total_moves": board.move_count,
        })

    except WebSocketDisconnect:
        print("[WS /ws/battle] Client disconnected")
    except Exception as e:
        print(f"[WS /ws/battle] Error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _load_model_or_default(model_path: Optional[str]):
    """Load a model from path, or use the current model, or create a default."""
    _ensure_imports()
    if model_path and os.path.exists(model_path):
        model = _network_mod.create_model(device='cpu', optimize_for_inference=False)
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model
    elif _current_model is not None:
        return _current_model
    else:
        model = _network_mod.create_model(device='cpu', optimize_for_inference=False)
        model.eval()
        return model


# ── 5. Config ─────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    """Return key configuration parameters."""
    # Only import config, not all AI modules
    global _config_mod
    if _config_mod is None:
        import config as _c
        _config_mod = _c
    return {
        "BOARD_SIZE": _config_mod.BOARD_SIZE,
        "WIN_LENGTH": _config_mod.WIN_LENGTH,
        "NUM_SIMULATIONS": _config_mod.NUM_SIMULATIONS,
        "NUM_RES_BLOCKS": _config_mod.NUM_RES_BLOCKS,
        "NUM_FILTERS": _config_mod.NUM_FILTERS,
        "INPUT_CHANNELS": _config_mod.INPUT_CHANNELS,
        "C_PUCT": _config_mod.C_PUCT,
        "DIRICHLET_ALPHA": _config_mod.DIRICHLET_ALPHA,
        "LEARNING_RATE": _config_mod.LEARNING_RATE,
        "TOTAL_ITERATIONS": _config_mod.TOTAL_ITERATIONS,
        "BATCH_SIZE": _config_mod.LEARNER_BATCH_SIZE,
        "USE_VCT": _config_mod.USE_VCT,
        "VCT_DEPTH_LIMIT": _config_mod.VCT_DEPTH_LIMIT,
        "VCF_DEPTH_LIMIT": _config_mod.VCF_DEPTH_LIMIT,
        "USE_MUST_MOVE": _config_mod.USE_MUST_MOVE,
        "USE_PATTERN_INJECTION": _config_mod.USE_PATTERN_INJECTION,
        "CHECKPOINT_DIR": _config_mod.CHECKPOINT_DIR,
        "BEST_MODEL_PATH": _config_mod.BEST_MODEL_PATH,
        "NUM_ACTORS": _config_mod.NUM_ACTORS,
        "NUM_GAMES_PER_ITER": _config_mod.NUM_GAMES_PER_ITER,
    }


@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    """Update configuration (for next training run)."""
    global _config_mod
    if _config_mod is None:
        import config as _c
        _config_mod = _c

    if training_state.running:
        raise HTTPException(status_code=409, detail="Cannot change config while training is running")

    updated = {}
    if req.num_simulations is not None:
        _config_mod.NUM_SIMULATIONS = req.num_simulations
        updated["NUM_SIMULATIONS"] = req.num_simulations
    if req.num_res_blocks is not None:
        _config_mod.NUM_RES_BLOCKS = req.num_res_blocks
        updated["NUM_RES_BLOCKS"] = req.num_res_blocks
    if req.num_filters is not None:
        _config_mod.NUM_FILTERS = req.num_filters
        updated["NUM_FILTERS"] = req.num_filters
    if req.learning_rate is not None:
        _config_mod.LEARNING_RATE = req.learning_rate
        updated["LEARNING_RATE"] = req.learning_rate
    if req.total_iterations is not None:
        _config_mod.TOTAL_ITERATIONS = req.total_iterations
        updated["TOTAL_ITERATIONS"] = req.total_iterations
    if req.c_puct is not None:
        _config_mod.C_PUCT = req.c_puct
        updated["C_PUCT"] = req.c_puct
    if req.dirichlet_alpha is not None:
        _config_mod.DIRICHLET_ALPHA = req.dirichlet_alpha
        updated["DIRICHLET_ALPHA"] = req.dirichlet_alpha

    return {"status": "updated", "changed": updated}


# ── 6. Board Analysis ─────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_position(req: AnalyzeRequest):
    """Analyze a board position."""
    _ensure_imports()

    try:
        board_arr = np.array(req.board, dtype=np.int8)
        current_player = req.current_player

        if board_arr.shape != (_config_mod.BOARD_SIZE, _config_mod.BOARD_SIZE):
            raise HTTPException(status_code=400,
                                detail=f"Board must be {_config_mod.BOARD_SIZE}x{_config_mod.BOARD_SIZE}")

        if current_player not in (1, 2):
            raise HTTPException(status_code=400, detail="current_player must be 1 or 2")

        # Must-move detection
        must_idx, must_type = _vct_mod.find_must_move(board_arr, current_player)
        must_move = None
        if must_idx >= 0:
            must_move = {
                "idx": int(must_idx),
                "row": int(must_idx // _config_mod.BOARD_SIZE),
                "col": int(must_idx % _config_mod.BOARD_SIZE),
                "type": int(must_type),
            }

        # VCT/VCF search
        vcf_result = -1
        vct_result = -1
        if _config_mod.USE_VCT:
            vcf_result = _vct_mod.vcf_search(board_arr, current_player, _config_mod.VCF_DEPTH_LIMIT)
            if vcf_result < 0:
                vct_result = _vct_mod.vct_search(board_arr, current_player, _config_mod.VCT_DEPTH_LIMIT)

        # Neural network evaluation (if model loaded)
        evaluation = 0.0
        policy = [0.0] * _config_mod.BOARD_SQUARES
        if _current_model is not None:
            # Create a temporary Board to get feature planes
            tmp_board = _board_mod.Board()
            tmp_board.board = board_arr.copy()
            tmp_board.current_player = current_player
            tmp_board.move_count = int(np.count_nonzero(board_arr))

            feature = tmp_board.get_feature_planes()
            legal_indices = tmp_board.get_legal_move_indices()
            legal_mask = np.zeros(_config_mod.BOARD_SQUARES, dtype=np.float32)
            for idx in legal_indices:
                legal_mask[idx] = 1.0

            p, v = _current_model.predict(feature, legal_mask)
            evaluation = float(v)
            policy = p.tolist()

        return {
            "must_move": must_move,
            "vcf_result": int(vcf_result),
            "vct_result": int(vct_result),
            "evaluation": evaluation,
            "policy": policy,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── Health Check ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "training_running": training_state.running,
        "model_loaded": _current_model is not None,
    }


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
