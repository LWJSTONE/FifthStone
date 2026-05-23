"""
训练引擎 (V2 — 全面修复+优化版)
================================
V2 修复:
  1. SWA 终结化不依赖 self.loader (修复崩溃)
  2. KL 正则正确实现 (保存旧策略分布, 计算 KL(old‖new))
  3. 课程学习真正在9×9棋盘上训练
  4. Replay buffer 合并效率优化

V2 优化:
  1. AdamW + SGD 切换
  2. EMA 模型权重
  3. 历史对手池
  4. 渐进式 MCTS 模拟数
  5. 余弦退火 + 预热 + Warm Restarts
  6. 正确的 Huber 损失
  7. 优先经验回放 (SumTree)
"""

import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from collections import defaultdict

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS, MAX_MOVES,
    LEARNING_RATE, LR_WARMUP_STEPS, LR_DECAY_STEPS,
    WEIGHT_DECAY, MOMENTUM, GRAD_CLIP,
    POLICY_LOSS_WEIGHT, VALUE_LOSS_WEIGHT, KL_REG_WEIGHT, HUBER_DELTA,
    LEARNER_BATCH_SIZE, REPLAY_MIN_SIZE, REPLAY_BUFFER_SIZE,
    TOTAL_ITERATIONS, EVAL_INTERVAL, EVAL_GAMES,
    SAVE_INTERVAL, CHECKPOINT_DIR, BEST_MODEL_PATH,
    NUM_GAMES_PER_ITER, NUM_ACTORS, NUM_SIMULATIONS,
    PRIORITY_BETA_START, PRIORITY_BETA_FRAMES,
    USE_SWA, SWA_START_STEP, SWA_UPDATE_FREQ, SWA_LR,
    USE_CHAMPION_EVAL, CHAMPION_WIN_RATE,
    USE_CURRICULUM, CURRICULUM_SMALL_SIZE, CURRICULUM_SMALL_ITERS,
    USE_OPTIMIZER_SWITCH, OPTIMIZER_SWITCH_STEP,
    USE_EMA, EMA_DECAY,
    USE_OPPONENT_POOL, OPPONENT_POOL_SIZE,
    USE_PROGRESSIVE_SIMS, PROGRESSIVE_SIMS_SCHEDULE,
    PRIORITY_ALPHA, USE_SUMTREE,
)
from network import create_model, create_inference_model
from self_play import generate_self_play_data, self_play_game, ReplayBuffer
from board import Board, BLACK, WHITE
from mcts import MCTS


def create_lr_scheduler(optimizer, warmup_steps=LR_WARMUP_STEPS, decay_steps=LR_DECAY_STEPS):
    """余弦退火 + 线性预热"""
    def lr_lambda(step):
        warmup_f = min(1.0, step / max(1, warmup_steps))
        decay_f = 0.5 * (1.0 + np.cos(np.pi * min(step, decay_steps) / decay_steps))
        return warmup_f * decay_f
    return LambdaLR(optimizer, lr_lambda)


class EMAModel:
    """V2: 指数移动平均模型权重 — 评估更稳定"""
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {}
        self._backup = {}  # V3: 备份原始权重, 用于 restore
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        """将 EMA 权重应用到模型"""
        # V3: 先备份原始权重
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """V3 修复: 恢复原始权重 (评估后)"""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()


class Trainer:
    """V2 AlphaZero 训练器 (全面修复版)"""

    def __init__(self, device='cpu', resume_path=None):
        self.device = device
        self.iteration = 0
        self.total_steps = 0
        self.best_elo = 0

        # 模型
        self.model = create_model(device=device)

        # V2: 优化器切换 (AdamW → SGD)
        if USE_OPTIMIZER_SWITCH:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )
            self._use_adamw = True
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=LEARNING_RATE,
                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True
            )
            self._use_adamw = False

        self.lr_scheduler = create_lr_scheduler(self.optimizer)

        # SWA
        if USE_SWA:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=SWA_LR)
            self.swa_started = False
        else:
            self.swa_model = None

        # V2: EMA
        if USE_EMA:
            self.ema_model = EMAModel(self.model, EMA_DECAY)
        else:
            self.ema_model = None

        # Champion 模型
        self.champion_model = None
        if USE_CHAMPION_EVAL:
            self.champion_model = copy.deepcopy(self.model)
            self.champion_model.eval()

        # V2: 历史对手池
        self.opponent_pool = [] if USE_OPPONENT_POOL else None

        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
        self.history = defaultdict(list)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

    def train(self):
        """V2 主训练循环"""
        print("\n" + "=" * 60)
        print("  五子棋 AI 训练 — V2 全面修复版")
        print("=" * 60)
        print(f"  设备: {self.device}")
        print(f"  参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  迭代: {TOTAL_ITERATIONS}")
        print(f"  对局: {NUM_GAMES_PER_ITER}/轮")
        print(f"  SWA: {USE_SWA}, Champion评估: {USE_CHAMPION_EVAL}")
        print(f"  课程学习: {USE_CURRICULUM}")
        print(f"  优化器切换: {USE_OPTIMIZER_SWITCH}")
        print(f"  EMA: {USE_EMA}, 对手池: {USE_OPPONENT_POOL}")
        print("=" * 60 + "\n")

        # 课程学习: 先在9x9上训练
        if USE_CURRICULUM and self.iteration == 0:
            self._curriculum_phase()

        for iteration in range(self.iteration, TOTAL_ITERATIONS):
            self.iteration = iteration
            iter_start = time.time()

            # V2: 渐进式 MCTS 模拟数
            current_sims = NUM_SIMULATIONS
            if USE_PROGRESSIVE_SIMS:
                for thresh, sims in PROGRESSIVE_SIMS_SCHEDULE:
                    if iteration >= thresh:
                        current_sims = sims

            print(f"\n{'─' * 50}")
            print(f"迭代 {iteration + 1}/{TOTAL_ITERATIONS} (MCTS={current_sims})")
            print(f"{'─' * 50}")

            # 自我对弈
            print("[1/3] 自我对弈...")
            sp_start = time.time()

            # V2: 传入对手池
            opp_pool = self.opponent_pool if USE_OPPONENT_POOL else None
            new_buffer = generate_self_play_data(
                self.model, num_games=NUM_GAMES_PER_ITER,
                num_actors=NUM_ACTORS, opponent_pool=opp_pool
            )
            sp_time = time.time() - sp_start

            # V3 修复: 合并回放缓冲区 — SumTree 存储的是 (state, policy, value) 样本, 不是 GameData
            if USE_SUMTREE:
                for i in range(len(new_buffer)):
                    data = new_buffer._tree.data[i]
                    if data is not None:
                        self.replay_buffer.add_sample(data)
            else:
                self.replay_buffer.buffer.extend(new_buffer.buffer)
                self.replay_buffer.priorities.extend(new_buffer.priorities)

            print(f"  耗时: {sp_time:.1f}s, 缓冲区: {len(self.replay_buffer)}")

            # 训练
            if len(self.replay_buffer) >= REPLAY_MIN_SIZE:
                print("[2/3] 训练...")
                train_start = time.time()
                stats = self._train_step(iteration)
                train_time = time.time() - train_start

                print(f"  耗时: {train_time:.1f}s")
                print(f"  策略损失: {stats['policy_loss']:.4f}")
                print(f"  价值损失: {stats['value_loss']:.4f}")
                print(f"  KL正则: {stats['kl_reg']:.4f}")
                print(f"  LR: {stats['lr']:.6f}")

                self.history['policy_loss'].append(stats['policy_loss'])
                self.history['value_loss'].append(stats['value_loss'])
            else:
                print(f"[2/3] 缓冲区不足 ({len(self.replay_buffer)}/{REPLAY_MIN_SIZE})")

            # V2: 更新历史对手池
            if USE_OPPONENT_POOL and self.opponent_pool is not None:
                self.opponent_pool.append(copy.deepcopy(self.model.state_dict()))
                if len(self.opponent_pool) > OPPONENT_POOL_SIZE:
                    self.opponent_pool.pop(0)

            # V2: 优化器切换
            if USE_OPTIMIZER_SWITCH and self._use_adamw and self.total_steps >= OPTIMIZER_SWITCH_STEP:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=LEARNING_RATE * 0.1,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True
                )
                self.lr_scheduler = create_lr_scheduler(self.optimizer)
                self._use_adamw = False
                print("  ★ 切换优化器: AdamW → SGD")

            # 评估 & 保存
            if (iteration + 1) % EVAL_INTERVAL == 0:
                print("[3/3] 评估...")
                if USE_CHAMPION_EVAL and self.champion_model is not None:
                    win_rate = self._champion_eval()
                    elo = max(0, -400 * np.log10(max(0.01, 1/max(0.01, win_rate) - 1)) + 1000) if win_rate > 0.5 else 0
                    print(f"  vs Champion 胜率: {win_rate:.1%}, ELO≈{elo:.0f}")

                    if win_rate >= CHAMPION_WIN_RATE:
                        self.champion_model = copy.deepcopy(self.model)
                        self.champion_model.eval()
                        self.best_elo = max(self.best_elo, elo)
                        self.save_checkpoint(BEST_MODEL_PATH)
                        print(f"  ★ 新Champion! 胜率={win_rate:.1%}")
                else:
                    elo = self._simple_eval()
                    print(f"  ELO估计: {elo:.0f}")
                    if elo > self.best_elo:
                        self.best_elo = elo
                        self.save_checkpoint(BEST_MODEL_PATH)

                self.history['elo'].append(elo)

            if (iteration + 1) % SAVE_INTERVAL == 0:
                path = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration + 1}.pt")
                self.save_checkpoint(path)

            iter_time = time.time() - iter_start
            print(f"  迭代耗时: {iter_time:.1f}s, 最佳ELO: {self.best_elo:.0f}")

        # V2 修复: SWA 终结化 (不依赖 self.loader)
        if USE_SWA and self.swa_model is not None and self.swa_started:
            self._swa_finalize()

        self._save_history()
        print(f"\n训练完成! 最佳ELO: {self.best_elo:.0f}")

    def _swa_finalize(self):
        """V6 修复: SWA 终结化 — 正确更新 BN 统计 (需 train 模式)"""
        print("[SWA] 终结化: 更新 BN 统计...")

        # V6 修复: 使用 torch.optim.swa_utils.update_bn 正式接口
        # 而非手动 forward (eval模式下BN不更新统计)
        try:
            from torch.optim.swa_utils import update_bn

            # 创建一个简单的 dataloader 从回放缓冲区采样
            sample = self.replay_buffer.sample(min(256, len(self.replay_buffer)))
            if sample is not None:
                states, _, _, _, _ = sample
                states_t = torch.from_numpy(states).to(self.device)

                # update_bn 需要 loader, 创建一个单批次的简易 loader
                class _SimpleLoader:
                    def __init__(self, data):
                        self.data = [data]
                    def __iter__(self):
                        for d in self.data:
                            yield d
                    def __len__(self):
                        return 1

                loader = _SimpleLoader(states_t)
                update_bn(loader, self.swa_model, device=self.device)
            print("[SWA] BN 更新完成 (使用 update_bn)")
        except (ImportError, Exception) as e:
            # 回退: 手动在 train 模式下前向传播更新 BN
            print(f"[SWA] update_bn 不可用 ({e}), 使用手动更新")
            # V6 关键: 必须在 train 模式下才能更新 BN running_mean/running_var
            self.swa_model.train()
            sample = self.replay_buffer.sample(min(256, len(self.replay_buffer)))
            if sample is not None:
                states, _, _, _, _ = sample
                states_t = torch.from_numpy(states).to(self.device)
                with torch.no_grad():
                    self.swa_model(states_t)
            self.swa_model.eval()
            print("[SWA] BN 更新完成 (手动模式)")

    def _curriculum_phase(self):
        """
        V3 修复: 课程学习 — 由于 Numba JIT 编译时 BOARD_SIZE 已固化,
        无法在运行时切换 9×9 棋盘. 改为在 15×15 棋盘上使用少量 MCTS 模拟快速预训练.
        """
        print(f"\n[课程学习] 在 15×15 棋盘上快速预训练 (低 MCTS 模拟数)...")
        print(f"  注意: Numba JIT 编译后 BOARD_SIZE 不可变, 无法切换到 9×9")

        for i in range(CURRICULUM_SMALL_ITERS):
            print(f"  课程 {i+1}/{CURRICULUM_SMALL_ITERS}")
            # 使用少量 MCTS 模拟在标准棋盘上快速生成数据
            game = self_play_game(self.model, num_simulations=max(30, NUM_SIMULATIONS // 4))
            self.replay_buffer.add_game(game)
            if len(self.replay_buffer) >= REPLAY_MIN_SIZE:
                self._train_step(i)

        print("[课程学习] 预训练完成, 切换到正式训练")

    def _train_step(self, iteration):
        """
        V2 训练步 — 修复 KL 正则 + Huber 损失
        """
        self.model.train()
        num_steps = max(10, min(100, len(self.replay_buffer) // LEARNER_BATCH_SIZE))

        total_p_loss = total_v_loss = total_kl = total_loss = 0.0

        for step in range(num_steps):
            beta = min(1.0, PRIORITY_BETA_START + self.total_steps / PRIORITY_BETA_FRAMES)
            sample = self.replay_buffer.sample(LEARNER_BATCH_SIZE, beta=beta)
            if sample is None:
                continue

            states, policies, values, weights, indices = sample
            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).to(self.device)
            weights_t = torch.from_numpy(weights).to(self.device)

            # 前向传播
            policy_logits, pred_values = self.model(states_t)

            # 策略损失: 交叉熵
            log_policy = F.log_softmax(policy_logits, dim=1)
            p_loss = -(policies_t * log_policy).sum(dim=1)
            p_loss = (p_loss * weights_t).mean()

            # 价值损失: Huber Loss
            v_loss = F.huber_loss(pred_values, values_t, reduction='none', delta=HUBER_DELTA)
            v_loss = (v_loss * weights_t).mean()

            # V3 修复: KL 正则 — 改为策略熵正则化, 鼓励探索
            # 之前的 KL(p_MCTS || p_model) 等价于交叉熵减去常数, 对训练无实质约束
            # 策略熵正则化: H(π) = -Σ π·log(π), 最小化负熵 = 鼓励策略更确定
            # 但加权重为正 → 实际上最大化熵 = 鼓励探索
            entropy_reg = torch.tensor(0.0, device=self.device)
            if KL_REG_WEIGHT > 0:
                # 计算当前策略的熵
                policy_probs = F.softmax(policy_logits, dim=1)
                entropy = -(policy_probs * log_policy).sum(dim=1).mean()
                # 负熵作为正则项 (最小化负熵 = 最大化熵 = 鼓励探索)
                entropy_reg = -entropy

            loss = POLICY_LOSS_WEIGHT * p_loss + VALUE_LOSS_WEIGHT * v_loss + KL_REG_WEIGHT * entropy_reg

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.total_steps += 1

            # V2: EMA 更新
            if USE_EMA and self.ema_model is not None:
                self.ema_model.update(self.model)

            # SWA 更新
            if USE_SWA and self.swa_model is not None and self.total_steps >= SWA_START_STEP:
                if self.total_steps % SWA_UPDATE_FREQ == 0:
                    self.swa_model.update_parameters(self.model)
                    self.swa_started = True

            # 更新优先级
            with torch.no_grad():
                td_errors = torch.abs(pred_values - values_t).cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            total_kl += entropy_reg.item()
            total_loss += loss.item()

        return {
            'policy_loss': total_p_loss / max(1, num_steps),
            'value_loss': total_v_loss / max(1, num_steps),
            'kl_reg': total_kl / max(1, num_steps),
            'total_loss': total_loss / max(1, num_steps),
            'lr': self.lr_scheduler.get_last_lr()[0]
        }

    def _champion_eval(self):
        """新旧模型对弈评估"""
        if self.champion_model is None:
            return 0.5

        wins = draws = losses = 0
        eval_sims = max(50, NUM_SIMULATIONS // 4)

        for game_idx in range(min(EVAL_GAMES, 10)):
            board = Board()
            new_is_black = (game_idx % 2 == 0)
            new_color = BLACK if new_is_black else WHITE

            mcts_new = MCTS(self.model, num_simulations=eval_sims, add_noise=False, temperature=0.0)
            mcts_champ = MCTS(self.champion_model, num_simulations=eval_sims, add_noise=False, temperature=0.0)

            while not board.game_over and board.move_count < MAX_MOVES:
                if board.current_player == new_color:
                    probs, _ = mcts_new.search(board)
                    action = np.argmax(probs)
                    mcts_new.advance(action)
                else:
                    probs, _ = mcts_champ.search(board)
                    action = np.argmax(probs)
                    mcts_champ.advance(action)

                board.place_stone(*board.index_to_move(action))

            if board.winner == 0:
                draws += 1
            elif board.winner == new_color:
                wins += 1
            else:
                losses += 1

        total = wins + draws + losses
        return (wins + 0.5 * draws) / max(1, total)

    def _simple_eval(self):
        """V5 修复: 与随机基线对弈评估 — 自博弈胜率恒≈50%无意义"""
        self.model.eval()
        wins = draws = losses = 0
        n_games = min(EVAL_GAMES, 6)

        for game_idx in range(n_games):
            board = Board()
            # 模型执黑/执白交替
            model_color = BLACK if game_idx % 2 == 0 else WHITE
            mcts = MCTS(self.model, num_simulations=max(50, NUM_SIMULATIONS // 4),
                        add_noise=False, temperature=0.0)

            while not board.game_over and board.move_count < MAX_MOVES:
                if board.current_player == model_color:
                    # 模型下棋
                    probs, _ = mcts.search(board)
                    action = np.argmax(probs)
                    mcts.advance(action)
                else:
                    # 随机基线: 从合法着法中随机选 (优先选邻居内的着法)
                    legal = board.get_legal_move_indices()
                    if legal:
                        action = legal[np.random.randint(len(legal))]
                    else:
                        break

                board.place_stone(*board.index_to_move(action))

            if board.winner == 0:
                draws += 1
            elif board.winner == model_color:
                wins += 1
            else:
                losses += 1

        total = wins + draws + losses
        wr = (wins + 0.5 * draws) / max(1, total)
        # ELO: 基于对随机基线胜率的估算 (随机≈0 ELO)
        if wr >= 0.99:
            elo = 1500
        elif wr <= 0.01:
            elo = 0
        else:
            elo = max(0, -400 * np.log10(1.0 / wr - 1) + 500)
        return elo

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'iteration': self.iteration,
            'total_steps': self.total_steps,
            'best_elo': self.best_elo,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'history': dict(self.history)
        }
        if USE_SWA and self.swa_model is not None:
            data['swa_state_dict'] = self.swa_model.state_dict()
        if USE_EMA and self.ema_model is not None:
            data['ema_shadow'] = self.ema_model.shadow
        if USE_OPPONENT_POOL and self.opponent_pool is not None:
            data['opponent_pool'] = self.opponent_pool
        torch.save(data, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        self.iteration = ckpt.get('iteration', 0) + 1
        self.total_steps = ckpt.get('total_steps', 0)
        self.best_elo = ckpt.get('best_elo', 0)
        if 'history' in ckpt:
            self.history = defaultdict(list, ckpt['history'])
        if USE_SWA and self.swa_model is not None and 'swa_state_dict' in ckpt:
            self.swa_model.load_state_dict(ckpt['swa_state_dict'])
        if USE_EMA and self.ema_model is not None and 'ema_shadow' in ckpt:
            self.ema_model.shadow = ckpt['ema_shadow']
        if USE_OPPONENT_POOL and 'opponent_pool' in ckpt:
            self.opponent_pool = ckpt['opponent_pool']
        if USE_CHAMPION_EVAL:
            self.champion_model = copy.deepcopy(self.model)
            self.champion_model.eval()
        print(f"  加载: {path} (迭代 {self.iteration}, ELO {self.best_elo})")

    def _save_history(self):
        path = os.path.join(CHECKPOINT_DIR, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)


def play_vs_model(model, human_color=BLACK, num_simulations=NUM_SIMULATIONS):
    """人机对弈"""
    board = Board()
    mcts = MCTS(model, num_simulations=num_simulations, add_noise=False, temperature=0.0)

    print("\n五子棋人机对弈 (V2)")
    print("输入: 行 列 (如: 7 7)")
    print(f"人类: {'●黑' if human_color == BLACK else '○白'}")
    print(board)

    while not board.game_over:
        if board.current_player == human_color:
            while True:
                try:
                    inp = input(f"\n落子 ({'●' if human_color == BLACK else '○'}): ")
                    r, c = map(int, inp.strip().split())
                    if board.is_legal(r, c):
                        break
                    print("非法着法")
                except (ValueError, IndexError):
                    print("格式: 行 列")
            board.place_stone(r, c)
            mcts.advance(board.get_move_index(r, c))
        else:
            print("\nAI思考中...")
            start = time.time()
            probs, value = mcts.search(board)
            t = time.time() - start
            action = np.argmax(probs)
            r, c = board.index_to_move(action)
            board.place_stone(r, c)
            mcts.advance(action)
            print(f"AI: ({r},{c}) 评估:{value:.3f} 用时:{t:.1f}s")
        print(board)

    if board.winner == 0:
        print("\n平局!")
    elif board.winner == human_color:
        print("\n你赢了!")
    else:
        print("\nAI赢了!")
