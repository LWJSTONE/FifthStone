"""
多进程自我对弈管线 (全面优化版)
===============================
优化:
  1. 数据质量过滤 — 过短/过长对局
  2. 异步 Actor-Learner (异步队列)
  3. 8对称增广
  4. 温度衰减
  5. VCT/Must-move 加速对弈
"""

import numpy as np
import torch
import multiprocessing as mp
import pickle
import time
from collections import deque

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS, MAX_GAME_LENGTH,
    NUM_GAMES_PER_ITER, TEMPERATURE_THRESHOLD, INITIAL_TEMPERATURE,
    NUM_SYMMETRIES, MAX_MOVES, REPLAY_BUFFER_SIZE, REPLAY_MIN_SIZE,
    LEARNER_BATCH_SIZE, NUM_ACTORS, NUM_SIMULATIONS,
    MIN_GAME_LENGTH, MAX_GAME_LENGTH_FILTER
)
from board import Board, BLACK, WHITE
from mcts import MCTS


class GameData:
    """单局自我对弈数据"""
    def __init__(self):
        self.states = []
        self.winner = 0
        self.length = 0

    def add_step(self, feature_planes, action_probs):
        self.states.append((feature_planes.copy(), action_probs.copy()))
        self.length += 1

    def set_winner(self, winner):
        self.winner = winner

    def is_valid(self):
        """数据质量过滤"""
        if self.length < MIN_GAME_LENGTH:
            return False
        if self.length > MAX_GAME_LENGTH_FILTER:
            return False
        return True

    def get_training_data(self):
        """生成训练样本 (含8对称增广)"""
        samples = []
        for i, (feature, policy) in enumerate(self.states):
            current_color = BLACK if i % 2 == 0 else WHITE
            if self.winner == 0:
                value = 0.0
            elif self.winner == current_color:
                value = 1.0
            else:
                value = -1.0

            symmetries = Board.get_symmetries(feature, policy)
            for sym_f, sym_p in symmetries:
                samples.append((sym_f, sym_p, value))
        return samples


class ReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def add_game(self, game_data):
        if not game_data.is_valid():
            return
        samples = game_data.get_training_data()
        for sample in samples:
            self.buffer.append(sample)
            self.priorities.append(self.max_priority)

    def add_games(self, games):
        for game in games:
            self.add_game(game)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** 0.6
        psum = probs.sum()
        if psum <= 0:
            return None
        probs /= psum
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-beta)
        wmax = weights.max()
        if wmax > 0:
            weights /= wmax

        states = np.zeros((batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policies = np.zeros((batch_size, BOARD_SQUARES), dtype=np.float32)
        values = np.zeros(batch_size, dtype=np.float32)
        for i, idx in enumerate(indices):
            state, policy, value = self.buffer[idx]
            states[i] = state
            policies[i] = policy
            values[i] = value

        return states, policies, values, weights.astype(np.float32), indices

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(error) + 1e-6
                self.max_priority = max(self.max_priority, self.priorities[idx])


def self_play_game(model, num_simulations=NUM_SIMULATIONS, add_noise=True):
    """执行一局自我对弈 (含VCT/必走检测加速)"""
    board = Board()
    game_data = GameData()
    mcts = MCTS(model, num_simulations=num_simulations, add_noise=add_noise)

    for step in range(MAX_GAME_LENGTH):
        temperature = INITIAL_TEMPERATURE if step < TEMPERATURE_THRESHOLD else 0.0
        mcts.temperature = temperature
        action_probs, root_value = mcts.search(board)

        feature = board.get_feature_planes()
        game_data.add_step(feature, action_probs)

        if step < TEMPERATURE_THRESHOLD and temperature > 0:
            action = np.random.choice(BOARD_SQUARES, p=action_probs)
        else:
            action = np.argmax(action_probs)

        r, c = board.index_to_move(action)
        if not board.place_stone(r, c):
            legal = board.get_legal_move_indices()
            if not legal:
                break
            action = legal[0]
            board.place_stone(*board.index_to_move(action))

        # 子树复用
        mcts.advance(action)

        if board.game_over:
            break

    game_data.set_winner(board.winner)
    return game_data


def actor_worker(worker_id, model_state_dict, num_games, result_queue,
                 num_simulations=NUM_SIMULATIONS):
    """Actor 工作进程"""
    from network import create_model
    model = create_model(device='cpu')
    model.load_state_dict(model_state_dict)

    games_played = 0
    while games_played < num_games:
        game_data = self_play_game(model, num_simulations=num_simulations, add_noise=True)
        result_queue.put(pickle.dumps(game_data))
        games_played += 1
        if games_played % 3 == 0:
            print(f"  [Actor-{worker_id}] {games_played}/{num_games} 局")


def generate_self_play_data(model, num_games=NUM_GAMES_PER_ITER, num_actors=NUM_ACTORS):
    """多进程并行生成自我对弈数据"""
    print(f"[SelfPlay] 启动 {num_actors} 个 Actor, 目标 {num_games} 局")
    buffer = ReplayBuffer()
    model_state_dict = model.state_dict()

    if num_actors <= 1:
        for i in range(num_games):
            game = self_play_game(model, num_simulations=NUM_SIMULATIONS)
            buffer.add_game(game)
            if (i + 1) % 5 == 0:
                valid_str = '有效' if game.is_valid() else '过滤'
                print(f"  [SelfPlay] {i+1}/{num_games} 局 "
                      f"({game.length}步, {'黑' if game.winner==1 else '白' if game.winner==2 else '平'}, {valid_str})")
        return buffer

    # 多进程模式
    games_per_actor = max(1, num_games // num_actors)
    processes = []
    result_queue = mp.Queue()

    for i in range(num_actors):
        actor_games = games_per_actor
        if i == num_actors - 1:
            actor_games = num_games - games_per_actor * (num_actors - 1)
        p = mp.Process(target=actor_worker,
                       args=(i, model_state_dict, actor_games, result_queue, NUM_SIMULATIONS))
        processes.append(p)

    for p in processes:
        p.start()

    games_collected = 0
    while games_collected < num_games:
        try:
            data = result_queue.get(timeout=120)
            game = pickle.loads(data)
            buffer.add_game(game)
            games_collected += 1
        except Exception:
            break

    for p in processes:
        p.join(timeout=30)

    print(f"[SelfPlay] 完成: {games_collected} 局, 缓冲区 {len(buffer)} 样本")
    return buffer
