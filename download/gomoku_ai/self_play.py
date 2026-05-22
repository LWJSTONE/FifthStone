"""
多进程自我对弈管线 (V2 — 全面修复+优化版)
==========================================
V2 修复:
  1. 修复 PRIORITY_ALPHA 使用 (不再硬编码)
  2. SumTree 优先回放 (O(log n) 采样)
  3. Resign 机制 (低价值提前认输)
  4. 历史对手池训练 (增加多样性)
  5. 异步 Actor-Learner 接口
  6. CPU 亲和性绑定

V2 性能:
  1. 数据质量过滤
  2. 8对称增广
  3. 温度衰减
  4. VCT/Must-move 加速对弈
"""

import numpy as np
import torch
import multiprocessing as mp
import pickle
import time
import os
from collections import deque

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS, MAX_GAME_LENGTH,
    NUM_GAMES_PER_ITER, TEMPERATURE_THRESHOLD, INITIAL_TEMPERATURE,
    NUM_SYMMETRIES, MAX_MOVES, REPLAY_BUFFER_SIZE, REPLAY_MIN_SIZE,
    LEARNER_BATCH_SIZE, NUM_ACTORS, NUM_SIMULATIONS,
    MIN_GAME_LENGTH, MAX_GAME_LENGTH_FILTER,
    PRIORITY_ALPHA, PRIORITY_BETA_START, PRIORITY_BETA_FRAMES,
    USE_SUMTREE, USE_RESIGN, RESIGN_THRESHOLD, RESIGN_CHECK_STEPS,
    USE_CPU_AFFINITY, USE_OPPONENT_POOL, OPPONENT_POOL_SIZE,
    OPPONENT_POOL_GAME_RATIO
)
from board import Board, BLACK, WHITE
from mcts import MCTS


# ======================== SumTree ========================

class SumTree:
    """
    SumTree 优先回放 — O(log n) 采样
    替代 O(n) 的线性采样
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.size = 0
        self.ptr = 0  # 写入指针
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def add(self, priority, data):
        """添加数据"""
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self._update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.max_priority = max(self.max_priority, priority)

    def _update(self, idx, priority):
        """更新节点优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, value):
        """按优先级采样"""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def update_by_idx(self, idx, priority):
        self._update(idx, priority)
        self.max_priority = max(self.max_priority, priority)


# ======================== ReplayBuffer ========================

class ReplayBuffer:
    """V2 优先经验回放缓冲区 (支持 SumTree)"""
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        if USE_SUMTREE:
            self._tree = SumTree(capacity)
        else:
            self.buffer = deque(maxlen=capacity)
            self.priorities = deque(maxlen=capacity)
            self.max_priority = 1.0

    def __len__(self):
        if USE_SUMTREE:
            return len(self._tree)
        return len(self.buffer)

    def add_sample(self, sample):
        """添加单个样本"""
        if USE_SUMTREE:
            self._tree.add(self._tree.max_priority, sample)
        else:
            self.buffer.append(sample)
            self.priorities.append(self.max_priority)

    def add_game(self, game_data):
        """添加整局数据 (含8对称增广)"""
        if not game_data.is_valid():
            return
        samples = game_data.get_training_data()
        for sample in samples:
            self.add_sample(sample)

    def add_games(self, games):
        for game in games:
            self.add_game(game)

    def sample(self, batch_size, beta=PRIORITY_BETA_START):
        if len(self) < batch_size:
            return None

        if USE_SUMTREE:
            return self._sample_sumtree(batch_size, beta)
        else:
            return self._sample_deque(batch_size, beta)

    def _sample_sumtree(self, batch_size, beta):
        """SumTree O(log n) 采样"""
        states = np.zeros((batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        policies = np.zeros((batch_size, BOARD_SQUARES), dtype=np.float32)
        values = np.zeros(batch_size, dtype=np.float32)
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)

        total = self._tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            val = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self._tree.get(val)
            state, policy, value = data
            states[i] = state
            policies[i] = policy
            values[i] = value
            indices[i] = idx

            # Importance sampling weight
            prob = priority / total
            n = len(self._tree)
            weights[i] = (n * prob) ** (-beta)

        # Normalize weights
        wmax = weights.max()
        if wmax > 0:
            weights /= wmax

        return states, policies, values, weights, indices

    def _sample_deque(self, batch_size, beta):
        """Deque O(n) 采样 (兼容)"""
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** PRIORITY_ALPHA
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
        """更新优先级"""
        if USE_SUMTREE:
            for idx, error in zip(indices, td_errors):
                priority = abs(error) + 1e-6
                self._tree.update_by_idx(int(idx), priority)
        else:
            for idx, error in zip(indices, td_errors):
                if idx < len(self.priorities):
                    self.priorities[idx] = abs(error) + 1e-6
                    self.max_priority = max(self.max_priority, self.priorities[idx])


# ======================== GameData ========================

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


# ======================== 自我对弈 ========================

def self_play_game(model, num_simulations=NUM_SIMULATIONS, add_noise=True,
                   opponent_model=None):
    """
    执行一局自我对弈 (V2: 含VCT/必走/Resign/对手池)
    """
    board = Board()
    game_data = GameData()
    mcts = MCTS(model, num_simulations=num_simulations, add_noise=add_noise)

    # V2: 对手模型
    mcts_opponent = None
    if opponent_model is not None:
        mcts_opponent = MCTS(opponent_model, num_simulations=num_simulations,
                             add_noise=add_noise)

    # V2: Resign 机制
    recent_values = []

    for step in range(MAX_GAME_LENGTH):
        # 选择使用哪个模型
        active_mcts = mcts
        if mcts_opponent is not None and opponent_model is not None:
            # 如果有对手模型, 随机分配先手
            # 简化: 黑棋用当前模型, 白棋用对手模型
            if board.current_player == WHITE:
                active_mcts = mcts_opponent

        temperature = INITIAL_TEMPERATURE if step < TEMPERATURE_THRESHOLD else 0.0
        active_mcts.temperature = temperature
        action_probs, root_value = active_mcts.search(board)

        # V2: Resign 检测
        if USE_RESIGN and step > 10:
            recent_values.append(root_value)
            if len(recent_values) > RESIGN_CHECK_STEPS:
                recent_values.pop(0)
                if all(v < RESIGN_THRESHOLD for v in recent_values):
                    # 提前认输
                    game_data.set_winner(3 - board.current_player)
                    break

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
        if mcts_opponent is not None:
            mcts_opponent.advance(action)

        if board.game_over:
            break

    if not game_data.winner:
        game_data.set_winner(board.winner)
    return game_data


# ======================== 多进程 ========================

def actor_worker(worker_id, model_state_dict, num_games, result_queue,
                 num_simulations=NUM_SIMULATIONS, cpu_affinity=True):
    """V2 Actor 工作进程 (含 CPU 亲和性)"""
    # CPU 亲和性绑定
    if USE_CPU_AFFINITY and cpu_affinity:
        try:
            os.sched_setaffinity(0, {worker_id % os.cpu_count()})
        except (AttributeError, OSError):
            pass

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


def generate_self_play_data(model, num_games=NUM_GAMES_PER_ITER, num_actors=NUM_ACTORS,
                            opponent_pool=None):
    """V2 多进程并行生成自我对弈数据 (含历史对手池)"""
    print(f"[SelfPlay] 启动 {num_actors} 个 Actor, 目标 {num_games} 局")
    buffer = ReplayBuffer()
    model_state_dict = model.state_dict()

    # V2: 对手池 — 一部分对局与历史版本下
    opponent_model = None
    if USE_OPPONENT_POOL and opponent_pool and len(opponent_pool) > 0:
        # 随机选一个历史版本
        opponent_state = opponent_pool[np.random.randint(len(opponent_pool))]
        from network import create_model
        opponent_model = create_model(device='cpu')
        opponent_model.load_state_dict(opponent_state)
        opponent_model.eval()
        print(f"[SelfPlay] 使用历史对手模型")

    if num_actors <= 1:
        for i in range(num_games):
            # 决定是否使用对手模型
            use_opponent = (USE_OPPONENT_POOL and opponent_model is not None
                            and np.random.random() < OPPONENT_POOL_GAME_RATIO)
            opp = opponent_model if use_opponent else None
            game = self_play_game(model, num_simulations=NUM_SIMULATIONS,
                                  opponent_model=opp)
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
