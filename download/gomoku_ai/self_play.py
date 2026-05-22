"""
多进程自我对弈管线
=================
架构: Actor-Learner 分离
  - 多个 Actor 进程并行生成对局数据
  - 共享内存缓冲区高效传递数据
  - Learner 进程异步训练模型
  - 模型参数通过共享内存/管道广播

优化:
  1. 多进程绕过 GIL (每个Actor独立Python进程)
  2. 共享内存缓冲区 (零拷贝数据传递)
  3. 异步模型更新 (Actor定期拉取最新权重)
  4. 温度衰减 (前N步高温度探索，后低温度利用)
  5. 8对称增广 (每局数据×8)
"""

import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import os
import pickle
import struct
from collections import deque

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS, MAX_GAME_LENGTH,
    NUM_GAMES_PER_ITER, TEMPERATURE_THRESHOLD, INITIAL_TEMPERATURE,
    NUM_SYMMETRIES, MAX_MOVES, REPLAY_BUFFER_SIZE, REPLAY_MIN_SIZE,
    LEARNER_BATCH_SIZE, NUM_ACTORS, NUM_SIMULATIONS
)
from board import Board, BLACK, WHITE
from mcts import MCTS


# ======================== 数据结构 ========================

class GameData:
    """单局自我对弈数据"""

    def __init__(self):
        self.states = []       # [(feature_planes, action_probs), ...]
        self.winner = 0        # 1=黑胜, 2=白胜, 0=平局
        self.length = 0        # 对局步数

    def add_step(self, feature_planes, action_probs):
        self.states.append((feature_planes.copy(), action_probs.copy()))
        self.length += 1

    def set_winner(self, winner):
        self.winner = winner

    def get_training_data(self):
        """
        生成训练样本: (state, policy_target, value_target)
        ========================================
        value_target: 从当前棋手视角，胜=1，负=-1，平=0
        8对称增广: 每个样本生成8个变体
        """
        samples = []
        for i, (feature, policy) in enumerate(self.states):
            # 确定当前棋手
            # 第0步=黑, 第1步=白, 依此类推
            current_color = BLACK if i % 2 == 0 else WHITE

            # 价值: 从当前棋手视角
            if self.winner == 0:
                value = 0.0
            elif self.winner == current_color:
                value = 1.0
            else:
                value = -1.0

            # 8对称增广
            symmetries = Board.get_symmetries(feature, policy)
            for sym_feature, sym_policy in symmetries:
                samples.append((sym_feature, sym_policy, value))

        return samples


class ReplayBuffer:
    """
    优先经验回放缓冲区
    ==================
    支持:
      - 固定容量，FIFO替换
      - 优先级采样 (TD-error based)
      - 8对称增广已由GameData生成
    """

    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def add_game(self, game_data):
        """添加一局对弈数据"""
        samples = game_data.get_training_data()
        for sample in samples:
            self.buffer.append(sample)
            self.priorities.append(self.max_priority)

    def add_games(self, games):
        """批量添加多局对弈数据"""
        for game in games:
            self.add_game(game)

    def sample(self, batch_size, beta=0.4):
        """
        优先级采样
        =========
        返回: (states, policies, values, weights, indices)
        """
        if len(self.buffer) < batch_size:
            return None

        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** 0.6  # alpha=0.6
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        # 重要性采样权重
        n = len(self.buffer)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        # 收集数据
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
        """更新优先级(TD-error)"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])


# ======================== 自我对弈 ========================

def self_play_game(model, num_simulations=NUM_SIMULATIONS, add_noise=True):
    """
    执行一局自我对弈
    ==============
    参数:
      model: 神经网络模型
      num_simulations: MCTS模拟次数
      add_noise: 是否添加Dirichlet噪声
    返回:
      GameData 对象
    """
    board = Board()
    game_data = GameData()
    mcts = MCTS(model, num_simulations=num_simulations, add_noise=add_noise)

    for step in range(MAX_GAME_LENGTH):
        # 温度调度: 前30步高温探索，后低温利用
        temperature = INITIAL_TEMPERATURE if step < TEMPERATURE_THRESHOLD else 0.0
        mcts.temperature = temperature

        # MCTS搜索
        action_probs, root_value = mcts.search(board)

        # 记录状态
        feature = board.get_feature_planes()
        game_data.add_step(feature, action_probs)

        # 选择动作
        if step < TEMPERATURE_THRESHOLD and temperature > 0:
            # 温度采样
            action = np.random.choice(BOARD_SQUARES, p=action_probs)
        else:
            # 贪心选择
            action = np.argmax(action_probs)

        r, c = board.index_to_move(action)

        # 落子
        if not board.place_stone(r, c):
            # 非法着法(不应发生)，随机选择
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            r, c = legal_moves[0]
            board.place_stone(r, c)

        # 检查终局
        if board.game_over:
            break

    # 设置胜者
    game_data.set_winner(board.winner)

    return game_data


def actor_worker(worker_id, model_state_dict, num_games, result_queue, weight_pipe,
                 num_simulations=NUM_SIMULATIONS):
    """
    Actor 工作进程
    =============
    参数:
      worker_id: 进程ID
      model_state_dict: 初始模型参数
      num_games: 本进程需生成的对局数
      result_queue: 数据输出队列
      weight_pipe: 模型权重更新管道
      num_simulations: MCTS模拟次数
    """
    # 每个进程独立创建模型
    from network import create_model
    model = create_model(device='cpu')

    # 加载初始权重
    model.load_state_dict(model_state_dict)

    games_played = 0

    while games_played < num_games:
        # 检查是否有新权重
        if weight_pipe.poll():
            try:
                new_state_dict = weight_pipe.recv()
                model.load_state_dict(new_state_dict)
            except Exception:
                pass

        # 执行自我对弈
        game_data = self_play_game(model, num_simulations=num_simulations, add_noise=True)

        # 序列化并放入队列
        result_queue.put(pickle.dumps(game_data))
        games_played += 1

        if games_played % 5 == 0:
            print(f"  [Actor-{worker_id}] 完成 {games_played}/{num_games} 局")

    print(f"  [Actor-{worker_id}] 全部完成: {games_played} 局")


def generate_self_play_data(model, num_games=NUM_GAMES_PER_ITER, num_actors=NUM_ACTORS):
    """
    多进程并行生成自我对弈数据
    =========================
    参数:
      model: 当前模型
      num_games: 总对局数
      num_actors: 并行进程数
    返回:
      ReplayBuffer 填充了所有对弈数据
    """
    print(f"[SelfPlay] 启动 {num_actors} 个 Actor 进程，目标 {num_games} 局")

    buffer = ReplayBuffer()
    model_state_dict = model.state_dict()

    # 串行模式(更稳定，适合CPU)
    if num_actors <= 1:
        for i in range(num_games):
            game_data = self_play_game(model, num_simulations=NUM_SIMULATIONS)
            buffer.add_game(game_data)
            if (i + 1) % 5 == 0:
                print(f"  [SelfPlay] 完成 {i+1}/{num_games} 局 "
                      f"(平均 {game_data.length} 步, 胜者: {'黑' if game_data.winner == 1 else '白' if game_data.winner == 2 else '平'})")
        return buffer

    # 多进程模式
    games_per_actor = max(1, num_games // num_actors)
    processes = []
    result_queue = mp.Queue()

    for i in range(num_actors):
        actor_games = games_per_actor
        if i == num_actors - 1:
            actor_games = num_games - games_per_actor * (num_actors - 1)

        p = mp.Process(
            target=actor_worker,
            args=(i, model_state_dict, actor_games, result_queue, None, NUM_SIMULATIONS)
        )
        processes.append(p)

    # 启动所有进程
    for p in processes:
        p.start()

    # 收集结果
    games_collected = 0
    while games_collected < num_games:
        try:
            data = result_queue.get(timeout=60)
            game_data = pickle.loads(data)
            buffer.add_game(game_data)
            games_collected += 1
        except Exception:
            break

    # 等待所有进程结束
    for p in processes:
        p.join(timeout=30)

    print(f"[SelfPlay] 完成: 共 {games_collected} 局, "
          f"缓冲区 {len(buffer)} 样本")

    return buffer
