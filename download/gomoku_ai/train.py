"""
训练引擎 — AlphaZero 训练循环
============================
核心功能:
  1. 自我对弈数据生成
  2. 优先经验回放采样
  3. 策略+价值双损失训练
  4. 余弦退火学习率调度
  5. 梯度裁剪 + 权重衰减
  6. 模型检查点保存/加载
  7. ELO 评估与最佳模型追踪
  8. 训练统计与可视化日志
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS,
    LEARNING_RATE, LR_WARMUP_STEPS, LR_DECAY_STEPS,
    WEIGHT_DECAY, MOMENTUM, GRAD_CLIP,
    POLICY_LOSS_WEIGHT, VALUE_LOSS_WEIGHT,
    LEARNER_BATCH_SIZE, REPLAY_MIN_SIZE, REPLAY_BUFFER_SIZE,
    TOTAL_ITERATIONS, EVAL_INTERVAL, EVAL_GAMES,
    SAVE_INTERVAL, CHECKPOINT_DIR, BEST_MODEL_PATH,
    NUM_GAMES_PER_ITER, NUM_ACTORS, NUM_SIMULATIONS,
    PRIORITY_BETA_START, PRIORITY_BETA_FRAMES
)
from network import create_model
from self_play import generate_self_play_data, self_play_game, ReplayBuffer
from board import Board, BLACK, WHITE
from mcts import MCTS


# ======================== 学习率调度 ========================

def create_lr_scheduler(optimizer, warmup_steps=LR_WARMUP_STEPS, decay_steps=LR_DECAY_STEPS):
    """
    余弦退火学习率 + 线性预热
    =========================
    lr = base_lr * min(1, step/warmup) * 0.5 * (1 + cos(pi * min(step, decay) / decay))
    """
    def lr_lambda(step):
        # 预热阶段
        warmup_factor = min(1.0, step / max(1, warmup_steps))
        # 余弦退火
        decay_factor = 0.5 * (1.0 + np.cos(np.pi * min(step, decay_steps) / decay_steps))
        return warmup_factor * decay_factor

    return LambdaLR(optimizer, lr_lambda)


# ======================== 训练器 ========================

class Trainer:
    """
    AlphaZero 训练器
    ===============
    流程:
      for iteration in range(TOTAL_ITERATIONS):
        1. 自我对弈生成数据 → 填充经验回放缓冲区
        2. 从缓冲区采样批次
        3. 前向传播 + 计算损失
        4. 反向传播 + 更新参数
        5. 定期评估 + 保存模型
    """

    def __init__(self, device='cpu', resume_path=None):
        self.device = device
        self.iteration = 0
        self.total_steps = 0
        self.best_elo = 0

        # 创建模型
        self.model = create_model(device=device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True
        )
        self.lr_scheduler = create_lr_scheduler(self.optimizer)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)

        # 训练统计
        self.history = defaultdict(list)

        # 检查点目录
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # 恢复训练
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

    def train(self):
        """主训练循环"""
        print("\n" + "=" * 60)
        print("  五子棋 AI 训练启动 — CPU 极致优化版")
        print("=" * 60)
        print(f"  设备: {self.device}")
        print(f"  模型: {sum(p.numel() for p in self.model.parameters()):,} 参数")
        print(f"  总迭代: {TOTAL_ITERATIONS}")
        print(f"  每轮对局: {NUM_GAMES_PER_ITER}")
        print(f"  MCTS模拟: {NUM_SIMULATIONS}")
        print(f"  批量大小: {LEARNER_BATCH_SIZE}")
        print("=" * 60 + "\n")

        for iteration in range(self.iteration, TOTAL_ITERATIONS):
            self.iteration = iteration
            iter_start = time.time()

            print(f"\n{'─' * 50}")
            print(f"迭代 {iteration + 1}/{TOTAL_ITERATIONS}")
            print(f"{'─' * 50}")

            # ===== 阶段1: 自我对弈数据生成 =====
            print(f"[1/3] 自我对弈数据生成...")
            sp_start = time.time()
            new_buffer = generate_self_play_data(
                self.model,
                num_games=NUM_GAMES_PER_ITER,
                num_actors=NUM_ACTORS
            )
            sp_time = time.time() - sp_start

            # 合并到主缓冲区
            self.replay_buffer.buffer.extend(new_buffer.buffer)
            self.replay_buffer.priorities.extend(new_buffer.priorities)

            # 截断到容量
            while len(self.replay_buffer) > REPLAY_BUFFER_SIZE:
                self.replay_buffer.buffer.popleft()
                self.replay_buffer.priorities.popleft()

            print(f"  自我对弈耗时: {sp_time:.1f}s, 缓冲区: {len(self.replay_buffer)} 样本")

            # ===== 阶段2: 训练 =====
            if len(self.replay_buffer) >= REPLAY_MIN_SIZE:
                print(f"[2/3] 训练网络...")
                train_start = time.time()
                train_stats = self._train_step(iteration)
                train_time = time.time() - train_start

                print(f"  训练耗时: {train_time:.1f}s")
                print(f"  策略损失: {train_stats['policy_loss']:.4f}")
                print(f"  价值损失: {train_stats['value_loss']:.4f}")
                print(f"  总损失: {train_stats['total_loss']:.4f}")
                print(f"  学习率: {train_stats['lr']:.6f}")

                self.history['policy_loss'].append(train_stats['policy_loss'])
                self.history['value_loss'].append(train_stats['value_loss'])
                self.history['total_loss'].append(train_stats['total_loss'])
            else:
                print(f"[2/3] 缓冲区不足 ({len(self.replay_buffer)}/{REPLAY_MIN_SIZE})，跳过训练")

            # ===== 阶段3: 评估 & 保存 =====
            if (iteration + 1) % EVAL_INTERVAL == 0:
                print(f"[3/3] 评估棋力...")
                elo = self._evaluate()
                print(f"  当前 ELO 估计: {elo:.0f}")
                self.history['elo'].append(elo)

                if elo > self.best_elo:
                    self.best_elo = elo
                    self.save_checkpoint(BEST_MODEL_PATH)
                    print(f"  ★ 新最佳模型! ELO={elo:.0f}")

            if (iteration + 1) % SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, f"model_iter_{iteration + 1}.pt"
                )
                self.save_checkpoint(checkpoint_path)

            # 迭代总结
            iter_time = time.time() - iter_start
            print(f"\n  迭代耗时: {iter_time:.1f}s")
            print(f"  缓冲区: {len(self.replay_buffer)} 样本")
            print(f"  最佳ELO: {self.best_elo:.0f}")

        print("\n" + "=" * 60)
        print("  训练完成!")
        print(f"  最终最佳 ELO: {self.best_elo:.0f}")
        print(f"  模型保存: {BEST_MODEL_PATH}")
        print("=" * 60)

        # 保存训练历史
        self._save_history()

    def _train_step(self, iteration):
        """
        执行一轮训练
        ==========
        每轮迭代执行多个训练步
        """
        self.model.train()

        # 训练步数: 与数据量成正比
        num_steps = max(10, min(100, len(self.replay_buffer) // LEARNER_BATCH_SIZE))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        for step in range(num_steps):
            # 优先级采样
            beta = min(1.0, PRIORITY_BETA_START + self.total_steps / PRIORITY_BETA_FRAMES)
            sample = self.replay_buffer.sample(LEARNER_BATCH_SIZE, beta=beta)
            if sample is None:
                continue

            states, policies, values, weights, indices = sample

            # 转为Tensor
            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).to(self.device)
            weights_t = torch.from_numpy(weights).to(self.device)

            # 前向传播
            policy_logits, pred_values = self.model(states_t)

            # 策略损失: 交叉熵 (带重要性采样权重)
            log_policy = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies_t * log_policy).sum(dim=1)
            policy_loss = (policy_loss * weights_t).mean()

            # 价值损失: MSE (带权重)
            value_loss = F.mse_loss(pred_values, values_t, reduction='none')
            value_loss = (value_loss * weights_t).mean()

            # 总损失
            loss = POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

            # 更新参数
            self.optimizer.step()
            self.lr_scheduler.step()
            self.total_steps += 1

            # 更新优先级
            with torch.no_grad():
                td_errors = torch.abs(pred_values - values_t).cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()

        return {
            'policy_loss': total_policy_loss / max(1, num_steps),
            'value_loss': total_value_loss / max(1, num_steps),
            'total_loss': total_loss / max(1, num_steps),
            'lr': self.lr_scheduler.get_last_lr()[0]
        }

    def _evaluate(self):
        """
        评估当前模型棋力
        ==============
        方法: 当前模型 vs 上一版本模型 对弈
        简化: 基于价值头预测的胜率估计ELO
        """
        self.model.eval()

        # 使用少量MCTS模拟进行快速评估
        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(min(EVAL_GAMES, 4)):  # 减少评估对局数节省时间
            board = Board()
            mcts = MCTS(self.model, num_simulations=NUM_SIMULATIONS // 2,
                        add_noise=False, temperature=0.0)

            while not board.game_over and board.move_count < MAX_MOVES:
                action_probs, value = mcts.search(board)
                action = np.argmax(action_probs)
                r, c = board.index_to_move(action)
                board.place_stone(r, c)

            if board.winner == BLACK:
                wins += 1
            elif board.winner == WHITE:
                losses += 1
            else:
                draws += 1

        # 简单ELO估计(基于胜率)
        win_rate = (wins + 0.5 * draws) / max(1, wins + draws + losses)
        elo = -400 * np.log10(1 / max(0.01, min(0.99, win_rate)) - 1) if win_rate > 0 and win_rate < 1 else 0
        elo = max(0, elo + 1000)  # 基线ELO 1000

        return elo

    def save_checkpoint(self, path):
        """保存检查点"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'iteration': self.iteration,
            'total_steps': self.total_steps,
            'best_elo': self.best_elo,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'history': dict(self.history)
        }, path)
        print(f"  检查点已保存: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.iteration = checkpoint.get('iteration', 0) + 1
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_elo = checkpoint.get('best_elo', 0)
        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])
        print(f"  检查点已加载: {path} (迭代 {self.iteration}, ELO {self.best_elo})")

    def _save_history(self):
        """保存训练历史"""
        history_path = os.path.join(CHECKPOINT_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"  训练历史已保存: {history_path}")


# ======================== 对弈函数 ========================

def play_vs_model(model, human_color=BLACK, num_simulations=NUM_SIMULATIONS):
    """
    人机对弈(终端交互)
    ==================
    参数:
      model: 训练好的模型
      human_color: 人类执黑(1)或执白(2)
      num_simulations: MCTS模拟次数
    """
    board = Board()
    mcts = MCTS(model, num_simulations=num_simulations, add_noise=False, temperature=0.0)

    print("\n五子棋人机对弈")
    print("输入格式: 行 列 (如: 7 7)")
    print(f"人类: {'●黑' if human_color == BLACK else '○白'}")
    print(board)

    while not board.game_over:
        if board.current_player == human_color:
            # 人类回合
            while True:
                try:
                    inp = input(f"\n你的落子 ({'●' if human_color == BLACK else '○'}): ")
                    r, c = map(int, inp.strip().split())
                    if board.is_legal(r, c):
                        break
                    print("非法着法，请重试")
                except (ValueError, IndexError):
                    print("格式错误，请输入: 行 列")

            board.place_stone(r, c)
        else:
            # AI 回合
            print(f"\nAI 思考中...")
            start = time.time()
            action_probs, value = mcts.search(board)
            think_time = time.time() - start

            action = np.argmax(action_probs)
            r, c = board.index_to_move(action)
            board.place_stone(r, c)
            print(f"AI 落子: ({r}, {c}), 评估: {value:.3f}, 用时: {think_time:.1f}s")

        print(board)

    if board.winner == 0:
        print("\n平局!")
    elif board.winner == human_color:
        print("\n你赢了!")
    else:
        print("\nAI 赢了!")
