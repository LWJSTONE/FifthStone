"""
五子棋 AI 训练全局配置
======================
针对 Intel i7-12700H (14核20线程) CPU-only 场景的极致优化配置。
核心策略：小网络 + 多MCTS + 多进程并行 + Numba JIT
"""

import os
import multiprocessing

# ======================== 硬件配置 ========================
# i7-12700H: 6P+8E = 14核20线程
# 分配: (总线程-2) 个 Actor + 1 个 Learner + 1 个 Evaluator
TOTAL_THREADS = os.cpu_count() or 20
NUM_ACTORS = max(1, TOTAL_THREADS - 2)        # 自我对弈进程数
NUM_MCTS_THREADS = 1                           # 每个Actor内MCTS并行线程(受GIL限制，用多进程)
LEARNER_BATCH_SIZE = 256                       # 训练批量大小

# ======================== 棋盘配置 ========================
BOARD_SIZE = 15
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE         # 225
WIN_LENGTH = 5                                  # 五子连珠
HISTORY_LENGTH = 8                              # 输入历史步数

# 输入通道数: 当前棋手8步 + 对手8步 + 当前颜色1 = 17
INPUT_CHANNELS = HISTORY_LENGTH * 2 + 1

# ======================== 神经网络配置 ========================
# 极致轻量化：6个残差块，64通道，深度可分离卷积
# 在CPU上推理一局约 0.5-1ms，保证自我对弈吞吐量
NUM_RES_BLOCKS = 6
NUM_FILTERS = 64                                # 主通道数
SE_REDUCTION = 16                               # SE注意力压缩比
POLICY_CHANNELS = 2                             # 策略头中间通道
VALUE_HIDDEN = 64                               # 价值头全连接层

# ======================== MCTS 配置 ========================
NUM_SIMULATIONS = 400                           # 每步MCTS模拟次数
C_PUCT = 1.5                                   # PUCT探索常数
C_PUCT_BASE = 19652.0                          # PUCT基数(渐进式探索)
DIRICHLET_ALPHA = 0.3                          # Dirichlet噪声alpha
DIRICHLET_EPSILON = 0.25                       # 噪声混合比
VIRTUAL_LOSS = 3                                # 虚拟损失(并行搜索)
TEMPERATURE_THRESHOLD = 30                      # 前N步使用温度采样
INITIAL_TEMPERATURE = 1.0                       # 初始温度
MAX_TREE_SIZE = 500000                          # MCTS树最大节点数
USE_RAVE = True                                 # 启用RAVE加速
RAVE_EQUIV = 250                                # RAVE等价参数
USE_TRANSPOSITION = True                        # 启用转置表
MAX_MOVES = BOARD_SQUARES                       # 最大步数

# ======================== 自我对弈配置 ========================
NUM_GAMES_PER_ITER = NUM_ACTORS * 10            # 每轮迭代对局数
MAX_GAME_LENGTH = MAX_MOVES                     # 单局最大步数

# ======================== 训练配置 ========================
LEARNING_RATE = 0.01                            # 初始学习率
LR_WARMUP_STEPS = 200                           # 预热步数
LR_DECAY_STEPS = 100000                         # 余弦退火周期
WEIGHT_DECAY = 1e-4                             # L2正则化
MOMENTUM = 0.9                                  # SGD动量
GRAD_CLIP = 1.0                                 # 梯度裁剪
POLICY_LOSS_WEIGHT = 1.0                        # 策略损失权重
VALUE_LOSS_WEIGHT = 1.5                         # 价值损失权重(五子棋价值更关键)

# 经验回放
REPLAY_BUFFER_SIZE = 500000                     # 回放缓冲区大小
REPLAY_MIN_SIZE = 5000                          # 最小回放数量(开始训练阈值)
PRIORITY_ALPHA = 0.6                            # 优先经验回放alpha
PRIORITY_BETA_START = 0.4                       # 优先经验回放beta起始值
PRIORITY_BETA_FRAMES = 1000000                  # beta增长到1.0的帧数

# 数据增广: 8种对称(4旋转 × 2翻转)
NUM_SYMMETRIES = 8

# ======================== 训练调度 ========================
TOTAL_ITERATIONS = 200                          # 总训练迭代数
EVAL_INTERVAL = 5                               # 每N轮评估一次
EVAL_GAMES = 20                                 # 评估对局数
SAVE_INTERVAL = 5                               # 每N轮保存模型
CHECKPOINT_DIR = "checkpoints"                  # 检查点目录
BEST_MODEL_PATH = "checkpoints/best_model.pt"   # 最佳模型路径

# ======================== Numba JIT 配置 ========================
NUMBA_CACHE = True                              # 启用Numba缓存
NUMBA_PARALLEL = False                          # 禁用自动并行(GIL冲突)

# ======================== 模式编码 ========================
# 棋型: 连子数 + 开放端数 → 分值
# open_four(活四)=100000, half_four(冲四)=10000
# open_three(活三)=5000, half_three(眠三)=500
# open_two(活二)=200, half_two(眠二)=50
PATTERN_SCORES = {
    (5, 0): 1000000,    # 连五
    (5, 1): 1000000,    # 连五
    (5, 2): 1000000,    # 连五
    (4, 2): 100000,     # 活四
    (4, 1): 10000,      # 冲四
    (4, 0): 10000,      # 冲四(一端被堵)
    (3, 2): 5000,       # 活三
    (3, 1): 500,        # 眠三
    (3, 0): 50,         # 死三
    (2, 2): 200,        # 活二
    (2, 1): 50,         # 眠二
    (2, 0): 10,         # 死二
    (1, 2): 10,         # 活一
    (1, 1): 1,          # 眠一
}

# ======================== Zobrist 哈希 ========================
import numpy as np
def _init_zobrist():
    """初始化Zobrist哈希表: [位置, 颜色] -> 随机64位整数"""
    rng = np.random.RandomState(42)
    return rng.randint(0, 2**63, size=(BOARD_SQUARES, 2), dtype=np.int64)

ZOBRIST_TABLE = _init_zobrist()
ZOBRIST_TURN = np.int64(0x9E3779B97F4A7C15 & 0x7FFFFFFFFFFFFFFF)

# 初始化随机种子
np.random.seed(42)
