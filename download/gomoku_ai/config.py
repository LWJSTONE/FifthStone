"""
五子棋 AI 训练全局配置 (全面优化版)
====================================
针对 Intel i7-12700H (14核20线程) CPU-only 场景的极致优化配置。
核心策略：小网络 + Gumbel MCTS + VCT战术 + 批量推理 + 多进程并行 + Numba JIT
"""

import os
import numpy as np

# ======================== 硬件配置 ========================
TOTAL_THREADS = os.cpu_count() or 20
NUM_ACTORS = max(1, TOTAL_THREADS - 2)
LEARNER_BATCH_SIZE = 256

# ======================== 棋盘配置 ========================
BOARD_SIZE = 15
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE
WIN_LENGTH = 5
HISTORY_LENGTH = 8
MAX_MOVES = BOARD_SQUARES
MAX_GAME_LENGTH = MAX_MOVES

# 输入通道数: 历史当前8步 + 历史对手8步 + 颜色1 + 领域知识2 = 19
INPUT_CHANNELS = HISTORY_LENGTH * 2 + 1 + 2  # +2 for pattern channels

# ======================== 神经网络配置 ========================
NUM_RES_BLOCKS = 6
NUM_FILTERS = 64
SE_REDUCTION = 16
POLICY_CHANNELS = 2
VALUE_HIDDEN = 64

# 推理优化
USE_TORCHSCRIPT = True          # TorchScript 编译
USE_INT8_QUANT = True           # INT8 动态量化
USE_BN_FUSE = True              # BN 融合 (推理时)
USE_NHWC = True                 # 通道最后内存格式

# ======================== MCTS 配置 ========================
NUM_SIMULATIONS = 400
C_PUCT = 1.5
C_PUCT_BASE = 19652.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
VIRTUAL_LOSS = 3
TEMPERATURE_THRESHOLD = 30
INITIAL_TEMPERATURE = 1.0
MAX_TREE_SIZE = 500000
MAX_MOVES = BOARD_SQUARES

# 批量推理
MCTS_BATCH_SIZE = 8             # 每次批量推理的叶节点数

# FPU (First Play Urgency)
USE_FPU = True
FPU_VALUE = -0.5                # 未访问节点初始Q值

# 模式注入
USE_PATTERN_INJECTION = True
PATTERN_INJECTION_WEIGHT = 0.3  # 模式先验权重 (0-1, 0=纯网络, 1=纯模式)

# Gumbel MCTS
USE_GUMBEL_MCTS = True
GUMBEL_TOPK = 16                # Gumbel top-k 候选数

# 对称感知 MCTS
USE_SYMMETRY_MCTS = True

# 子树复用
USE_SUBTREE_REUSE = True

# 必走着法检测 (跳过MCTS)
USE_MUST_MOVE = True

# VCT/VCF 搜索
USE_VCT = True
VCT_DEPTH_LIMIT = 12            # VCT 最大深度
VCF_DEPTH_LIMIT = 20            # VCF 最大深度

# RAVE
USE_RAVE = True
RAVE_EQUIV = 250

# 转置表
USE_TRANSPOSITION = True

# 动态模拟次数
USE_DYNAMIC_SIMS = True
MIN_SIMULATIONS = 50
MAX_SIMULATIONS = 800
SIM_ENTROPY_SCALE = 200.0       # 策略熵到模拟次数的缩放

# ======================== 自我对弈配置 ========================
NUM_GAMES_PER_ITER = NUM_ACTORS * 10

# 数据质量过滤
MIN_GAME_LENGTH = 8             # 过滤过短对局
MAX_GAME_LENGTH_FILTER = 180    # 过滤过长对局(双方都不会赢)

# ======================== 训练配置 ========================
LEARNING_RATE = 0.01
LR_WARMUP_STEPS = 200
LR_DECAY_STEPS = 100000
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP = 1.0
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.5
KL_REG_WEIGHT = 0.01           # KL散度正则权重(防策略突变)
HUBER_DELTA = 1.0              # 价值损失Huber参数

# SWA (随机权重平均)
USE_SWA = True
SWA_START_STEP = 5000
SWA_UPDATE_FREQ = 100

# 经验回放
REPLAY_BUFFER_SIZE = 500000
REPLAY_MIN_SIZE = 3000
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 1000000

# 数据增广
NUM_SYMMETRIES = 8

# 课程学习
USE_CURRICULUM = True           # 启用课程学习(9x9→15x15)
CURRICULUM_SMALL_SIZE = 9       # 小棋盘尺寸
CURRICULUM_SMALL_ITERS = 15     # 小棋盘训练轮数

# ======================== 训练调度 ========================
TOTAL_ITERATIONS = 200
EVAL_INTERVAL = 5
EVAL_GAMES = 20
SAVE_INTERVAL = 5
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "checkpoints/best_model.pt"

# 新旧模型对弈评估
USE_CHAMPION_EVAL = True
CHAMPION_WIN_RATE = 0.55        # 新模型必须达到的胜率才替换旧模型

# ======================== Numba JIT 配置 ========================
NUMBA_CACHE = True

# ======================== 邻居表预计算 ========================
NEIGHBOR_RADIUS = 2             # 合法着法搜索半径

def _init_neighbor_table():
    """预计算每个位置在指定半径内的邻居列表"""
    table = []
    for pos in range(BOARD_SQUARES):
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        neighbors = []
        for dr in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            for dc in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    neighbors.append(nr * BOARD_SIZE + nc)
        table.append(np.array(neighbors, dtype=np.int32))
    return table

NEIGHBOR_TABLE = _init_neighbor_table()

# ======================== Zobrist 哈希 ========================
def _init_zobrist():
    rng = np.random.RandomState(42)
    return rng.randint(0, 2**63, size=(BOARD_SQUARES, 2), dtype=np.int64)

ZOBRIST_TABLE = _init_zobrist()
ZOBRIST_TURN = np.int64(0x9E3779B97F4A7C15 & 0x7FFFFFFFFFFFFFFF)

np.random.seed(42)
