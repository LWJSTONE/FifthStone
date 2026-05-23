"""
五子棋 AI 训练全局配置 (V2 — 全面修复+优化版)
================================================
针对 Intel i7-12700H (14核20线程) CPU-only 场景的极致优化配置。
核心策略: 小网络 + Gumbel MCTS + VCT战术 + 批量推理 + 多进程并行 + Numba JIT

V2 修复:
  - 删除重复 MAX_MOVES 定义
  - 所有配置项都有对应消费方
  - 新增 Node Pool / Root Parallel / Undo-MCTS / ONNX 等配置
"""

import os
import numpy as np

# ======================== 硬件配置 ========================
TOTAL_THREADS = os.cpu_count() or 20
NUM_ACTORS = max(1, min(TOTAL_THREADS - 2, 12))  # 最多12个Actor
LEARNER_BATCH_SIZE = 256

# CPU 亲和性绑定
USE_CPU_AFFINITY = True

# ======================== 棋盘配置 ========================
BOARD_SIZE = 15
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE
WIN_LENGTH = 5
HISTORY_LENGTH = 8
MAX_MOVES = BOARD_SQUARES
MAX_GAME_LENGTH = BOARD_SQUARES

# 输入通道数: 历史当前8步 + 历史对手8步 + 颜色1 + 领域知识2 = 19
INPUT_CHANNELS = HISTORY_LENGTH * 2 + 1 + 2

# ======================== 神经网络配置 ========================
NUM_RES_BLOCKS = 6
NUM_FILTERS = 64
SE_REDUCTION = 16
POLICY_CHANNELS = 2
VALUE_HIDDEN = 64

# 推理优化
USE_TORCHSCRIPT = True          # TorchScript 编译 (仅推理时)
USE_INT8_QUANT = True           # INT8 动态量化 (仅Linear, CPU上quantize_dynamic不支持Conv2d)
USE_BN_FUSE = True              # BN 融合 (推理时)
USE_NHWC = True                 # 通道最后内存格式
USE_ONNX_RUNTIME = False        # ONNX Runtime 推理 (需安装 onnxruntime)
USE_TORCH_COMPILE = True        # torch.compile (PyTorch 2.0+)

# ======================== MCTS 配置 ========================
NUM_SIMULATIONS = 400
C_PUCT = 1.5
C_PUCT_BASE = 19652.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
VIRTUAL_LOSS = 3
TEMPERATURE_THRESHOLD = 30
INITIAL_TEMPERATURE = 1.0

# Undo-based MCTS (替代 Board.copy)
# 注意: 实测 save/restore 快照方式比 Board.copy() 慢
# 因为需要保存/恢复整个棋盘快照 (两次copy)
# Board.copy() 只需一次 copy 且更简洁
USE_UNDO_MCTS = False

# Node Pool 预分配
USE_NODE_POOL = False  # Node Pool有bug, 暂时禁用
NODE_POOL_SIZE = 600000

# 批量推理
MCTS_BATCH_SIZE = 8

# FPU (First Play Urgency)
USE_FPU = True
FPU_VALUE = -0.5

# 模式注入
USE_PATTERN_INJECTION = True
PATTERN_INJECTION_WEIGHT = 0.3

# Gumbel MCTS
USE_GUMBEL_MCTS = False  # Gumbel搜索有bug, 暂时禁用
GUMBEL_TOPK = 16
GUMBEL_SEQUENTIAL_HALVING = True  # 多轮淘汰

# 对称感知 MCTS
USE_SYMMETRY_MCTS = False  # 实现复杂度高，收益有限，默认关闭

# 子树复用
USE_SUBTREE_REUSE = True

# 必走着法检测
USE_MUST_MOVE = True
MUST_MOVE_INCLUDE_OPEN_FOUR = True  # 包含活四/堵活四

# VCT/VCF 搜索
USE_VCT = True
VCT_DEPTH_LIMIT = 12
VCF_DEPTH_LIMIT = 20

# RAVE
USE_RAVE = True
RAVE_EQUIV = 250

# 转置表
USE_TRANSPOSITION = False  # 转置表暂时禁用, 优先修复核心MCTS
TRANSPOSITION_TABLE_SIZE = 1 << 20  # 1M 条目

# 动态模拟次数
USE_DYNAMIC_SIMS = False  # 动态模拟次数有bug(last_entropy初始值导致模拟数爆炸), 暂时禁用
MIN_SIMULATIONS = 50
MAX_SIMULATIONS = 800
SIM_ENTROPY_SCALE = 200.0

# Root Parallelization
USE_ROOT_PARALLEL = False  # Root Parallel有bug(不合并回原始root), 暂时禁用
ROOT_PARALLEL_THREADS = min(4, NUM_ACTORS)

# Progressive Widening
USE_PROGRESSIVE_WIDENING = True
PW_C = 0.5  # pw_k = C * N^alpha, 只考虑 top-k

# Q-value Normalization
USE_Q_NORM = True

# ======================== 自我对弈配置 ========================
NUM_GAMES_PER_ITER = NUM_ACTORS * 10

# 数据质量过滤
MIN_GAME_LENGTH = 8
MAX_GAME_LENGTH_FILTER = 180

# Resign 机制
USE_RESIGN = True
RESIGN_THRESHOLD = -0.95     # 价值持续低于此值则认输
RESIGN_CHECK_STEPS = 5      # 连续N步低于阈值则认输

# ======================== 训练配置 ========================
LEARNING_RATE = 0.01
LR_WARMUP_STEPS = 200
LR_DECAY_STEPS = 100000
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
GRAD_CLIP = 1.0
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 1.5
KL_REG_WEIGHT = 0.01
HUBER_DELTA = 1.0

# SWA (随机权重平均)
USE_SWA = True
SWA_START_STEP = 5000
SWA_UPDATE_FREQ = 100
SWA_LR = 0.0005              # SWA 使用的学习率

# 优化器切换: AdamW → SGD
USE_OPTIMIZER_SWITCH = True
OPTIMIZER_SWITCH_STEP = 20000  # 在此步之后切换到 SGD

# EMA 模型权重
USE_EMA = True
EMA_DECAY = 0.999

# 经验回放
REPLAY_BUFFER_SIZE = 500000
REPLAY_MIN_SIZE = 3000
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 1000000
USE_SUMTREE = True            # SumTree 优先回放

# 数据增广
NUM_SYMMETRIES = 8

# 课程学习 (9x9→15x15)
USE_CURRICULUM = True
CURRICULUM_SMALL_SIZE = 9
CURRICULUM_SMALL_ITERS = 15

# 历史对手池
USE_OPPONENT_POOL = True
OPPONENT_POOL_SIZE = 5        # 保留最近5个历史版本
OPPONENT_POOL_GAME_RATIO = 0.3  # 30% 对局与历史对手下

# 渐进式 MCTS 模拟数
USE_PROGRESSIVE_SIMS = True
PROGRESSIVE_SIMS_SCHEDULE = [  # (iteration, num_sims)
    (0, 50), (20, 100), (50, 200), (100, 300), (150, 400)
]

# ======================== 训练调度 ========================
TOTAL_ITERATIONS = 200
EVAL_INTERVAL = 5
EVAL_GAMES = 20
SAVE_INTERVAL = 5
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "checkpoints/best_model.pt"

# 新旧模型对弈评估
USE_CHAMPION_EVAL = True
CHAMPION_WIN_RATE = 0.55

# ======================== Numba JIT 配置 ========================
NUMBA_CACHE = True
NUMBA_FASTMATH = True
NUMBA_PARALLEL = False         # Numba parallel 在小函数上开销大，默认关闭

# ======================== 邻居表预计算 ========================
NEIGHBOR_RADIUS = 2

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

# ======================== 模式查找表 ========================
# V3 注: _init_pattern_lookup 原为空函数, 已移除
# 棋型检测通过 vct.py 中的 Numba JIT 函数直接计算

# ======================== 中心距离预计算 ========================
_CENTER = BOARD_SIZE // 2
CENTER_DISTANCE = np.zeros(BOARD_SQUARES, dtype=np.int32)
for _i in range(BOARD_SIZE):
    for _j in range(BOARD_SIZE):
        CENTER_DISTANCE[_i * BOARD_SIZE + _j] = abs(_i - _CENTER) + abs(_j - _CENTER)

# 着法紧迫度预排序: 按中心距离升序
MOVE_ORDER_BY_CENTER = np.argsort(CENTER_DISTANCE).astype(np.int32)

# V4 修复: 移除全局 np.random.seed(42), 避免影响其他模块的随机性
# Zobrist 表的确定性由 RandomState(42) 保证, 不需要全局种子
