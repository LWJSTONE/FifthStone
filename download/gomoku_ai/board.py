"""
五子棋棋盘引擎 — Numba JIT + 位运算 + 增量评估
================================================
核心优化:
  1. 平面数组表示 + 4方向增量模式识别
  2. Numba @njit 编译热路径为原生代码
  3. Zobrist 哈希支持转置表
  4. 快速合法着法生成 + 中心距离排序
  5. 增量评估: 每步只更新受影响方向的棋型计数
"""

import numpy as np
from numba import njit

from config import (
    BOARD_SIZE, BOARD_SQUARES, WIN_LENGTH, PATTERN_SCORES,
    ZOBRIST_TABLE, ZOBRIST_TURN
)

# ======================== 常量 ========================
EMPTY = 0
BLACK = 1
WHITE = 2

# 4个方向: 水平、垂直、对角线、反对角线
DIRECTIONS = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)
NUM_DIRS = 4

# 棋型索引
PATTERN_FIVE = 0
PATTERN_OPEN_FOUR = 1
PATTERN_HALF_FOUR = 2
PATTERN_OPEN_THREE = 3
PATTERN_HALF_THREE = 4
PATTERN_OPEN_TWO = 5
PATTERN_HALF_TWO = 6
NUM_PATTERNS = 7


# ======================== Numba JIT 核心函数 ========================
# 使用惰性编译(njit不带签名)，让Numba自动推断类型，更健壮

@njit(cache=True)
def _count_direction(board, r, c, dr, dc, color):
    """从(r,c)沿(dr,dc)方向统计连续同色棋子数"""
    count = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        count += 1
        nr += dr
        nc += dc
    return count


@njit(cache=True)
def _analyze_direction(board, r, c, dr, dc, color):
    """分析某方向上的棋型: 返回 (连子数, 开放端数)"""
    # 正方向连续
    pos_count = _count_direction(board, r, c, dr, dc, color)
    # 反方向连续
    neg_count = _count_direction(board, r, c, -dr, -dc, color)
    total = pos_count + neg_count + 1  # +1 for the stone at (r,c)

    if total >= WIN_LENGTH:
        return (total, 2)

    # 计算开放端
    open_ends = 0
    # 正方向端点
    er, ec = r + dr * (pos_count + 1), c + dc * (pos_count + 1)
    if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE and board[er, ec] == EMPTY:
        open_ends += 1
    # 反方向端点
    br, bc = r - dr * (neg_count + 1), c - dc * (neg_count + 1)
    if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and board[br, bc] == EMPTY:
        open_ends += 1

    return (total, open_ends)


@njit(cache=True)
def _pattern_index(length, open_ends):
    """将(连子数, 开放端数)映射到棋型索引"""
    if length >= 5:
        return PATTERN_FIVE
    if length == 4:
        if open_ends == 2:
            return PATTERN_OPEN_FOUR
        elif open_ends == 1:
            return PATTERN_HALF_FOUR
        return -1
    if length == 3:
        if open_ends == 2:
            return PATTERN_OPEN_THREE
        elif open_ends == 1:
            return PATTERN_HALF_THREE
        return -1
    if length == 2:
        if open_ends == 2:
            return PATTERN_OPEN_TWO
        elif open_ends == 1:
            return PATTERN_HALF_TWO
        return -1
    return -1


@njit(cache=True)
def compute_zobrist_hash(board, current_player):
    """计算当前棋盘的Zobrist哈希值"""
    h = np.int64(0)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                idx = r * BOARD_SIZE + c
                h ^= ZOBRIST_TABLE[idx, board[r, c] - 1]
    if current_player == WHITE:
        h ^= ZOBRIST_TURN
    return h


@njit(cache=True)
def check_win_at(board, r, c, color):
    """检查在(r,c)落子后是否五连(仅检查含该子的方向)"""
    for d in range(NUM_DIRS):
        dr = DIRECTIONS[d, 0]
        dc = DIRECTIONS[d, 1]
        count = 1
        # 正方向
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1
            nr += dr
            nc += dc
        # 反方向
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1
            nr -= dr
            nc -= dc
        if count >= WIN_LENGTH:
            return True
    return False


@njit(cache=True)
def _get_legal_moves_sorted(board):
    """获取合法着法并按中心距离排序(近→远)，仅返回已有棋子周围的空位"""
    center = BOARD_SIZE // 2
    result = np.empty((BOARD_SQUARES, 2), dtype=np.int32)
    count = 0

    # 如果棋盘为空，返回中心点
    board_empty = True
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                board_empty = False
                break
        if not board_empty:
            break

    if board_empty:
        result[0, 0] = center
        result[0, 1] = center
        return result[:1]

    # 标记已有棋子周围的空位(2格范围)
    has_neighbor = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == EMPTY:
                            has_neighbor[nr, nc] = 1

    # 收集合法着法
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == EMPTY and has_neighbor[r, c] == 1:
                result[count, 0] = r
                result[count, 1] = c
                count += 1

    if count == 0:
        # 没有邻近空位，取所有空位
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] == EMPTY:
                    result[count, 0] = r
                    result[count, 1] = c
                    count += 1

    # 按曼哈顿距离排序(冒泡，棋盘小)
    for i in range(count):
        for j in range(i + 1, count):
            di = abs(result[i, 0] - center) + abs(result[i, 1] - center)
            dj = abs(result[j, 0] - center) + abs(result[j, 1] - center)
            if dj < di:
                tmp0 = result[i, 0]
                tmp1 = result[i, 1]
                result[i, 0] = result[j, 0]
                result[i, 1] = result[j, 1]
                result[j, 0] = tmp0
                result[j, 1] = tmp1

    return result[:count]


@njit(cache=True)
def _quick_evaluate(board, color):
    """快速评估函数: 统计棋型得分差(用于MCTS回滚)"""
    my_score = 0
    opp_score = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                stone_color = board[r, c]
                for d in range(NUM_DIRS):
                    dr = DIRECTIONS[d, 0]
                    dc = DIRECTIONS[d, 1]
                    # 只向正方向统计，避免重复
                    br, bc = r - dr, c - dc
                    if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and board[br, bc] == stone_color:
                        continue

                    length, open_ends = _analyze_direction(board, r, c, dr, dc, stone_color)
                    pidx = _pattern_index(length, open_ends)
                    score = 0
                    if pidx == PATTERN_FIVE:
                        score = 1000000
                    elif pidx == PATTERN_OPEN_FOUR:
                        score = 100000
                    elif pidx == PATTERN_HALF_FOUR:
                        score = 10000
                    elif pidx == PATTERN_OPEN_THREE:
                        score = 5000
                    elif pidx == PATTERN_HALF_THREE:
                        score = 500
                    elif pidx == PATTERN_OPEN_TWO:
                        score = 200
                    elif pidx == PATTERN_HALF_TWO:
                        score = 50

                    if stone_color == color:
                        my_score += score
                    else:
                        opp_score += score

    return my_score - opp_score


# ======================== Python 层 Board 类 ========================

class Board:
    """
    五子棋棋盘对象
    =============
    维护:
      - 15x15 棋盘数组 (int8)
      - 当前玩家
      - 历史记录(支持撤销)
      - Zobrist哈希(支持转置表)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.move_history = []
        self.zobrist_hash = np.int64(0)
        self.game_over = False
        self.winner = 0
        self.move_count = 0

    def copy(self):
        """深拷贝棋盘"""
        b = Board.__new__(Board)
        b.board = self.board.copy()
        b.current_player = self.current_player
        b.move_history = list(self.move_history)
        b.zobrist_hash = self.zobrist_hash
        b.game_over = self.game_over
        b.winner = self.winner
        b.move_count = self.move_count
        return b

    def place_stone(self, r, c):
        """在(r,c)落子，返回是否成功"""
        if self.game_over:
            return False
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.board[r, c] != EMPTY:
            return False

        color = self.current_player
        self.board[r, c] = color

        # 更新Zobrist哈希
        idx = r * BOARD_SIZE + c
        self.zobrist_hash ^= ZOBRIST_TABLE[idx, color - 1]
        self.zobrist_hash ^= ZOBRIST_TURN

        # 记录历史
        self.move_history.append((r, c, color))
        self.move_count += 1

        # 检查胜负
        if check_win_at(self.board, r, c, color):
            self.game_over = True
            self.winner = color
        elif self.move_count >= BOARD_SQUARES:
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = 3 - color

        return True

    def undo_stone(self):
        """撤销最后一步"""
        if not self.move_history:
            return False
        r, c, color = self.move_history.pop()
        self.move_count -= 1

        # 更新Zobrist哈希
        idx = r * BOARD_SIZE + c
        self.zobrist_hash ^= ZOBRIST_TABLE[idx, color - 1]
        self.zobrist_hash ^= ZOBRIST_TURN

        self.board[r, c] = EMPTY
        self.current_player = color
        self.game_over = False
        self.winner = 0
        return True

    def get_legal_moves(self):
        """获取排序后的合法着法列表(距中心近→远)"""
        moves_arr = _get_legal_moves_sorted(self.board)
        return [(int(moves_arr[i, 0]), int(moves_arr[i, 1])) for i in range(len(moves_arr))]

    def is_legal(self, r, c):
        """检查着法是否合法"""
        return (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                and self.board[r, c] == EMPTY and not self.game_over)

    def check_win(self, r, c):
        """检查在(r,c)落子后是否获胜"""
        color = self.board[r, c]
        if color == EMPTY:
            return False
        return check_win_at(self.board, r, c, color)

    def get_feature_planes(self):
        """
        生成神经网络输入特征平面
        =======================
        通道布局 (共17通道):
          0-7:   当前棋手最近8步
          8-15:  对手最近8步
          16:    当前棋手颜色指示(全1=黑, 全0=白)
        """
        from config import INPUT_CHANNELS, HISTORY_LENGTH
        planes = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        history = self.move_history
        my_color = self.current_player

        # 当前棋手的最近8步
        my_moves = [(r, c) for r, c, col in history if col == my_color][-HISTORY_LENGTH:]
        for i, (r, c) in enumerate(reversed(my_moves)):
            if i < HISTORY_LENGTH:
                planes[i, r, c] = 1.0

        # 对手的最近8步
        opp_moves = [(r, c) for r, c, col in history if col != my_color][-HISTORY_LENGTH:]
        for i, (r, c) in enumerate(reversed(opp_moves)):
            if i < HISTORY_LENGTH:
                planes[HISTORY_LENGTH + i, r, c] = 1.0

        # 当前棋手颜色指示
        if my_color == BLACK:
            planes[INPUT_CHANNELS - 1, :, :] = 1.0

        return planes

    def quick_evaluate(self):
        """快速评估(用于MCTS回滚)"""
        return _quick_evaluate(self.board, self.current_player)

    def get_move_index(self, r, c):
        """将(row,col)转换为动作索引"""
        return r * BOARD_SIZE + c

    def index_to_move(self, idx):
        """将动作索引转换为(row,col)"""
        return idx // BOARD_SIZE, idx % BOARD_SIZE

    def __str__(self):
        """可视化棋盘"""
        symbols = {EMPTY: '·', BLACK: '●', WHITE: '○'}
        lines = []
        header = '   ' + ' '.join(f'{c:2d}' for c in range(BOARD_SIZE))
        lines.append(header)
        for r in range(BOARD_SIZE):
            row = f'{r:2d} ' + ' '.join(f' {symbols[self.board[r, c]]}' for c in range(BOARD_SIZE))
            lines.append(row)
        return '\n'.join(lines)

    @staticmethod
    def get_symmetries(feature_planes, policy):
        """
        8种对称增广(4旋转 × 2翻转)
        =============================
        返回: [(feature, policy), ...] 共8组
        """
        results = []
        for k in range(4):
            # 旋转k×90°
            rotated_f = np.rot90(feature_planes, k, axes=(1, 2))
            rotated_p = np.rot90(policy.reshape(BOARD_SIZE, BOARD_SIZE), k).flatten()
            results.append((rotated_f.copy(), rotated_p.copy()))

            # 翻转
            flipped_f = np.flip(rotated_f, axis=2)
            flipped_p = np.fliplr(rotated_p.reshape(BOARD_SIZE, BOARD_SIZE)).flatten()
            results.append((flipped_f.copy(), flipped_p.copy()))

        return results
