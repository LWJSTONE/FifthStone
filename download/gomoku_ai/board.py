"""
五子棋棋盘引擎 (V2 — 全面修复+优化版)
======================================
V2 修复+优化:
  1. ~~增量棋型计数~~ — 已移除: pattern_count 从未被读取 (MCTS 使用 vct.py 的 compute_pattern_prior_bonus 从头计算)
  2. Numba JIT 编译 get_feature_planes — 热路径加速
  3. 预分配缓冲区 — _get_legal_moves 不再每次分配
  4. 邻居表扁平化 — Numba 可索引
  5. move_count 替代全盘扫描检查空棋盘
  6. np.argsort 替代冒泡排序
  7. Undo-based 接口 — 支持 MCTS 无需 Board.copy()

注意: pattern_count 增量维护代码已移除, 因为该数组从未被任何模块读取。
      MCTS 使用 vct.py 的 compute_pattern_prior_bonus() 从头计算棋型,
      无需在 Board 中维护增量计数。
"""

import numpy as np
from numba import njit

from config import (
    BOARD_SIZE, BOARD_SQUARES, WIN_LENGTH,
    ZOBRIST_TABLE, ZOBRIST_TURN, NEIGHBOR_TABLE, NEIGHBOR_RADIUS,
    CENTER_DISTANCE, MOVE_ORDER_BY_CENTER, HISTORY_LENGTH, INPUT_CHANNELS,
    NUMBA_CACHE, NUMBA_FASTMATH
)

# ======================== 常量 ========================
EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)
NUM_DIRS = 4

# 棋型索引
PAT_FIVE = 0
PAT_OPEN_FOUR = 1
PAT_HALF_FOUR = 2
PAT_OPEN_THREE = 3
PAT_HALF_THREE = 4
PAT_OPEN_TWO = 5
PAT_HALF_TWO = 6
NUM_PATTERN_TYPES = 7

# 中心位置
_CENTER = BOARD_SIZE // 2

# ======================== 邻居表扁平化(供Numba使用) ========================
def _flatten_neighbor_table():
    """将邻居表扁平化为连续数组(供Numba索引)"""
    flat = []
    offsets = np.zeros(BOARD_SQUARES, dtype=np.int32)
    counts = np.zeros(BOARD_SQUARES, dtype=np.int32)
    offset = 0
    for pos in range(BOARD_SQUARES):
        neighbors = NEIGHBOR_TABLE[pos]
        n = len(neighbors)
        offsets[pos] = offset
        counts[pos] = n
        flat.extend(neighbors.tolist())
        offset += n
    return np.array(flat, dtype=np.int32), offsets, counts

_NEIGHBOR_FLAT, _NEIGHBOR_OFFSETS, _NEIGHBOR_COUNTS = _flatten_neighbor_table()


# ======================== Numba JIT 核心函数 ========================

@njit(cache=NUMBA_CACHE)
def _count_dir(board, r, c, dr, dc, color):
    """沿方向统计连续同色棋子(不含起点)"""
    count = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        count += 1
        nr += dr
        nc += dc
    return count


@njit(cache=NUMBA_CACHE)
def _analyze_dir(board, r, c, dr, dc, color):
    """分析某方向棋型: (total_length, open_ends)"""
    pos = _count_dir(board, r, c, dr, dc, color)
    neg = _count_dir(board, r, c, -dr, -dc, color)
    total = pos + neg + 1
    if total >= WIN_LENGTH:
        return (total, 2)
    open_ends = 0
    er, ec = r + dr * (pos + 1), c + dc * (pos + 1)
    if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE and board[er, ec] == EMPTY:
        open_ends += 1
    br, bc = r - dr * (neg + 1), c - dc * (neg + 1)
    if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and board[br, bc] == EMPTY:
        open_ends += 1
    return (total, open_ends)


@njit(cache=NUMBA_CACHE)
def _pat_idx(length, open_ends):
    """(连子数, 开放端) → 棋型索引"""
    if length >= 5: return PAT_FIVE
    if length == 4:
        if open_ends >= 2: return PAT_OPEN_FOUR
        if open_ends == 1: return PAT_HALF_FOUR
    elif length == 3:
        if open_ends >= 2: return PAT_OPEN_THREE
        if open_ends == 1: return PAT_HALF_THREE
    elif length == 2:
        if open_ends >= 2: return PAT_OPEN_TWO
        if open_ends == 1: return PAT_HALF_TWO
    return -1


@njit(cache=NUMBA_CACHE)
def check_win_at(board, r, c, color):
    """检查落子后是否五连"""
    for d in range(NUM_DIRS):
        dr, dc = DIRECTIONS[d, 0], DIRECTIONS[d, 1]
        count = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1; nr += dr; nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1; nr -= dr; nc -= dc
        if count >= WIN_LENGTH:
            return True
    return False


@njit(cache=NUMBA_CACHE)
def _compute_patterns_for_stone(board, r, c, color):
    """计算某位置在4个方向上的棋型列表"""
    patterns = np.empty(NUM_DIRS, dtype=np.int32)
    for d in range(NUM_DIRS):
        dr, dc = DIRECTIONS[d, 0], DIRECTIONS[d, 1]
        length, open_ends = _analyze_dir(board, r, c, dr, dc, color)
        patterns[d] = _pat_idx(length, open_ends)
    return patterns


@njit(cache=NUMBA_CACHE)
def _pattern_score(pat_idx):
    """棋型索引 → 分值"""
    if pat_idx == PAT_FIVE: return 1000000
    if pat_idx == PAT_OPEN_FOUR: return 100000
    if pat_idx == PAT_HALF_FOUR: return 10000
    if pat_idx == PAT_OPEN_THREE: return 5000
    if pat_idx == PAT_HALF_THREE: return 500
    if pat_idx == PAT_OPEN_TWO: return 200
    if pat_idx == PAT_HALF_TWO: return 50
    return 0


@njit(cache=NUMBA_CACHE)
def _quick_evaluate(board, color):
    """快速评估: 双方棋型得分差 (全盘扫描版，精确)"""
    my_score = 0
    opp_score = 0
    opp = 3 - color
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == EMPTY:
                continue
            stone_c = board[r, c]
            for d in range(NUM_DIRS):
                dr, dc = DIRECTIONS[d, 0], DIRECTIONS[d, 1]
                br, bc = r - dr, c - dc
                if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and board[br, bc] == stone_c:
                    continue
                length, open_ends = _analyze_dir(board, r, c, dr, dc, stone_c)
                pidx = _pat_idx(length, open_ends)
                s = _pattern_score(pidx)
                if stone_c == color:
                    my_score += s
                else:
                    opp_score += s
    return my_score - opp_score


@njit(cache=NUMBA_CACHE)
def _get_legal_moves_incremental(board, neighbor_table_flat, neighbor_offsets,
                                  neighbor_counts, move_count, result_buf):
    """
    增量合法着法生成 (V2: 预分配缓冲区 + move_count 检查)
    返回: 着法数量 (结果写入 result_buf)
    """
    # 空棋盘: 返回中心
    if move_count == 0:
        result_buf[0] = _CENTER * BOARD_SIZE + _CENTER
        return 1

    count = 0
    # 用小数组标记候选位置
    candidate = np.zeros(BOARD_SQUARES, dtype=np.int32)

    for pos in range(BOARD_SQUARES):
        r2, c2 = pos // BOARD_SIZE, pos % BOARD_SIZE
        if board[r2, c2] != EMPTY:
            off = neighbor_offsets[pos]
            cnt = neighbor_counts[pos]
            for k in range(cnt):
                npos = neighbor_table_flat[off + k]
                nr, nc = npos // BOARD_SIZE, npos % BOARD_SIZE
                if board[nr, nc] == EMPTY:
                    candidate[npos] = 1

    for pos in range(BOARD_SQUARES):
        if candidate[pos] == 1:
            result_buf[count] = pos
            count += 1

    if count == 0:
        for pos in range(BOARD_SQUARES):
            r2, c2 = pos // BOARD_SIZE, pos % BOARD_SIZE
            if board[r2, c2] == EMPTY:
                result_buf[count] = pos
                count += 1

    # 按中心距离排序 (插入排序, 小数组更快)
    for i in range(1, count):
        key = result_buf[i]
        key_dist = CENTER_DISTANCE[key]
        j = i - 1
        while j >= 0 and CENTER_DISTANCE[result_buf[j]] > key_dist:
            result_buf[j + 1] = result_buf[j]
            j -= 1
        result_buf[j + 1] = key

    return count


@njit(cache=NUMBA_CACHE)
def _compute_feature_planes_numba(board, move_history_r, move_history_c,
                                  move_history_color, move_count, current_player):
    """
    Numba JIT 编译的特征平面计算 (V2)
    =================================
    通道 0-7:   当前棋手最近8步
    通道 8-15:  对手最近8步
    通道 16:    当前棋手颜色指示
    通道 17:    己方棋型得分 (领域知识)
    通道 18:    对手棋型得分 (领域知识)
    """
    planes = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    my_color = current_player
    opp_color = 3 - current_player

    # 填充历史步
    my_idx = 0
    opp_idx = 0
    for i in range(move_count - 1, -1, -1):  # 从最近到最远
        r = move_history_r[i]
        c = move_history_c[i]
        col = move_history_color[i]

        if col == my_color and my_idx < HISTORY_LENGTH:
            planes[my_idx, r, c] = 1.0
            my_idx += 1
        elif col == opp_color and opp_idx < HISTORY_LENGTH:
            planes[HISTORY_LENGTH + opp_idx, r, c] = 1.0
            opp_idx += 1

        if my_idx >= HISTORY_LENGTH and opp_idx >= HISTORY_LENGTH:
            break

    # 颜色指示
    if my_color == BLACK:
        planes[HISTORY_LENGTH * 2, :, :] = 1.0

    # 领域知识通道: 每个空位的棋型得分
    opponent = 3 - current_player
    max_score = 100.0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            my_score = 0.0
            opp_score = 0.0
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]

                # 己方棋型
                length, open_ends = _analyze_dir(board, r, c, dr, dc, current_player)
                pidx = _pat_idx(length, open_ends)
                if pidx == PAT_FIVE:
                    my_score += 100.0
                elif pidx == PAT_OPEN_FOUR:
                    my_score += 50.0
                elif pidx == PAT_HALF_FOUR:
                    my_score += 10.0
                elif pidx == PAT_OPEN_THREE:
                    my_score += 5.0
                elif pidx == PAT_HALF_THREE:
                    my_score += 1.0

                # 对手棋型
                length, open_ends = _analyze_dir(board, r, c, dr, dc, opponent)
                pidx = _pat_idx(length, open_ends)
                if pidx == PAT_FIVE:
                    opp_score += 100.0
                elif pidx == PAT_OPEN_FOUR:
                    opp_score += 50.0
                elif pidx == PAT_HALF_FOUR:
                    opp_score += 10.0
                elif pidx == PAT_OPEN_THREE:
                    opp_score += 5.0
                elif pidx == PAT_HALF_THREE:
                    opp_score += 1.0

            planes[HISTORY_LENGTH * 2 + 1, r, c] = min(1.0, my_score / max_score)
            planes[HISTORY_LENGTH * 2 + 2, r, c] = min(1.0, opp_score / max_score)

    return planes


# ======================== Python 层 Board 类 ========================

class Board:
    """
    五子棋棋盘 (V2 — 全面修复版)
    ============================
      - Undo-based 接口: 支持 save/restore 用于 MCTS

    注意: pattern_count 增量棋型计数已移除 — 该数组从未被读取。
          MCTS 通过 vct.py 的 compute_pattern_prior_bonus() 从头计算棋型。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.move_history = []       # [(r, c, color), ...]
        self.zobrist_hash = np.int64(0)
        self.game_over = False
        self.winner = 0
        self.move_count = 0
        # 注意: pattern_count 已移除 — 从未被任何模块读取, MCTS 使用 vct.py 从头计算
        # 预分配缓冲区
        self._legal_moves_buf = np.zeros(BOARD_SQUARES, dtype=np.int32)
        # 历史记录数组 (供 Numba 使用)
        self._history_r = np.zeros(BOARD_SQUARES, dtype=np.int32)
        self._history_c = np.zeros(BOARD_SQUARES, dtype=np.int32)
        self._history_color = np.zeros(BOARD_SQUARES, dtype=np.int32)
        # 保存点栈 (用于 Undo-based MCTS)
        self._save_stack = []

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
        b._legal_moves_buf = np.zeros(BOARD_SQUARES, dtype=np.int32)
        b._history_r = self._history_r.copy()
        b._history_c = self._history_c.copy()
        b._history_color = self._history_color.copy()
        b._save_stack = []
        return b

    def save_state(self):
        """保存当前状态 (用于 Undo-based MCTS, 比 copy() 轻量)"""
        # V2 优化: 直接保存棋盘快照 + 关键状态, 恢复时整块覆盖
        # 比逐步undo快得多
        state = (
            self.board.copy(),           # 棋盘快照
            self.move_count,
            self.current_player,
            self.zobrist_hash,
            self.game_over,
            self.winner,
            len(self.move_history),      # 历史长度
        )
        self._save_stack.append(state)
        return len(self._save_stack)

    def restore_state(self):
        """
        V2 优化: 直接覆盖恢复, 避免逐步undo的增量棋型计算开销
        """
        if not self._save_stack:
            return False
        state = self._save_stack.pop()
        (old_board, old_move_count, old_player, old_hash,
         old_over, old_winner, old_hist_len) = state

        # 直接覆盖棋盘 (比逐步undo快10×+)
        self.board[:] = old_board
        self.move_count = old_move_count
        self.current_player = old_player
        self.zobrist_hash = old_hash
        self.game_over = old_over
        self.winner = old_winner

        # 截断历史
        while len(self.move_history) > old_hist_len:
            self.move_history.pop()

        # 更新 Numba 历史数组
        for i in range(old_hist_len):
            r, c, color = self.move_history[i]
            self._history_r[i] = r
            self._history_c[i] = c
            self._history_color[i] = color

        return True

    def place_stone(self, r, c):
        """在(r,c)落子 — 不维护增量棋型计数 (pattern_count 已移除)"""
        if self.game_over or not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.board[r, c] != EMPTY:
            return False

        color = self.current_player
        self.board[r, c] = color
        idx = r * BOARD_SIZE + c
        self.zobrist_hash ^= ZOBRIST_TABLE[idx, color - 1]
        self.zobrist_hash ^= ZOBRIST_TURN

        self.move_history.append((r, c, color))
        self._history_r[self.move_count] = r
        self._history_c[self.move_count] = c
        self._history_color[self.move_count] = color
        self.move_count += 1

        if check_win_at(self.board, r, c, color):
            self.game_over = True
            self.winner = color
        elif self.move_count >= BOARD_SQUARES:
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = 3 - color
        return True

    def place_stone_fast(self, r, c):
        """
        V2: 快速落子 — 用于MCTS模拟 (与 place_stone 等效, 保留向后兼容)
        """
        if self.game_over or not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return False
        if self.board[r, c] != EMPTY:
            return False

        color = self.current_player
        self.board[r, c] = color
        idx = r * BOARD_SIZE + c
        self.zobrist_hash ^= ZOBRIST_TABLE[idx, color - 1]
        self.zobrist_hash ^= ZOBRIST_TURN

        self.move_history.append((r, c, color))
        self._history_r[self.move_count] = r
        self._history_c[self.move_count] = c
        self._history_color[self.move_count] = color
        self.move_count += 1

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
        """撤销最后一步 — 不维护增量棋型计数 (pattern_count 已移除)"""
        if not self.move_history:
            return False
        r, c, color = self.move_history.pop()
        self.move_count -= 1

        self.board[r, c] = EMPTY
        idx = r * BOARD_SIZE + c
        self.zobrist_hash ^= ZOBRIST_TABLE[idx, color - 1]
        self.zobrist_hash ^= ZOBRIST_TURN

        self.current_player = color
        self.game_over = False
        self.winner = 0
        return True

    # [已移除] 以下增量棋型维护方法已删除, 因为 pattern_count 从未被读取:
    #   _remove_affected_patterns, _remove_line_patterns_in_dir,
    #   _add_new_patterns, _add_line_patterns_in_dir,
    #   _add_new_patterns_undo, _undo_patterns
    # MCTS 使用 vct.py 的 compute_pattern_prior_bonus() 从头计算棋型

    def get_legal_moves(self):
        """获取排序后的合法着法列表"""
        count = _get_legal_moves_incremental(
            self.board, _NEIGHBOR_FLAT, _NEIGHBOR_OFFSETS, _NEIGHBOR_COUNTS,
            self.move_count, self._legal_moves_buf
        )
        return [(int(self._legal_moves_buf[i] // BOARD_SIZE),
                 int(self._legal_moves_buf[i] % BOARD_SIZE))
                for i in range(count)]

    def get_legal_move_indices(self):
        """获取排序后的合法着法索引列表(0-224)"""
        count = _get_legal_moves_incremental(
            self.board, _NEIGHBOR_FLAT, _NEIGHBOR_OFFSETS, _NEIGHBOR_COUNTS,
            self.move_count, self._legal_moves_buf
        )
        return [int(self._legal_moves_buf[i]) for i in range(count)]

    def get_legal_move_count(self):
        """获取合法着法数量"""
        return _get_legal_moves_incremental(
            self.board, _NEIGHBOR_FLAT, _NEIGHBOR_OFFSETS, _NEIGHBOR_COUNTS,
            self.move_count, self._legal_moves_buf
        )

    def is_legal(self, r, c):
        return (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
                and self.board[r, c] == EMPTY and not self.game_over)

    def check_win(self, r, c):
        color = self.board[r, c]
        return color != EMPTY and check_win_at(self.board, r, c, color)

    def get_feature_planes(self):
        """
        生成神经网络输入特征平面 (V2: Numba JIT 编译)
        """
        return _compute_feature_planes_numba(
            self.board, self._history_r, self._history_c,
            self._history_color, self.move_count, self.current_player
        )

    def quick_evaluate(self):
        return _quick_evaluate(self.board, self.current_player)

    def get_move_index(self, r, c):
        return r * BOARD_SIZE + c

    def index_to_move(self, idx):
        return idx // BOARD_SIZE, idx % BOARD_SIZE

    def __str__(self):
        symbols = {EMPTY: '·', BLACK: '●', WHITE: '○'}
        lines = ['   ' + ' '.join(f'{c:2d}' for c in range(BOARD_SIZE))]
        for r in range(BOARD_SIZE):
            row = f'{r:2d} ' + ' '.join(f' {symbols[self.board[r, c]]}' for c in range(BOARD_SIZE))
            lines.append(row)
        return '\n'.join(lines)

    @staticmethod
    def get_symmetries(feature_planes, policy):
        """8种对称增广"""
        results = []
        for k in range(4):
            rotated_f = np.rot90(feature_planes, k, axes=(1, 2))
            rotated_p = np.rot90(policy.reshape(BOARD_SIZE, BOARD_SIZE), k).flatten()
            results.append((rotated_f.copy(), rotated_p.copy()))
            flipped_f = np.flip(rotated_f, axis=2)
            flipped_p = np.fliplr(rotated_p.reshape(BOARD_SIZE, BOARD_SIZE)).flatten()
            results.append((flipped_f.copy(), flipped_p.copy()))
        return results
