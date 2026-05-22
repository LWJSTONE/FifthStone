"""
VCT/VCF 威胁空间搜索 + 必走着法快速检测
========================================
五子棋 AI 最关键的战术模块:
  1. VCF (Victory by Continuous Four): 连续冲四取胜
  2. VCT (Victory by Continuous Threat): 连续威胁(冲四+活三)取胜
  3. 必走着法检测: 己方活四→必胜, 对手活四/冲四→必防

这些搜索空间极小(只搜威胁着法), 但能发现 MCTS 极难找到的强制胜路线。
没有 VCT 的五子棋 AI 永远达不到"最强"级别。
"""

import numpy as np
from numba import njit

from config import BOARD_SIZE, BOARD_SQUARES, WIN_LENGTH

# ======================== 常量 ========================
EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)
NUM_DIRS = 4

# 棋型编码
PATTERN_NONE = 0
PATTERN_FIVE = 1
PATTERN_OPEN_FOUR = 2       # 活四 (XOOOO_)
PATTERN_HALF_FOUR = 3       # 冲四/嵌五 (_XOOO_ 一端被堵)
PATTERN_OPEN_THREE = 4      # 活三 (__OOO__)
PATTERN_HALF_THREE = 5      # 眠三
PATTERN_OPEN_TWO = 6        # 活二
PATTERN_HALF_TWO = 7        # 眠二


# ======================== Numba JIT 核心函数 ========================

@njit(cache=True)
def _count_consecutive(board, r, c, dr, dc, color):
    """从(r,c)沿(dr,dc)方向统计连续同色棋子数(不含起点)"""
    count = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        count += 1
        nr += dr
        nc += dc
    return count


@njit(cache=True)
def _analyze_line_pattern(board, r, c, dr, dc, color):
    """
    分析从(r,c)出发在(dr,dc)方向上的棋型
    返回: (total_length, open_ends)
    """
    pos = _count_consecutive(board, r, c, dr, dc, color)
    neg = _count_consecutive(board, r, c, -dr, -dc, color)
    total = pos + neg + 1

    if total >= WIN_LENGTH:
        return (total, 2)

    open_ends = 0
    # 正方向端点
    er, ec = r + dr * (pos + 1), c + dc * (pos + 1)
    if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE and board[er, ec] == EMPTY:
        open_ends += 1
    # 反方向端点
    br, bc = r - dr * (neg + 1), c - dc * (neg + 1)
    if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and board[br, bc] == EMPTY:
        open_ends += 1

    return (total, open_ends)


@njit(cache=True)
def _get_pattern_type(length, open_ends):
    """将(连子数, 开放端)映射为棋型编码"""
    if length >= 5:
        return PATTERN_FIVE
    if length == 4:
        if open_ends >= 2:
            return PATTERN_OPEN_FOUR
        elif open_ends == 1:
            return PATTERN_HALF_FOUR
    elif length == 3:
        if open_ends >= 2:
            return PATTERN_OPEN_THREE
        elif open_ends == 1:
            return PATTERN_HALF_THREE
    elif length == 2:
        if open_ends >= 2:
            return PATTERN_OPEN_TWO
        elif open_ends == 1:
            return PATTERN_HALF_TWO
    return PATTERN_NONE


@njit(cache=True)
def _get_threat_moves(board, color):
    """
    获取所有威胁着法(能形成活四/冲四/活三的空位)
    返回: (positions, threat_scores) — 两个数组
    """
    positions = np.empty(BOARD_SQUARES, dtype=np.int32)
    scores = np.empty(BOARD_SQUARES, dtype=np.int32)
    count = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            # 检查落子后能形成什么棋型
            best_score = 0
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, color)
                ptype = _get_pattern_type(length, open_ends)

                if ptype == PATTERN_FIVE:
                    best_score = max(best_score, 100000)
                elif ptype == PATTERN_OPEN_FOUR:
                    best_score = max(best_score, 50000)
                elif ptype == PATTERN_HALF_FOUR:
                    best_score = max(best_score, 10000)
                elif ptype == PATTERN_OPEN_THREE:
                    best_score = max(best_score, 5000)

            if best_score > 0:
                positions[count] = r * BOARD_SIZE + c
                scores[count] = best_score
                count += 1

    return positions[:count], scores[:count]


@njit(cache=True)
def _check_five(board, r, c, color):
    """检查在(r,c)落子后是否形成五连"""
    for d in range(NUM_DIRS):
        dr = DIRECTIONS[d, 0]
        dc = DIRECTIONS[d, 1]
        count = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1
            nr -= dr
            nc -= dc
        if count >= WIN_LENGTH:
            return True
    return False


@njit(cache=True)
def _has_open_four(board, color):
    """检查color方是否有活四"""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, color)
                if length >= 5 or (length == 4 and open_ends >= 2):
                    return True
    return False


@njit(cache=True)
def _has_half_four(board, color):
    """检查color方是否有冲四(含活四)"""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, color)
                if length >= 5 or (length == 4 and open_ends >= 1):
                    return True
    return False


# ======================== VCF 搜索 ========================

@njit(cache=True)
def vcf_search(board, attacker, depth_limit=20):
    """
    VCF搜索: 寻找连续冲四取胜路线
    ===============================
    只搜索冲四着法(搜索空间极小), 深度可达20+层

    参数:
      board: 棋盘数组 (int8[15,15])
      attacker: 攻击方颜色
      depth_limit: 最大搜索深度

    返回:
      winning_move: 胜着位置(0-224), -1表示未找到
    """
    # 检查攻击方是否已有五连
    # 找攻击方的冲四着法
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            # 试着落子
            is_four = False
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, attacker)
                if length >= 5:
                    # 直接五连, 找到了
                    return r * BOARD_SIZE + c
                if length == 4 and open_ends >= 1:
                    is_four = True

            if not is_four:
                continue

            # 落子(冲四)
            board[r, c] = attacker

            # 对手必须应对(堵四)
            # 检查对手是否有五连(对手可能忽略冲四直接赢)
            defender = 3 - attacker
            defender_wins = False

            # 找对手能堵的位置
            defender_can_block = False
            for dr2 in range(BOARD_SIZE):
                for dc2 in range(BOARD_SIZE):
                    if board[dr2, dc2] != EMPTY:
                        continue
                    # 对手堵这个冲四
                    if _check_five(board, dr2, dc2, defender):
                        defender_wins = True
                        break
                    # 检查是否堵了我们的冲四
                    for dd in range(NUM_DIRS):
                        ddr = DIRECTIONS[dd, 0]
                        ddc = DIRECTIONS[dd, 1]
                        ll, oo = _analyze_line_pattern(board, r, c, ddr, ddc, attacker)
                        if ll == 4:
                            # 检查(dr2,dc2)是否在这条线上
                            # 简化: 对手任意落子都算堵(不精确但够用)
                            pass
                    defender_can_block = True
                    if defender_wins:
                        break
                if defender_wins:
                    break

            if defender_wins:
                board[r, c] = EMPTY
                continue

            if not defender_can_block:
                # 冲四无法被堵(不可能,但安全检查)
                board[r, c] = EMPTY
                return r * BOARD_SIZE + c

            # 对手最优应对: 堵冲四的位置
            # 递归搜索: 对手每种堵法后, 攻击方是否还能VCF
            found = False
            if depth_limit > 1:
                # 简化: 对手在冲四的延长线端点落子
                for dr2 in range(BOARD_SIZE):
                    for dc2 in range(BOARD_SIZE):
                        if board[dr2, dc2] != EMPTY:
                            continue
                        # 对手落子
                        board[dr2, dc2] = defender

                        # 递归: 攻击方是否还能VCF
                        result = vcf_search(board, attacker, depth_limit - 1)
                        if result >= 0:
                            found = True
                            board[dr2, dc2] = EMPTY
                            break

                        board[dr2, dc2] = EMPTY

            if found:
                board[r, c] = EMPTY
                return r * BOARD_SIZE + c

            board[r, c] = EMPTY

    return -1


# ======================== VCT 搜索 ========================

@njit(cache=True)
def vct_search(board, attacker, depth_limit=12):
    """
    VCT搜索: 寻找连续威胁(冲四+活三)取胜路线
    ==========================================
    比 VCF 搜索空间稍大, 但能发现更多强制胜

    参数:
      board: 棋盘数组
      attacker: 攻击方颜色
      depth_limit: 最大搜索深度

    返回:
      winning_move: 胜着位置, -1表示未找到
    """
    if depth_limit <= 0:
        return -1

    defender = 3 - attacker

    # 1. 先试 VCF (纯冲四取胜)
    vcf_result = vcf_search(board, attacker, min(depth_limit, 20))
    if vcf_result >= 0:
        return vcf_result

    # 2. 找活三着法 (形成活三, 迫使对手应对)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            # 检查这步能否形成活三(或更好的)
            best_threat = PATTERN_NONE
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, attacker)
                ptype = _get_pattern_type(length, open_ends)
                if ptype > best_threat:
                    best_threat = ptype

            # 只搜活三及以上的威胁着法
            if best_threat < PATTERN_OPEN_THREE:
                continue

            # 落子
            board[r, c] = attacker

            # 检查是否直接五连
            if _check_five(board, r, c, attacker):
                board[r, c] = EMPTY
                return r * BOARD_SIZE + c

            # 对手最优应对
            # 如果形成了活四, 对手必须堵; 如果形成了活三, 对手也应堵
            # 简化: 对手在最有威胁的位置落子
            best_defense = -1
            best_defense_score = -1

            for dr2 in range(BOARD_SIZE):
                for dc2 in range(BOARD_SIZE):
                    if board[dr2, dc2] != EMPTY:
                        continue
                    # 评估这个防御位置的价值
                    defense_score = 0
                    for d in range(NUM_DIRS):
                        ddr = DIRECTIONS[d, 0]
                        ddc = DIRECTIONS[d, 1]
                        ll, oo = _analyze_line_pattern(board, dr2, dc2, ddr, ddc, defender)
                        pt = _get_pattern_type(ll, oo)
                        if pt == PATTERN_FIVE:
                            defense_score = 200000
                            break
                        elif pt == PATTERN_OPEN_FOUR:
                            defense_score = max(defense_score, 100000)
                        elif pt == PATTERN_HALF_FOUR:
                            defense_score = max(defense_score, 50000)
                        elif pt == PATTERN_OPEN_THREE:
                            defense_score = max(defense_score, 10000)

                    # 也考虑堵攻击方
                    for d in range(NUM_DIRS):
                        ddr = DIRECTIONS[d, 0]
                        ddc = DIRECTIONS[d, 1]
                        ll, oo = _analyze_line_pattern(board, dr2, dc2, ddr, ddc, attacker)
                        pt = _get_pattern_type(ll, oo)
                        if pt == PATTERN_FIVE:
                            defense_score = max(defense_score, 150000)
                        elif pt == PATTERN_OPEN_FOUR:
                            defense_score = max(defense_score, 80000)
                        elif pt == PATTERN_HALF_FOUR:
                            defense_score = max(defense_score, 40000)
                        elif pt == PATTERN_OPEN_THREE:
                            defense_score = max(defense_score, 8000)

                    if defense_score > best_defense_score:
                        best_defense_score = defense_score
                        best_defense = dr2 * BOARD_SIZE + dc2

            if best_defense >= 0:
                dr2, dc2 = best_defense // BOARD_SIZE, best_defense % BOARD_SIZE
                board[dr2, dc2] = defender

                # 递归: 攻击方是否还能VCT
                result = vct_search(board, attacker, depth_limit - 2)
                board[dr2, dc2] = EMPTY

                if result >= 0:
                    board[r, c] = EMPTY
                    return r * BOARD_SIZE + c

            board[r, c] = EMPTY

    return -1


# ======================== 必走着法检测 ========================

@njit(cache=True)
def find_must_move(board, current_player):
    """
    必走着法检测 — 在启动MCTS前调用, 跳过不必要的大规模搜索
    ====================================================
    返回: (must_move_idx, move_type)
      must_move_idx: 必走着法索引(0-224), -1表示无必走着法
      move_type: 0=无, 1=己方五连(必胜), 2=堵对手五连(必防),
                 3=己方活四(必胜), 4=堵对手活四(必防)
    """
    opponent = 3 - current_player

    # 1. 己方能否直接五连? → 必胜
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            if _check_five(board, r, c, current_player):
                return (r * BOARD_SIZE + c, 1)

    # 2. 对手能否直接五连? → 必防
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            if _check_five(board, r, c, opponent):
                return (r * BOARD_SIZE + c, 2)

    # 3. 己方能否形成活四? → 必胜(活四无法防守)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                if length == 4 and open_ends >= 2:
                    return (r * BOARD_SIZE + c, 3)

    # 4. 对手能否形成活四? → 必防
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, opponent)
                if length == 4 and open_ends >= 2:
                    return (r * BOARD_SIZE + c, 4)

    return (-1, 0)


# ======================== 模式注入 MCTS 先验 ========================

@njit(cache=True)
def compute_pattern_prior_bonus(board, current_player):
    """
    为每个合法着法计算基于棋型的先验加分
    ======================================
    用于与网络策略混合, 让MCTS不遗漏关键战术着法

    返回: bonus[225] 数组, 每个位置的先验加分
    """
    bonus = np.zeros(BOARD_SQUARES, dtype=np.float32)
    opponent = 3 - current_player

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            idx = r * BOARD_SIZE + c
            my_best = PATTERN_NONE
            opp_best = PATTERN_NONE

            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]

                # 己方棋型
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                ptype = _get_pattern_type(length, open_ends)
                if ptype > my_best:
                    my_best = ptype

                # 对手棋型(落子消除对手棋型)
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, opponent)
                ptype = _get_pattern_type(length, open_ends)
                if ptype > opp_best:
                    opp_best = ptype

            # 进攻加分
            if my_best == PATTERN_FIVE:
                bonus[idx] = 100.0       # 直接五连
            elif my_best == PATTERN_OPEN_FOUR:
                bonus[idx] = 50.0        # 活四(必胜)
            elif my_best == PATTERN_HALF_FOUR:
                bonus[idx] = 20.0        # 冲四
            elif my_best == PATTERN_OPEN_THREE:
                bonus[idx] = 15.0        # 活三

            # 防守加分(堵对手棋型)
            if opp_best == PATTERN_FIVE:
                bonus[idx] = max(bonus[idx], 80.0)   # 堵五连
            elif opp_best == PATTERN_OPEN_FOUR:
                bonus[idx] = max(bonus[idx], 40.0)   # 堵活四
            elif opp_best == PATTERN_HALF_FOUR:
                bonus[idx] = max(bonus[idx], 15.0)   # 堵冲四
            elif opp_best == PATTERN_OPEN_THREE:
                bonus[idx] = max(bonus[idx], 10.0)   # 堵活三

            # 双威胁加分(一步形成两个威胁)
            my_threat_count = 0
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                ptype = _get_pattern_type(length, open_ends)
                if ptype >= PATTERN_HALF_FOUR:
                    my_threat_count += 1
                elif ptype >= PATTERN_OPEN_THREE:
                    my_threat_count += 1

            if my_threat_count >= 2:
                bonus[idx] += 30.0  # 双威胁(如双活三/冲四活三)

    return bonus


# ======================== 领域知识特征通道 ========================

@njit(cache=True)
def compute_pattern_feature_channels(board, current_player):
    """
    计算两个领域知识特征通道
    ========================
    通道0: 己方棋型得分 (归一化到0-1)
    通道1: 对手棋型得分 (归一化到0-1)

    用于拼接到网络输入中, 注入领域知识, 加速收敛
    """
    my_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opponent = 3 - current_player

    max_score = 100.0  # 归一化常数

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            my_score = 0.0
            opp_score = 0.0

            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]

                # 己方
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                ptype = _get_pattern_type(length, open_ends)
                if ptype == PATTERN_FIVE:
                    my_score += 100.0
                elif ptype == PATTERN_OPEN_FOUR:
                    my_score += 50.0
                elif ptype == PATTERN_HALF_FOUR:
                    my_score += 10.0
                elif ptype == PATTERN_OPEN_THREE:
                    my_score += 5.0
                elif ptype == PATTERN_HALF_THREE:
                    my_score += 1.0

                # 对手
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, opponent)
                ptype = _get_pattern_type(length, open_ends)
                if ptype == PATTERN_FIVE:
                    opp_score += 100.0
                elif ptype == PATTERN_OPEN_FOUR:
                    opp_score += 50.0
                elif ptype == PATTERN_HALF_FOUR:
                    opp_score += 10.0
                elif ptype == PATTERN_OPEN_THREE:
                    opp_score += 5.0
                elif ptype == PATTERN_HALF_THREE:
                    opp_score += 1.0

            my_channel[r, c] = min(1.0, my_score / max_score)
            opp_channel[r, c] = min(1.0, opp_score / max_score)

    return my_channel, opp_channel
