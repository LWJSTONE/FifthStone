"""
VCT/VCF 威胁空间搜索 + 必走着法快速检测 (V3 — gap模式修复版)
=========================================================
V3 修复:
  1. 新增 _analyze_line_pattern_with_gap: 检测断四/断三等gap模式
  2. _find_four_blocking_moves: 同时返回断四的gap堵点
  3. _get_threat_type_at / _count_threats_at: 综合连续+gap分析
  4. find_must_move: 双威胁检测包含gap模式
  5. VCT深度修正: depth_limit - 1 (而非 - 2)
  6. VCF/VCT攻防: 均检测gap模式
  7. 棋型常量重排序: 数值越大=威胁越强, 修正比较逻辑

V2 修复:
  1. VCF: 只搜索堵四着法 + 全称逻辑(所有堵法后都能VCF才算赢)
  2. VCT: 试所有强制防御着法 + 全称逻辑
  3. 双威胁检测: 双冲四、冲四活三、双活三
  4. 必走着法: 包含所有双威胁模式
  5. 模式注入先验: 更精确的攻守综合评分

这些搜索空间极小(只搜威胁着法), 但能发现 MCTS 极难找到的强制胜路线。
"""

import numpy as np
from numba import njit

from config import (
    BOARD_SIZE, BOARD_SQUARES, WIN_LENGTH,
    NUMBA_CACHE
)

# ======================== 常量 ========================
EMPTY = 0
BLACK = 1
WHITE = 2

DIRECTIONS = np.array([[0, 1], [1, 0], [1, 1], [1, -1]], dtype=np.int32)
NUM_DIRS = 4

# 棋型编码 (V3修复: 数值越大=威胁越强, 使 >/>=/< 比较正确)
PATTERN_NONE = 0
PATTERN_HALF_TWO = 1        # 眠二
PATTERN_OPEN_TWO = 2        # 活二
PATTERN_HALF_THREE = 3      # 眠三
PATTERN_OPEN_THREE = 4      # 活三
PATTERN_HALF_FOUR = 5       # 冲四/嵌五
PATTERN_OPEN_FOUR = 6       # 活四
PATTERN_FIVE = 7            # 五连


# ======================== Numba JIT 核心函数 ========================

@njit(cache=NUMBA_CACHE)
def _count_consecutive(board, r, c, dr, dc, color):
    """从(r,c)沿(dr,dc)方向统计连续同色棋子数(不含起点)"""
    count = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        count += 1
        nr += dr
        nc += dc
    return count


@njit(cache=NUMBA_CACHE)
def _analyze_line_pattern(board, r, c, dr, dc, color):
    """分析从(r,c)出发在(dr,dc)方向上的棋型 → (total_length, open_ends)"""
    pos = _count_consecutive(board, r, c, dr, dc, color)
    neg = _count_consecutive(board, r, c, -dr, -dc, color)
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
def _analyze_line_pattern_with_gap(board, r, c, dr, dc, color):
    """
    Gap-aware line pattern analysis (断四/断三检测)
    ===============================================
    扫描所有包含(r,c)的5格窗口, 检测含1个gap的模式:
      - 断四: 4子+1空在5格窗口 → 等价于冲四(堵gap即防五)
        例: O_OOO, OO_OO, OOO_O
      - 断三: 3子+1内部gap在5格窗口 → 等价于活三/眠三
        例: _O_OO_, _OO_O_

    返回: (effective_length, open_ends, gap_pos)
      effective_length: 4=断四, 3=断三, 0=无gap模式
      open_ends: 开放端数(用于棋型分类)
      gap_pos: gap位置的flat index, -1=无
    """
    best_length = 0
    best_open_ends = 0
    best_gap_pos = -1

    # 尝试所有包含(r,c)的5格窗口
    for start_offset in range(-4, 1):
        stone_count = 0
        empty_count = 0
        opponent_count = 0
        gap_flat = -1

        cell_vals = np.zeros(5, dtype=np.int32)   # 0=empty, 1=stone, 2=opponent
        cell_flats = np.zeros(5, dtype=np.int32)

        valid = True
        for i in range(5):
            cr = r + (start_offset + i) * dr
            cc = c + (start_offset + i) * dc
            if cr < 0 or cr >= BOARD_SIZE or cc < 0 or cc >= BOARD_SIZE:
                valid = False
                break
            flat = cr * BOARD_SIZE + cc
            cell_flats[i] = flat
            cell = board[cr, cc]
            if cr == r and cc == c:
                # 正在评估的位置, 视为己方棋子
                cell_vals[i] = 1
                stone_count += 1
            elif cell == color:
                cell_vals[i] = 1
                stone_count += 1
            elif cell == EMPTY:
                cell_vals[i] = 0
                empty_count += 1
            else:
                cell_vals[i] = 2
                opponent_count += 1

        if not valid or opponent_count > 0:
            continue

        # ---- 断四: 4子 + 1空 ----
        if empty_count == 1 and stone_count == 4:
            for i in range(5):
                if cell_vals[i] == 0:
                    gap_flat = cell_flats[i]
                    break

            # 断四等价于冲四: 只有一个必堵点(gap)
            if best_length < 4:
                best_length = 4
                best_open_ends = 1
                best_gap_pos = gap_flat

        # ---- 断三: 3子 + 2空, 其中1空为"内部gap" ----
        elif empty_count == 2 and stone_count == 3:
            # 检查是否存在"内部gap": 填入后形成4连
            for g in range(5):
                if cell_vals[g] != 0:
                    continue
                # 从gap位置向两侧数连续己方棋子数(含gap本身)
                consec = 1
                for k in range(g - 1, -1, -1):
                    if cell_vals[k] == 1:
                        consec += 1
                    else:
                        break
                for k in range(g + 1, 5):
                    if cell_vals[k] == 1:
                        consec += 1
                    else:
                        break

                if consec >= 4:
                    # 内部gap: 填入后形成4连
                    # 计算4连的范围
                    four_start = g
                    for k in range(g - 1, -1, -1):
                        if cell_vals[k] == 1:
                            four_start = k
                        else:
                            break
                    four_end = g
                    for k in range(g + 1, 5):
                        if cell_vals[k] == 1:
                            four_end = k
                        else:
                            break

                    # 计算开放端
                    open_ends = 0
                    if four_start > 0 and cell_vals[four_start - 1] == 0:
                        open_ends += 1
                    elif four_start == 0:
                        br2 = r + (start_offset - 1) * dr
                        bc2 = c + (start_offset - 1) * dc
                        if 0 <= br2 < BOARD_SIZE and 0 <= bc2 < BOARD_SIZE and board[br2, bc2] == EMPTY:
                            open_ends += 1
                    if four_end < 4 and cell_vals[four_end + 1] == 0:
                        open_ends += 1
                    elif four_end == 4:
                        ar2 = r + (start_offset + 5) * dr
                        ac2 = c + (start_offset + 5) * dc
                        if 0 <= ar2 < BOARD_SIZE and 0 <= ac2 < BOARD_SIZE and board[ar2, ac2] == EMPTY:
                            open_ends += 1

                    if 3 > best_length or (3 == best_length and open_ends > best_open_ends):
                        best_length = 3
                        best_open_ends = open_ends
                        best_gap_pos = cell_flats[g]
                    break  # 找到内部gap, 不再检查其他空位

    return (best_length, best_open_ends, best_gap_pos)


@njit(cache=NUMBA_CACHE)
def _get_pattern_type(length, open_ends):
    """将(连子数, 开放端)映射为棋型编码 (V3: 数值越大=威胁越强)"""
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


@njit(cache=NUMBA_CACHE)
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


# ======================== VCF 搜索 (V3 gap修复版) ========================

@njit(cache=NUMBA_CACHE)
def _find_four_blocking_moves(board, four_r, four_c, attacker):
    """
    找到堵住冲四的着法位置 (V3: 包含断四gap检测)
    ================================================
    当(four_r, four_c)形成冲四(4连+1开放端)或断四(4子1空)时,
    返回堵住这个四的位置

    返回: (positions, count) — 堵四位置, 最多4个
    """
    positions = np.empty(4, dtype=np.int32)
    count = 0
    seen = np.zeros(BOARD_SQUARES, dtype=np.int32)

    for d in range(NUM_DIRS):
        dr = DIRECTIONS[d, 0]
        dc = DIRECTIONS[d, 1]
        pos_count = _count_consecutive(board, four_r, four_c, dr, dc, attacker)
        neg_count = _count_consecutive(board, four_r, four_c, -dr, -dc, attacker)
        total = pos_count + neg_count + 1

        if total == 4:
            # 找到了连续四连的方向
            # 正方向端点
            er, ec = four_r + dr * (pos_count + 1), four_c + dc * (pos_count + 1)
            if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE and board[er, ec] == EMPTY:
                flat = er * BOARD_SIZE + ec
                if seen[flat] == 0 and count < 4:
                    seen[flat] = 1
                    positions[count] = flat
                    count += 1
            # 反方向端点
            br, bc = four_r - dr * (neg_count + 1), four_c - dc * (neg_count + 1)
            if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE and board[br, bc] == EMPTY:
                flat = br * BOARD_SIZE + bc
                if seen[flat] == 0 and count < 4:
                    seen[flat] = 1
                    positions[count] = flat
                    count += 1

        # 检查断四(gap four): 4子+1空在5格窗口内
        for start_offset in range(-4, 1):
            stone_count = 0
            empty_count = 0
            opponent_count = 0
            gap_flat = -1

            valid = True
            for i in range(5):
                cr = four_r + (start_offset + i) * dr
                cc = four_c + (start_offset + i) * dc
                if cr < 0 or cr >= BOARD_SIZE or cc < 0 or cc >= BOARD_SIZE:
                    valid = False
                    break
                flat = cr * BOARD_SIZE + cc
                cell = board[cr, cc]
                if cr == four_r and cc == four_c:
                    stone_count += 1
                elif cell == attacker:
                    stone_count += 1
                elif cell == EMPTY:
                    empty_count += 1
                    gap_flat = flat
                else:
                    opponent_count += 1

            if not valid or opponent_count > 0:
                continue

            if empty_count == 1 and stone_count == 4:
                # 断四: 空位是必须堵的位置
                if gap_flat >= 0 and seen[gap_flat] == 0 and count < 4:
                    seen[gap_flat] = 1
                    positions[count] = gap_flat
                    count += 1

    return positions, count


@njit(cache=NUMBA_CACHE)
def _find_forced_defense_moves(board, attacker, defender, buf):
    """
    找到对手的强制防御着法 (V3: 包含断四检测)
    ==========================================
    对手必须防守的着法包括:
      1. 对手能直接五连的位置
      2. 堵攻击方冲四/断四的位置

    返回: (positions, count)
    """
    count = 0

    # 1. 对手五连(对手可以忽略防守直接赢)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            if _check_five(board, r, c, defender):
                if count < 30:
                    buf[count] = r * BOARD_SIZE + c
                    count += 1
                # 对手有五连, 这是必须防守的
                return buf, count

    # 2. 堵攻击方的冲四/断四
    # 先找到攻击方的所有冲四/断四位置
    four_positions = np.empty(BOARD_SQUARES, dtype=np.int32)
    four_count = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            is_four = False
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, attacker)
                if length >= 5:
                    # 攻击方直接五连, 不需要防守
                    return buf, 0
                if length == 4 and open_ends >= 1:
                    is_four = True
                    break
                # 检查断四
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, attacker
                )
                if gap_length == 4:
                    is_four = True
                    break

            if is_four:
                four_positions[four_count] = r * BOARD_SIZE + c
                four_count += 1

    # 对于每个冲四/断四位置, 找到堵住它的位置
    seen = np.zeros(BOARD_SQUARES, dtype=np.int32)
    for i in range(four_count):
        fr, fc = four_positions[i] // BOARD_SIZE, four_positions[i] % BOARD_SIZE
        # 在这个位置落子形成冲四/断四
        board[fr, fc] = attacker
        blocking_positions, blocking_count = _find_four_blocking_moves(
            board, fr, fc, attacker
        )
        board[fr, fc] = EMPTY

        for j in range(blocking_count):
            bpos = blocking_positions[j]
            if seen[bpos] == 0 and count < 30:
                seen[bpos] = 1
                buf[count] = bpos
                count += 1

    return buf, count


@njit(cache=NUMBA_CACHE)
def vcf_search(board, attacker, depth_limit=20):
    """
    VCF搜索 V3: 正确的连续冲四取胜搜索 (含断四检测)
    ================================================
    修复:
      - 搜索冲四+断四着法
      - 对手只搜堵四位置(含断四gap堵点)
      - 使用全称逻辑: 攻击方赢 = 对手所有堵法后攻击方都能VCF

    返回: winning_move 位置(0-224), -1 表示未找到
    """
    if depth_limit <= 0:
        return -1

    defender = 3 - attacker

    # 找攻击方的冲四/断四着法
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            # 检查这步能否形成冲四/断四或五连
            is_four = False
            is_five = False
            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, attacker)
                if length >= 5:
                    is_five = True
                    break
                if length == 4 and open_ends >= 1:
                    is_four = True
                # 检查断四
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, attacker
                )
                if gap_length == 4:
                    is_four = True

            if is_five:
                # 直接五连, 找到了
                return r * BOARD_SIZE + c

            if not is_four:
                continue

            # 落子(冲四/断四)
            board[r, c] = attacker

            # 检查对手是否有直接五连(对手可能选择不堵而自己五连)
            opponent_can_win = False
            for r2 in range(BOARD_SIZE):
                for c2 in range(BOARD_SIZE):
                    if board[r2, c2] != EMPTY:
                        continue
                    if _check_five(board, r2, c2, defender):
                        opponent_can_win = True
                        break
                if opponent_can_win:
                    break

            if opponent_can_win:
                board[r, c] = EMPTY
                continue

            # 找堵四的位置(含断四gap堵点)
            blocking_positions, blocking_count = _find_four_blocking_moves(
                board, r, c, attacker
            )

            if blocking_count == 0:
                # 冲四无法被堵(活四?) — 实际上如果有开放端就应该有堵点
                board[r, c] = EMPTY
                return r * BOARD_SIZE + c

            # V2 核心: 全称逻辑 — 对手所有堵法后攻击方都能VCF才算赢
            all_defense_fail = True
            for j in range(blocking_count):
                bpos = blocking_positions[j]
                br, bc = bpos // BOARD_SIZE, bpos % BOARD_SIZE

                board[br, bc] = defender

                # 递归: 攻击方是否还能VCF
                result = vcf_search(board, attacker, depth_limit - 1)

                board[br, bc] = EMPTY

                if result < 0:
                    # 存在一个堵法使得攻击方无法VCF → 攻击方不能通过这条路线赢
                    all_defense_fail = False
                    break

            board[r, c] = EMPTY

            if all_defense_fail:
                return r * BOARD_SIZE + c

    return -1


# ======================== VCT 搜索 (V3 gap修复版) ========================

@njit(cache=NUMBA_CACHE)
def _get_threat_type_at(board, r, c, color):
    """获取(r,c)落子后能形成的最强棋型 (包含gap/断四模式)"""
    best = PATTERN_NONE
    for d in range(NUM_DIRS):
        dr = DIRECTIONS[d, 0]
        dc = DIRECTIONS[d, 1]
        # 连续模式
        length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, color)
        ptype = _get_pattern_type(length, open_ends)
        if ptype > best:
            best = ptype
        # gap模式
        gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
            board, r, c, dr, dc, color
        )
        if gap_length > 0:
            gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
            if gap_ptype > best:
                best = gap_ptype
    return best


@njit(cache=NUMBA_CACHE)
def _count_threats_at(board, r, c, color, min_threat):
    """统计(r,c)落子后形成的威胁数量(≥min_threat的棋型数, 包含gap模式)"""
    count = 0
    for d in range(NUM_DIRS):
        dr = DIRECTIONS[d, 0]
        dc = DIRECTIONS[d, 1]
        # 连续模式
        length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, color)
        ptype = _get_pattern_type(length, open_ends)
        # gap模式 — 取两者中更强的
        gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
            board, r, c, dr, dc, color
        )
        if gap_length > 0:
            gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
            if gap_ptype > ptype:
                ptype = gap_ptype
        if ptype >= min_threat:
            count += 1
    return count


@njit(cache=NUMBA_CACHE)
def _find_vct_defense_moves(board, attacker, defender, buf):
    """
    找到VCT中对手的强制防御着法 (V3: 包含断三gap堵点)
    =================================================
    对手必须防守:
      1. 对手直接五连
      2. 堵攻击方的冲四/断四(必须堵)
      3. 堵攻击方的活四(必须堵)
      4. 堵攻击方的活三/断三(通常必须堵)

    返回: (positions, count)
    """
    count = 0

    # 1. 对手五连
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            if _check_five(board, r, c, defender):
                if count < 30:
                    buf[count] = r * BOARD_SIZE + c
                    count += 1
                return buf, count

    # 2. 堵攻击方的冲四/活四/断四
    seen = np.zeros(BOARD_SQUARES, dtype=np.int32)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            threat_type = _get_threat_type_at(board, r, c, attacker)

            if threat_type >= PATTERN_HALF_FOUR:
                # 冲四/断四或更强: 对手必须堵
                board[r, c] = attacker
                blocking_positions, blocking_count = _find_four_blocking_moves(
                    board, r, c, attacker
                )
                board[r, c] = EMPTY

                for j in range(blocking_count):
                    bpos = blocking_positions[j]
                    if seen[bpos] == 0 and count < 30:
                        seen[bpos] = 1
                        buf[count] = bpos
                        count += 1

            if threat_type == PATTERN_OPEN_FOUR:
                # 活四: 对手堵任一端即可, 但如果不堵必输
                # 堵活四的位置已经在上面处理了
                pass

    # 3. 堵攻击方的活三/断三 (可选防御, 但通常必须堵)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            threat_type = _get_threat_type_at(board, r, c, attacker)
            if threat_type == PATTERN_OPEN_THREE:
                # 活三/断三: 找堵它的位置
                board[r, c] = attacker

                # 3a. 连续活三的堵点
                for d in range(NUM_DIRS):
                    dr = DIRECTIONS[d, 0]
                    dc = DIRECTIONS[d, 1]
                    length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, attacker)
                    if length == 3 and open_ends >= 2:
                        # 堵正端
                        pos_count = _count_consecutive(board, r, c, dr, dc, attacker)
                        er, ec = r + dr * (pos_count + 1), c + dc * (pos_count + 1)
                        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                            epos = er * BOARD_SIZE + ec
                            if seen[epos] == 0 and count < 30:
                                seen[epos] = 1
                                buf[count] = epos
                                count += 1
                        # 堵反端
                        neg_count = _count_consecutive(board, r, c, -dr, -dc, attacker)
                        br, bc = r - dr * (neg_count + 1), c - dc * (neg_count + 1)
                        if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                            bpos2 = br * BOARD_SIZE + bc
                            if seen[bpos2] == 0 and count < 30:
                                seen[bpos2] = 1
                                buf[count] = bpos2
                                count += 1

                # 3b. 断三的gap堵点
                for d in range(NUM_DIRS):
                    dr = DIRECTIONS[d, 0]
                    dc = DIRECTIONS[d, 1]
                    gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                        board, r, c, dr, dc, attacker
                    )
                    if gap_length == 3 and gap_open_ends >= 2 and gap_pos >= 0:
                        # 断活三: gap位置是关键堵点
                        if seen[gap_pos] == 0 and count < 30:
                            seen[gap_pos] = 1
                            buf[count] = gap_pos
                            count += 1

                board[r, c] = EMPTY

    # 4. 对手自己的高价值着法(可以不堵而进攻)
    # 简化: 不在此处处理, 让VCT递归自行处理

    return buf, count


@njit(cache=NUMBA_CACHE)
def vct_search(board, attacker, depth_limit=12):
    """
    VCT搜索 V3: 正确的连续威胁取胜搜索 (含gap模式)
    ==============================================
    修复:
      - 先试VCF
      - 只搜活三/断三及以上威胁着法
      - 对手搜所有强制防御着法(含gap堵点)
      - 全称逻辑: 对手所有强制防御后攻击方都能VCT才算赢
      - V3: depth递减修正为-1 (而非-2)

    返回: winning_move 位置, -1 表示未找到
    """
    if depth_limit <= 0:
        return -1

    defender = 3 - attacker

    # 1. 先试 VCF
    vcf_result = vcf_search(board, attacker, min(depth_limit * 2, 20))
    if vcf_result >= 0:
        return vcf_result

    # 2. 找活三/断三/冲四着法
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            # 检查这步能形成什么棋型 (包含gap模式)
            best_threat = _get_threat_type_at(board, r, c, attacker)

            # 只搜活三及以上的威胁着法 (V3: 数值越大=越强, >=PATTERN_OPEN_THREE)
            if best_threat < PATTERN_OPEN_THREE:
                continue

            # 落子
            board[r, c] = attacker

            # 检查是否直接五连
            if _check_five(board, r, c, attacker):
                board[r, c] = EMPTY
                return r * BOARD_SIZE + c

            # 找对手的强制防御着法
            defense_buf = np.empty(30, dtype=np.int32)
            defense_moves, defense_count = _find_vct_defense_moves(
                board, attacker, defender, defense_buf
            )

            if defense_count == 0:
                # 没有强制防御 → 攻击方赢了
                board[r, c] = EMPTY
                return r * BOARD_SIZE + c

            # V2 核心: 全称逻辑
            all_defense_fail = True
            for j in range(defense_count):
                dpos = defense_moves[j]
                dr2, dc2 = dpos // BOARD_SIZE, dpos % BOARD_SIZE

                board[dr2, dc2] = defender

                # 递归: 攻击方是否还能VCT
                # V3修复: depth_limit - 1 (而非 - 2)
                # 理由: depth_limit表示攻击方还能走多少步,
                #        每次VCT步骤(攻击+防御)应只消耗1层深度
                result = vct_search(board, attacker, depth_limit - 1)

                board[dr2, dc2] = EMPTY

                if result < 0:
                    all_defense_fail = False
                    break

            board[r, c] = EMPTY

            if all_defense_fail:
                return r * BOARD_SIZE + c

    return -1


# ======================== 必走着法检测 (V3 gap增强版) ========================

@njit(cache=NUMBA_CACHE)
def find_must_move(board, current_player):
    """
    必走着法检测 V3 — 包含双威胁+gap模式检测
    ========================================
    返回: (must_move_idx, move_type)
      must_move_idx: 0-224, -1=无必走
      move_type:
        1 = 己方五连(必胜)
        2 = 堵对手五连(必防)
        3 = 己方活四(必胜)
        4 = 堵对手活四(必防)
        5 = 己方双冲四(必胜)
        6 = 己方冲四活三(必胜)
        7 = 己方双活三(极大概率胜)
        8 = 堵对手双冲四(必防)
        9 = 堵对手冲四活三(必防)
    """
    opponent = 3 - current_player

    # 1. 己方五连? → 必胜
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            if _check_five(board, r, c, current_player):
                return (r * BOARD_SIZE + c, 1)

    # 2. 对手五连? → 必防
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            if _check_five(board, r, c, opponent):
                return (r * BOARD_SIZE + c, 2)

    # 3. 己方活四? → 必胜
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            threat = _get_threat_type_at(board, r, c, current_player)
            if threat == PATTERN_OPEN_FOUR:
                return (r * BOARD_SIZE + c, 3)

    # 4. 对手活四? → 必防(堵活四的唯一位置)
    opp_open_four_pos = -1
    opp_open_four_count = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue
            threat = _get_threat_type_at(board, r, c, opponent)
            if threat == PATTERN_OPEN_FOUR:
                opp_open_four_count += 1
                if opp_open_four_pos < 0:
                    opp_open_four_pos = r * BOARD_SIZE + c

    if opp_open_four_count > 0:
        # 对手有活四, 需要堵 — 但活四有两个端点, 只堵一个不够
        # 如果对手有多个活四, 无法防守
        if opp_open_four_count >= 2:
            # 双活四, 无法防守
            # 返回任意一个, 表示必须走(实际上已经输了)
            return (opp_open_four_pos, 4)
        else:
            # 单活四: 必须同时形成己方冲四或活四来反杀, 否则必输
            # 简化: 返回对手活四位置(让MCTS决定怎么堵)
            return (opp_open_four_pos, 4)

    # 5. 双威胁检测 — 五子棋最重要的战术模式! (V3: 包含gap模式)
    # 己方双威胁
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            # 统计己方在各方向形成的威胁 (连续+gap取最强)
            half_four_count = 0  # 冲四数(含断四)
            open_three_count = 0  # 活三数(含断三)

            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                # 连续模式
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                ptype = _get_pattern_type(length, open_ends)
                # gap模式 — 取更强者
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, current_player
                )
                if gap_length > 0:
                    gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
                    if gap_ptype > ptype:
                        ptype = gap_ptype

                if ptype == PATTERN_HALF_FOUR:
                    half_four_count += 1
                elif ptype == PATTERN_OPEN_THREE:
                    open_three_count += 1

            # 双冲四: 必胜(两个冲四, 对手只能堵一个)
            if half_four_count >= 2:
                return (r * BOARD_SIZE + c, 5)

            # 冲四+活三: 必胜(冲四迫使对手堵, 然后活三变活四)
            if half_four_count >= 1 and open_three_count >= 1:
                return (r * BOARD_SIZE + c, 6)

            # 双活三: 极大概率胜(一步形成两个活三)
            if open_three_count >= 2:
                return (r * BOARD_SIZE + c, 7)

    # 6. 对手双威胁检测 (V3: 包含gap模式)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                continue

            half_four_count = 0
            open_three_count = 0

            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]
                # 连续模式
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, opponent)
                ptype = _get_pattern_type(length, open_ends)
                # gap模式 — 取更强者
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, opponent
                )
                if gap_length > 0:
                    gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
                    if gap_ptype > ptype:
                        ptype = gap_ptype

                if ptype == PATTERN_HALF_FOUR:
                    half_four_count += 1
                elif ptype == PATTERN_OPEN_THREE:
                    open_three_count += 1

            if half_four_count >= 2:
                return (r * BOARD_SIZE + c, 8)

            if half_four_count >= 1 and open_three_count >= 1:
                return (r * BOARD_SIZE + c, 9)

    return (-1, 0)


# ======================== 模式注入 MCTS 先验 (V3 gap增强版) ========================

@njit(cache=NUMBA_CACHE)
def compute_pattern_prior_bonus(board, current_player):
    """
    为每个合法着法计算基于棋型的先验加分 (V3: 包含gap模式)
    ====================================================

    返回: bonus[225] 数组
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
            my_half_four = 0
            my_open_three = 0
            opp_half_four = 0
            opp_open_three = 0

            for d in range(NUM_DIRS):
                dr = DIRECTIONS[d, 0]
                dc = DIRECTIONS[d, 1]

                # 己方棋型 (连续+gap取最强)
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                ptype = _get_pattern_type(length, open_ends)
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, current_player
                )
                if gap_length > 0:
                    gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
                    if gap_ptype > ptype:
                        ptype = gap_ptype
                if ptype > my_best:
                    my_best = ptype
                if ptype == PATTERN_HALF_FOUR:
                    my_half_four += 1
                elif ptype == PATTERN_OPEN_THREE:
                    my_open_three += 1

                # 对手棋型 (连续+gap取最强)
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, opponent)
                ptype = _get_pattern_type(length, open_ends)
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, opponent
                )
                if gap_length > 0:
                    gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
                    if gap_ptype > ptype:
                        ptype = gap_ptype
                if ptype > opp_best:
                    opp_best = ptype
                if ptype == PATTERN_HALF_FOUR:
                    opp_half_four += 1
                elif ptype == PATTERN_OPEN_THREE:
                    opp_open_three += 1

            # 进攻加分
            if my_best == PATTERN_FIVE:
                bonus[idx] = 100.0
            elif my_best == PATTERN_OPEN_FOUR:
                bonus[idx] = 50.0
            elif my_best == PATTERN_HALF_FOUR:
                bonus[idx] = 20.0
            elif my_best == PATTERN_OPEN_THREE:
                bonus[idx] = 15.0
            elif my_best == PATTERN_HALF_THREE:
                bonus[idx] = 3.0

            # 防守加分
            if opp_best == PATTERN_FIVE:
                bonus[idx] = max(bonus[idx], 80.0)
            elif opp_best == PATTERN_OPEN_FOUR:
                bonus[idx] = max(bonus[idx], 40.0)
            elif opp_best == PATTERN_HALF_FOUR:
                bonus[idx] = max(bonus[idx], 15.0)
            elif opp_best == PATTERN_OPEN_THREE:
                bonus[idx] = max(bonus[idx], 10.0)
            elif opp_best == PATTERN_HALF_THREE:
                bonus[idx] = max(bonus[idx], 2.0)

            # V2: 双威胁加分
            if my_half_four >= 2:
                bonus[idx] += 60.0     # 双冲四
            if my_half_four >= 1 and my_open_three >= 1:
                bonus[idx] += 45.0     # 冲四+活三
            if my_open_three >= 2:
                bonus[idx] += 35.0     # 双活三

            # V2: 堵对手双威胁
            if opp_half_four >= 2:
                bonus[idx] += 50.0     # 堵双冲四
            if opp_half_four >= 1 and opp_open_three >= 1:
                bonus[idx] += 35.0     # 堵冲四+活三
            if opp_open_three >= 2:
                bonus[idx] += 25.0     # 堵双活三

    return bonus


# ======================== 领域知识特征通道 ========================

@njit(cache=NUMBA_CACHE)
def compute_pattern_feature_channels(board, current_player):
    """计算两个领域知识特征通道 (归一化到0-1, V3: 包含gap模式)"""
    my_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp_channel = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
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

                # 己方 (连续+gap取最强)
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, current_player)
                ptype = _get_pattern_type(length, open_ends)
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, current_player
                )
                if gap_length > 0:
                    gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
                    if gap_ptype > ptype:
                        ptype = gap_ptype
                if ptype == PATTERN_FIVE:        my_score += 100.0
                elif ptype == PATTERN_OPEN_FOUR: my_score += 50.0
                elif ptype == PATTERN_HALF_FOUR: my_score += 10.0
                elif ptype == PATTERN_OPEN_THREE: my_score += 5.0
                elif ptype == PATTERN_HALF_THREE: my_score += 1.0

                # 对手 (连续+gap取最强)
                length, open_ends = _analyze_line_pattern(board, r, c, dr, dc, opponent)
                ptype = _get_pattern_type(length, open_ends)
                gap_length, gap_open_ends, gap_pos = _analyze_line_pattern_with_gap(
                    board, r, c, dr, dc, opponent
                )
                if gap_length > 0:
                    gap_ptype = _get_pattern_type(gap_length, gap_open_ends)
                    if gap_ptype > ptype:
                        ptype = gap_ptype
                if ptype == PATTERN_FIVE:        opp_score += 100.0
                elif ptype == PATTERN_OPEN_FOUR: opp_score += 50.0
                elif ptype == PATTERN_HALF_FOUR: opp_score += 10.0
                elif ptype == PATTERN_OPEN_THREE: opp_score += 5.0
                elif ptype == PATTERN_HALF_THREE: opp_score += 1.0

            my_channel[r, c] = min(1.0, my_score / max_score)
            opp_channel[r, c] = min(1.0, opp_score / max_score)

    return my_channel, opp_channel
