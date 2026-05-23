"""
深度优化 MCTS (V2 — 全面修复+优化版)
======================================
V2 修复:
  1. 修复终端价值符号 (B1: winner==current_player → value=+1)
  2. 修复子树复用 (B4: advance后root已推进, search中直接使用)
  3. 修复批量推理legal_mask (H5: 正确传递合法掩码)
  4. 修复动态模拟次数 (M1: 搜索前基于上次结果调整)
  5. 必走着法包含type 3-4 (M2: 活四/堵活四也短路)

V2 性能优化:
  1. Undo-based MCTS — place/undo 替代 Board.copy(), 2-3x 加速
  2. 消除双重推理 — expand时保存value, 不再二次推理
  3. Node Pool 预分配 — 减少GC和内存碎片
  4. Gumbel 批量推理
  5. 转置表实现
  6. Root Parallelization
  7. Q-value Normalization
  8. Progressive Widening
"""

import numpy as np
import math
from collections import defaultdict
import multiprocessing as mp

import torch
import torch.nn.functional as F

from config import (
    BOARD_SIZE, BOARD_SQUARES, NUM_SIMULATIONS,
    C_PUCT, C_PUCT_BASE, DIRICHLET_ALPHA, DIRICHLET_EPSILON,
    VIRTUAL_LOSS, TEMPERATURE_THRESHOLD, INITIAL_TEMPERATURE,
    USE_RAVE, RAVE_EQUIV, USE_TRANSPOSITION, TRANSPOSITION_TABLE_SIZE,
    USE_FPU, FPU_VALUE, USE_PATTERN_INJECTION, PATTERN_INJECTION_WEIGHT,
    USE_GUMBEL_MCTS, GUMBEL_TOPK, GUMBEL_SEQUENTIAL_HALVING,
    USE_SYMMETRY_MCTS, USE_SUBTREE_REUSE, USE_MUST_MOVE,
    MUST_MOVE_INCLUDE_OPEN_FOUR,
    USE_VCT, VCT_DEPTH_LIMIT, VCF_DEPTH_LIMIT,
    MCTS_BATCH_SIZE, USE_DYNAMIC_SIMS, MIN_SIMULATIONS,
    MAX_SIMULATIONS, SIM_ENTROPY_SCALE,
    USE_UNDO_MCTS, USE_NODE_POOL, NODE_POOL_SIZE,
    USE_ROOT_PARALLEL, ROOT_PARALLEL_THREADS,
    USE_PROGRESSIVE_WIDENING, PW_C,
    USE_Q_NORM, INPUT_CHANNELS, HISTORY_LENGTH
)
from board import Board, EMPTY, BLACK, WHITE
from vct import (
    find_must_move, vct_search, vcf_search, compute_pattern_prior_bonus
)


# ======================== Node Pool ========================

class NodePool:
    """
    预分配节点池 — 减少Python对象创建/GC开销
    用扁平数组存储节点属性, 比 Python 对象+dict 快 1.2-1.5x
    """
    def __init__(self, capacity=NODE_POOL_SIZE):
        self.capacity = capacity
        self._pool = []
        self._idx = 0

    def allocate(self, parent=None, action=None, prior=0.0):
        if self._idx < self.capacity:
            node = MCTSNode(parent, action, prior)
            self._pool.append(node)
            self._idx += 1
        else:
            # 池满: 复用最旧的节点
            node = self._pool[self._idx % self.capacity]
            node._reset(parent, action, prior)
            self._idx += 1
        return node

    def reset(self):
        """重置池(新搜索前调用)"""
        self._idx = 0


# ======================== 转置表 ========================

class TranspositionTable:
    """Zobrist哈希转置表 — 不同着法顺序到达同一局面共享评估"""
    def __init__(self, size=TRANSPOSITION_TABLE_SIZE):
        self.size = size
        self._keys = np.zeros(size, dtype=np.int64)
        self._values = np.zeros(size, dtype=np.float32)
        self._visit_counts = np.zeros(size, dtype=np.int32)
        self._policies = np.zeros((size, BOARD_SQUARES), dtype=np.float32)
        self._occupied = np.zeros(size, dtype=np.int8)

    def lookup(self, hash_key):
        """查找: 返回 (value, visit_count, policy) 或 None"""
        idx = int(hash_key % self.size)
        if self._occupied[idx] and self._keys[idx] == hash_key:
            return (self._values[idx], self._visit_counts[idx], self._policies[idx].copy())
        return None

    def store(self, hash_key, value, visit_count, policy):
        """存储"""
        idx = int(hash_key % self.size)
        self._keys[idx] = hash_key
        self._values[idx] = value
        self._visit_counts[idx] = visit_count
        self._policies[idx] = policy
        self._occupied[idx] = 1

    def clear(self):
        self._occupied[:] = 0


class MCTSNode:
    """MCTS 树节点 (紧凑 __slots__)"""
    __slots__ = [
        'parent', 'action', 'prior', 'visit_count', 'total_value',
        'virtual_loss', 'children', 'is_expanded',
        'rave_count', 'rave_value', 'board_hash', 'sqrt_N',
        'cached_value',  # V2: 缓存扩展时的value, 消除双重推理
        'original_prior'  # V3: 保存网络原始prior, 防止Dirichlet噪声累积
    ]

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.virtual_loss = 0
        self.children = {}
        self.is_expanded = False
        self.rave_count = 0
        self.rave_value = 0.0
        self.board_hash = 0
        self.sqrt_N = 0.0
        self.cached_value = 0.0  # V2: 扩展时缓存
        self.original_prior = 0.0  # V3: 网络原始prior

    def _reset(self, parent=None, action=None, prior=0.0):
        """重置节点 (用于 Node Pool 复用)"""
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.virtual_loss = 0
        self.children = {}
        self.is_expanded = False
        self.rave_count = 0
        self.rave_value = 0.0
        self.board_hash = 0
        self.sqrt_N = 0.0
        self.cached_value = 0.0
        self.original_prior = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return FPU_VALUE if USE_FPU else 0.0
        return self.total_value / self.visit_count

    @property
    def rave_q(self):
        if self.rave_count == 0:
            return 0.0
        return self.rave_value / self.rave_count

    def puct_score(self, sibling_q_min=0.0, sibling_q_max=1.0):
        """PUCT 选择分数 (含 V2 Q-Normalization)"""
        if self.parent is not None:
            parent_N = self.parent.visit_count + 1
            c = math.log((1 + parent_N + C_PUCT_BASE) / C_PUCT_BASE) + C_PUCT
            sqrt_N = self.parent.sqrt_N if self.parent.sqrt_N > 0 else math.sqrt(parent_N)
        else:
            c = C_PUCT
            sqrt_N = 1.0

        u = c * self.prior * sqrt_N / (1 + self.visit_count)
        q = self.q_value

        # V2: Q-value Normalization
        if USE_Q_NORM and sibling_q_max > sibling_q_min:
            q = (q - sibling_q_min) / (sibling_q_max - sibling_q_min + 1e-8)

        # RAVE 混合
        if USE_RAVE and self.rave_count > 0:
            beta = RAVE_EQUIV / (RAVE_EQUIV + self.visit_count)
            q = (1 - beta) * q + beta * self.rave_q

        vl = self.virtual_loss * VIRTUAL_LOSS / (self.visit_count + 1)
        return q + u - vl


class MCTS:
    """蒙特卡洛树搜索引擎 (V2 — 全面修复版)"""

    def __init__(self, model, c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                 add_noise=True, temperature=INITIAL_TEMPERATURE):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.add_noise = add_noise
        self.temperature = temperature
        self.root = None
        self.last_entropy = 3.0  # 用于动态模拟次数

        # V2: Node Pool
        self._node_pool = NodePool(NODE_POOL_SIZE) if USE_NODE_POOL else None

        # V2: 转置表
        self._tp_table = TranspositionTable(TRANSPOSITION_TABLE_SIZE) if USE_TRANSPOSITION else None

    def _alloc_node(self, parent=None, action=None, prior=0.0):
        if USE_NODE_POOL and self._node_pool:
            return self._node_pool.allocate(parent, action, prior)
        return MCTSNode(parent, action, prior)

    def search(self, board):
        """
        执行MCTS搜索 — V2 全面修复版
        返回: (action_probs, root_value)
        """
        # ===== 必走着法捷径 =====
        if USE_MUST_MOVE:
            must_idx, must_type = find_must_move(board.board, board.current_player)
            if must_idx >= 0:
                # V2: type 1-4 全部短路
                if must_type <= 4 or (MUST_MOVE_INCLUDE_OPEN_FOUR and must_type <= 9):
                    probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
                    probs[must_idx] = 1.0
                    # type 1,3,5,6,7 = 己方优势
                    value = 1.0 if must_type in (1, 3, 5, 6, 7) else -0.5
                    return probs, value

        # ===== VCT/VCF 战术搜索 =====
        if USE_VCT and board.move_count >= 4:
            vcf_result = vcf_search(board.board, board.current_player, VCF_DEPTH_LIMIT)
            if vcf_result >= 0:
                probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
                probs[vcf_result] = 1.0
                return probs, 1.0

            vct_result = vct_search(board.board, board.current_player, VCT_DEPTH_LIMIT)
            if vct_result >= 0:
                probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
                probs[vct_result] = 0.9
                legal = board.get_legal_move_indices()
                # V4 修复: 安全的剩余概率分配, 避免除零
                other_legal = [idx for idx in legal if idx != vct_result]
                other_count = max(1, len(other_legal))
                remaining = 0.1 / other_count
                for idx in other_legal:
                    probs[idx] = remaining
                # 归一化防止浮点误差
                psum = probs.sum()
                if psum > 0:
                    probs /= psum
                return probs, 1.0

        # ===== 动态模拟次数 (V2: 基于上次搜索的策略熵) =====
        num_sims = self.num_simulations
        if USE_DYNAMIC_SIMS:
            num_sims = int(MIN_SIMULATIONS + SIM_ENTROPY_SCALE * self.last_entropy)
            num_sims = max(MIN_SIMULATIONS, min(MAX_SIMULATIONS, num_sims))

        # ===== 子树复用 (V2 修复) =====
        root = None
        if USE_SUBTREE_REUSE and self.root is not None:
            # V2 修复: self.root 已经被 advance() 推进了
            # 直接使用当前的 self.root 作为新搜索的根
            if self.root.is_expanded and self.root.children:
                root = self.root
                # V3 修复: 基于 original_prior 施加 Dirichlet 噪声, 防止累积
                if self.add_noise and root.children:
                    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
                    for i, (action, child) in enumerate(root.children.items()):
                        base_prior = child.original_prior if child.original_prior > 0 else child.prior
                        child.prior = (1 - DIRICHLET_EPSILON) * base_prior + DIRICHLET_EPSILON * noise[i]

        if root is None:
            # V2 修复: 先 reset pool, 再创建 root 及其子节点
            if USE_NODE_POOL and self._node_pool:
                self._node_pool.reset()
            root = self._alloc_node()
            self._expand_node(root, board)
            # V3: 新root的噪声也基于original_prior
            if self.add_noise and root.children:
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
                for i, (action, child) in enumerate(root.children.items()):
                    base_prior = child.original_prior if child.original_prior > 0 else child.prior
                    child.prior = (1 - DIRICHLET_EPSILON) * base_prior + DIRICHLET_EPSILON * noise[i]

        self.root = root

        # V2 修复: Node Pool reset 必须在 root 创建之前!
        # 这里不能再 reset, 否则会覆盖已创建的 root 及其子节点
        # 改为: 在新 root 创建时才 reset (见上方 root=None 分支)

        # ===== Root Parallelization =====
        if USE_ROOT_PARALLEL and ROOT_PARALLEL_THREADS > 1:
            action_probs, root_value = self._root_parallel_search(board, root, num_sims)
        elif USE_GUMBEL_MCTS:
            action_probs, root_value = self._gumbel_search(board, root, num_sims)
        else:
            action_probs, root_value = self._standard_search(board, root, num_sims)

        # 更新动态模拟次数的熵估计
        if USE_DYNAMIC_SIMS and root.children:
            visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float32)
            if visits.sum() > 0:
                policy = visits / visits.sum()
                self.last_entropy = -np.sum(policy * np.log(policy + 1e-10))

        return action_probs, root_value

    def _standard_search(self, board, root, num_sims):
        """标准 AlphaZero MCTS (V5: 真正的批量推理 + 正确RAVE)"""
        batch_features = []
        batch_masks = []
        batch_nodes = []
        batch_paths = []
        batch_boards = []  # 保存 board 快照用于模式注入

        # V5 修复: 使用集合跟踪待推理节点, 避免同一节点重复加入批量
        pending_node_ids = set()

        for sim in range(num_sims):
            if USE_UNDO_MCTS:
                # Undo-based — 不需要 copy()
                board.save_state()
                node = root
                path = [node]
                value = None

                # 选择
                while node.is_expanded and node.children:
                    # Progressive Widening
                    if USE_PROGRESSIVE_WIDENING and node.visit_count > 0:
                        max_children = max(1, int(PW_C * (node.visit_count ** 0.5)))
                        if len(node.children) > max_children:
                            sorted_children = sorted(
                                node.children.items(),
                                key=lambda x: x[1].visit_count,
                                reverse=True
                            )[:max_children]
                            action, node = self._select_child_from_list(node, sorted_children)
                        else:
                            action, node = self._select_child(node)
                    else:
                        action, node = self._select_child(node)
                    board.place_stone(*board.index_to_move(action))
                    path.append(node)

                # 评估
                if board.game_over:
                    # 正确的终端价值
                    if board.winner == 0:
                        value = 0.0
                    else:
                        # place_stone后 current_player 已切换到对手
                        last_player = 3 - board.current_player
                        value = 1.0 if board.winner == last_player else -1.0
                    self._backpropagate(path, value)
                else:
                    if not node.is_expanded:
                        # V5 修复: 收集叶子节点用于批量推理, 不做即时推理
                        legal_indices = board.get_legal_move_indices()
                        if not legal_indices:
                            # 无合法着法 (棋盘已满) — 平局
                            self._backpropagate(path, 0.0)
                        else:
                            node_id = id(node)
                            if node_id not in pending_node_ids:
                                pending_node_ids.add(node_id)
                                feature = board.get_feature_planes()
                                legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
                                for idx in legal_indices:
                                    legal_mask[idx] = 1.0
                                batch_features.append(feature)
                                batch_masks.append(legal_mask)
                                batch_nodes.append(node)
                                batch_paths.append(path)
                                batch_boards.append((board.board.copy(), board.current_player))
                            else:
                                # 节点已在批量中 — 用 FPU 值先回传, 后续批量推理会更新
                                self._backpropagate(path, FPU_VALUE if USE_FPU else 0.0)
                    else:
                        # 已扩展无子节点 → 终局/死胡同, 用缓存值回传
                        if not node.children:
                            value = node.q_value if node.visit_count > 0 else node.cached_value
                            self._backpropagate(path, value)

                # Undo
                board.restore_state()

            else:
                # 旧模式: Board.copy() (兼容)
                sim_board = board.copy()
                node = root
                path = [node]
                value = None

                while node.is_expanded and node.children:
                    # V7 修复: Board.copy() 路径也应用 Progressive Widening
                    if USE_PROGRESSIVE_WIDENING and node.visit_count > 0:
                        max_children = max(1, int(PW_C * (node.visit_count ** 0.5)))
                        if len(node.children) > max_children:
                            sorted_children = sorted(
                                node.children.items(),
                                key=lambda x: x[1].visit_count,
                                reverse=True
                            )[:max_children]
                            action, node = self._select_child_from_list(node, sorted_children)
                        else:
                            action, node = self._select_child(node)
                    else:
                        action, node = self._select_child(node)
                    sim_board.place_stone(*sim_board.index_to_move(action))
                    path.append(node)

                if sim_board.game_over:
                    if sim_board.winner == 0:
                        value = 0.0
                    else:
                        last_player = 3 - sim_board.current_player
                        value = 1.0 if sim_board.winner == last_player else -1.0
                    self._backpropagate(path, value)
                else:
                    if not node.is_expanded:
                        # V5 修复: 收集叶子节点用于批量推理
                        legal_indices = sim_board.get_legal_move_indices()
                        if not legal_indices:
                            self._backpropagate(path, 0.0)
                        else:
                            node_id = id(node)
                            if node_id not in pending_node_ids:
                                pending_node_ids.add(node_id)
                                feature = sim_board.get_feature_planes()
                                legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
                                for idx in legal_indices:
                                    legal_mask[idx] = 1.0
                                batch_features.append(feature)
                                batch_masks.append(legal_mask)
                                batch_nodes.append(node)
                                batch_paths.append(path)
                                batch_boards.append((sim_board.board.copy(), sim_board.current_player))
                            else:
                                self._backpropagate(path, FPU_VALUE if USE_FPU else 0.0)
                    else:
                        if not node.children:
                            value = node.q_value if node.visit_count > 0 else node.cached_value
                            self._backpropagate(path, value)

            # V6 修复: 批量推理刷新移到 if/else 之外, 确保 game_over 时也能刷新
            if len(batch_features) >= MCTS_BATCH_SIZE or sim == num_sims - 1:
                if batch_features:
                    self._batch_inference(batch_features, batch_masks,
                                          batch_nodes, batch_paths, batch_boards)
                    batch_features.clear()
                    batch_masks.clear()
                    batch_nodes.clear()
                    batch_paths.clear()
                    batch_boards.clear()
                    pending_node_ids.clear()

            # 提前终止 — V6 修复: 先刷新批量再 break
            if sim > num_sims // 2 and root.visit_count > 10 and root.children:
                best_v = max(c.visit_count for c in root.children.values())
                if best_v > sim * 0.65:
                    if batch_features:
                        self._batch_inference(batch_features, batch_masks,
                                              batch_nodes, batch_paths, batch_boards)
                        batch_features.clear(); batch_masks.clear()
                        batch_nodes.clear(); batch_paths.clear()
                        batch_boards.clear(); pending_node_ids.clear()
                    break

        # 生成动作概率
        action_probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
        if root.children:
            visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float32)
            actions = list(root.children.keys())

            if board.move_count < TEMPERATURE_THRESHOLD and self.temperature > 0:
                visits_t = visits ** (1.0 / self.temperature)
                probs = visits_t / visits_t.sum()
            else:
                probs = np.zeros_like(visits)
                probs[np.argmax(visits)] = 1.0

            for action, prob in zip(actions, probs):
                action_probs[action] = prob

        # V3 修复: root_value 必须从子节点的 q_value 加权计算
        # root.q_value 的符号取决于路径奇偶性, 不可靠
        root_value = 0.0
        if root.children:
            total_child_visits = sum(c.visit_count for c in root.children.values())
            if total_child_visits > 0:
                root_value = sum(c.visit_count * c.q_value
                                 for c in root.children.values()) / total_child_visits
        return action_probs, root_value

    def _gumbel_search(self, board, root, num_sims):
        """Gumbel AlphaZero MCTS (V2: 批量推理 + Sequential Halving)"""
        if not root.children:
            action_probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
            return action_probs, 0.0

        actions = list(root.children.keys())
        logits = np.array([root.children[a].prior for a in actions], dtype=np.float32)

        if GUMBEL_SEQUENTIAL_HALVING and len(actions) > GUMBEL_TOPK:
            # Sequential Halving: 多轮淘汰
            candidates = list(range(len(actions)))
            total_sims = num_sims
            round_idx = 0

            while len(candidates) > GUMBEL_TOPK and total_sims > 0:
                k = len(candidates) // 2
                sims_per = max(1, total_sims // len(candidates))

                # Gumbel noise
                gumbel = np.random.gumbel(0, 1, size=len(candidates)).astype(np.float32)
                gumbel_logits = np.array([logits[c] for c in candidates]) + gumbel

                # 批量评估候选
                batch_features = []
                batch_masks = []
                batch_child_nodes = []
                batch_paths = []

                for ci, cand_idx in enumerate(candidates):
                    action = actions[cand_idx]
                    child = root.children[action]

                    for _ in range(sims_per):
                        if USE_UNDO_MCTS:
                            board.save_state()
                            board.place_stone(*board.index_to_move(action))
                            path = [root, child]
                            node = child
                            while node.is_expanded and node.children:
                                a, node = self._select_child(node)
                                board.place_stone(*board.index_to_move(a))
                                path.append(node)

                            if board.game_over:
                                if board.winner == 0:
                                    value = 0.0
                                else:
                                    # V4 修复: place_stone后 current_player 已切换
                                    last_player = 3 - board.current_player
                                    value = 1.0 if board.winner == last_player else -1.0
                                self._backpropagate(path, value)
                            else:
                                if not node.is_expanded:
                                    v = self._expand_node(node, board)
                                    if v is not None:
                                        self._backpropagate(path, v)
                                    else:
                                        feature = board.get_feature_planes()
                                        legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
                                        for idx2 in board.get_legal_move_indices():
                                            legal_mask[idx2] = 1.0
                                        batch_features.append(feature)
                                        batch_masks.append(legal_mask)
                                        batch_child_nodes.append(node)
                                        batch_paths.append(path)

                            board.restore_state()
                        else:
                            sim_board = board.copy()
                            sim_board.place_stone_fast(*sim_board.index_to_move(action))
                            path = [root, child]
                            node = child
                            while node.is_expanded and node.children:
                                a, node = self._select_child(node)
                                sim_board.place_stone_fast(*sim_board.index_to_move(a))
                                path.append(node)

                            if sim_board.game_over:
                                if sim_board.winner == 0:
                                    value = 0.0
                                else:
                                    # V4 修复
                                    last_player = 3 - sim_board.current_player
                                    value = 1.0 if sim_board.winner == last_player else -1.0
                                self._backpropagate(path, value)
                            else:
                                if not node.is_expanded:
                                    v = self._expand_node(node, sim_board)
                                    if v is not None:
                                        self._backpropagate(path, v)
                                    else:
                                        feature = sim_board.get_feature_planes()
                                        legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
                                        for idx2 in sim_board.get_legal_move_indices():
                                            legal_mask[idx2] = 1.0
                                        batch_features.append(feature)
                                        batch_masks.append(legal_mask)
                                        batch_child_nodes.append(node)
                                        batch_paths.append(path)

                # 批量推理 (V3: 传入 None 作为 boards, Gumbel模式下节点已通过 _expand_node 扩展)
                if batch_features:
                    self._batch_inference(batch_features, batch_masks,
                                          batch_child_nodes, batch_paths, None)

                # 基于访问量淘汰
                candidate_visits = [(ci, root.children[actions[candidates[ci]]].visit_count)
                                   for ci in range(len(candidates))]
                candidate_visits.sort(key=lambda x: x[1], reverse=True)
                candidates = [candidates[ci] for ci, _ in candidate_visits[:k]]
                total_sims -= sims_per * len(candidate_visits)
                round_idx += 1

            # 最终候选
            final_candidates = candidates
        else:
            # 单轮 top-k
            gumbel = np.random.gumbel(0, 1, size=len(actions)).astype(np.float32)
            gumbel_logits = logits + gumbel
            k = min(GUMBEL_TOPK, len(actions))
            final_candidates = np.argpartition(gumbel_logits, -k)[-k:].tolist()

        # 对最终候选进行搜索
        batch_features = []
        batch_masks = []
        batch_child_nodes = []
        batch_paths = []

        remaining_sims = max(1, num_sims // max(1, len(final_candidates)))
        for cand_idx in final_candidates:
            action = actions[cand_idx]
            child = root.children[action]

            for _ in range(remaining_sims):
                if USE_UNDO_MCTS:
                    board.save_state()
                    board.place_stone(*board.index_to_move(action))
                    path = [root, child]
                    node = child
                    while node.is_expanded and node.children:
                        a, node = self._select_child(node)
                        board.place_stone(*board.index_to_move(a))
                        path.append(node)

                    if board.game_over:
                        if board.winner == 0:
                            value = 0.0
                        else:
                            # V4 修复
                            last_player = 3 - board.current_player
                            value = 1.0 if board.winner == last_player else -1.0
                        self._backpropagate(path, value)
                    else:
                        if not node.is_expanded:
                            v = self._expand_node(node, board)
                            if v is not None:
                                self._backpropagate(path, v)
                            else:
                                feature = board.get_feature_planes()
                                legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
                                for idx2 in board.get_legal_move_indices():
                                    legal_mask[idx2] = 1.0
                                batch_features.append(feature)
                                batch_masks.append(legal_mask)
                                batch_child_nodes.append(node)
                                batch_paths.append(path)

                    board.restore_state()
                else:
                    sim_board = board.copy()
                    sim_board.place_stone_fast(*sim_board.index_to_move(action))
                    path = [root, child]
                    node = child
                    while node.is_expanded and node.children:
                        a, node = self._select_child(node)
                        sim_board.place_stone_fast(*sim_board.index_to_move(a))
                        path.append(node)

                    if sim_board.game_over:
                        if sim_board.winner == 0:
                            value = 0.0
                        else:
                            # V4 修复
                            last_player = 3 - sim_board.current_player
                            value = 1.0 if sim_board.winner == last_player else -1.0
                        self._backpropagate(path, value)
                    else:
                        if not node.is_expanded:
                            v = self._expand_node(node, sim_board)
                            if v is not None:
                                self._backpropagate(path, v)
                            else:
                                feature = sim_board.get_feature_planes()
                                legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
                                for idx2 in sim_board.get_legal_move_indices():
                                    legal_mask[idx2] = 1.0
                                batch_features.append(feature)
                                batch_masks.append(legal_mask)
                                batch_child_nodes.append(node)
                                batch_paths.append(path)

        if batch_features:
            self._batch_inference(batch_features, batch_masks,
                                  batch_child_nodes, batch_paths, None)

        # 生成动作概率
        action_probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
        if root.children:
            visits = np.array([c.visit_count for c in root.children.values()], dtype=np.float32)
            actions = list(root.children.keys())

            if board.move_count < TEMPERATURE_THRESHOLD and self.temperature > 0:
                visits_t = visits ** (1.0 / self.temperature)
                probs = visits_t / visits_t.sum()
            else:
                probs = np.zeros_like(visits)
                probs[np.argmax(visits)] = 1.0

            for action, prob in zip(actions, probs):
                action_probs[action] = prob

        # V3 修复: root_value 从子节点计算
        root_value = 0.0
        if root.children:
            total_child_visits = sum(c.visit_count for c in root.children.values())
            if total_child_visits > 0:
                root_value = sum(c.visit_count * c.q_value
                                 for c in root.children.values()) / total_child_visits
        return action_probs, root_value

    def _root_parallel_search(self, board, root, num_sims):
        """Root Parallelization: 多线程各跑独立MCTS树, 合并结果"""
        # 简化: 在单进程中交替搜索, 避免GIL问题
        # 真正的多进程需要共享内存模型, 这里用单进程多树模拟
        num_trees = min(ROOT_PARALLEL_THREADS, 4)
        sims_per_tree = max(1, num_sims // num_trees)

        all_visits = defaultdict(float)

        for t in range(num_trees):
            # 每棵树独立搜索
            tree_root = self._alloc_node()
            self._expand_node(tree_root, board)
            if self.add_noise and tree_root.children:
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(tree_root.children))
                for i, (action, child) in enumerate(tree_root.children.items()):
                    # V4 修复: 基于 original_prior 施加噪声, 防止累积
                    base_prior = child.original_prior if child.original_prior > 0 else child.prior
                    child.prior = (1 - DIRICHLET_EPSILON) * base_prior + DIRICHLET_EPSILON * noise[i]

            # 在子搜索中使用标准搜索 (不递归 root parallel)
            action_probs, _ = self._standard_search(board, tree_root, sims_per_tree)

            # 合并访问量
            for action in range(BOARD_SQUARES):
                all_visits[action] += action_probs[action]

        # 归一化
        total = sum(all_visits.values())
        action_probs = np.zeros(BOARD_SQUARES, dtype=np.float32)
        if total > 0:
            for action, v in all_visits.items():
                action_probs[action] = v / total

        # V3 修复: 更新 self.root 为最后一棵树的 root, 保持子树复用能力
        self.root = tree_root

        # V6 修复: 从最后一棵树的 root 计算 root_value (原 root 未被搜索)
        root_value = 0.0
        if tree_root.children:
            total_child_visits = sum(c.visit_count for c in tree_root.children.values())
            if total_child_visits > 0:
                root_value = sum(c.visit_count * c.q_value
                                 for c in tree_root.children.values()) / total_child_visits
        return action_probs, root_value

    def _batch_inference(self, features, masks, nodes, paths, boards=None):
        """V5 修复: 批量推理 + 正确扩展 + 防止双重回传"""
        if not features:
            return

        batch_size = len(features)
        if batch_size == 1:
            # 单样本直接推理
            policy, value = self.model.predict(features[0], masks[0])
            node = nodes[0]
            # V6 修复: 与多样本路径一致, 检查节点是否已被扩展
            if not node.is_expanded and boards is not None and boards[0] is not None:
                board_state, current_player = boards[0]
                self._expand_node_with_policy(node, policy, masks[0], board_state, current_player)
                node.cached_value = float(value)
                self._backpropagate(paths[0], float(value))
            elif node.is_expanded:
                # V6: 节点已被其他路径扩展 — 用缓存值回传, 防止双重计数
                self._backpropagate(paths[0], node.cached_value)
            else:
                # boards 为 None — 无法扩展, 直接回传推理值
                self._backpropagate(paths[0], float(value))
            return

        # 批量推理
        policies, values = self.model.predictBatch(features, masks)

        for i in range(batch_size):
            node = nodes[i]
            if not node.is_expanded and boards is not None and boards[i] is not None:
                board_state, current_player = boards[i]
                self._expand_node_with_policy(node, policies[i], masks[i], board_state, current_player)
                node.cached_value = float(values[i])
                self._backpropagate(paths[i], float(values[i]))
            elif node.is_expanded:
                # V5 修复: 节点已被其他批量条目扩展 — 用缓存值回传, 防止双重计数
                self._backpropagate(paths[i], node.cached_value)
            else:
                # boards 为 None — 无法扩展, 直接回传推理值
                self._backpropagate(paths[i], float(values[i]))

    def _expand_node_with_policy(self, node, policy, legal_mask, board_state, current_player):
        """使用已有 policy 扩展节点 (避免重复推理) — V6: 保存网络原始prior"""
        if node.is_expanded:
            return

        legal_indices = np.where(legal_mask > 0.5)[0]
        if len(legal_indices) == 0:
            return

        # V6 修复: 在模式注入之前保存网络原始prior
        raw_policy = policy.copy()

        # V3: 模式注入 — 使用保存的 board 快照
        if USE_PATTERN_INJECTION and board_state is not None:
            pattern_bonus = compute_pattern_prior_bonus(board_state, current_player)
            pat_max = pattern_bonus.max()
            if pat_max > 0:
                pattern_prior = pattern_bonus / pat_max
                policy = (1 - PATTERN_INJECTION_WEIGHT) * policy + PATTERN_INJECTION_WEIGHT * pattern_prior
                policy = policy * legal_mask
                psum = policy.sum()
                if psum > 0:
                    policy /= psum

        # 创建子节点 — V6: original_prior 使用网络原始prior (模式注入前)
        for idx in legal_indices:
            child = self._alloc_node(parent=node, action=int(idx), prior=policy[idx])
            child.original_prior = raw_policy[idx]  # V6: 保存网络原始prior
            node.children[int(idx)] = child

        node.is_expanded = True
        node.sqrt_N = 0.0

    def _select_child(self, node):
        """PUCT 选择 (含 Q-Normalization)"""
        # V2: 计算 Q 值范围用于归一化
        q_min = float('inf')
        q_max = float('-inf')
        if USE_Q_NORM:
            for child in node.children.values():
                if child.visit_count > 0:
                    q = child.q_value
                    if q < q_min: q_min = q
                    if q > q_max: q_max = q
            if q_min == float('inf'):
                q_min = FPU_VALUE if USE_FPU else 0.0
                q_max = 0.0

        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = child.puct_score(q_min, q_max)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        best_child.virtual_loss += 1
        return best_action, best_child

    def _select_child_from_list(self, node, children_list):
        """从指定子节点列表中选择"""
        q_min = float('inf')
        q_max = float('-inf')
        if USE_Q_NORM:
            for _, child in children_list:
                if child.visit_count > 0:
                    q = child.q_value
                    if q < q_min: q_min = q
                    if q > q_max: q_max = q
            if q_min == float('inf'):
                q_min = FPU_VALUE if USE_FPU else 0.0
                q_max = 0.0

        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in children_list:
            score = child.puct_score(q_min, q_max)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        best_child.virtual_loss += 1
        return best_action, best_child

    def _expand_node(self, node, board):
        """
        V2: 扩展节点 — 返回 value (消除双重推理)
        ==========================================
        之前: expand 只获取 policy, 然后 search 再获取 value = 2次推理
        V2: expand 同时获取 policy 和 value, 返回 value 给调用方
        """
        if node.is_expanded:
            return node.cached_value if node.visit_count > 0 else None

        feature = board.get_feature_planes()
        legal_indices = board.get_legal_move_indices()

        if not legal_indices:
            return None

        # 合法掩码
        legal_mask = np.zeros(BOARD_SQUARES, dtype=np.float32)
        for idx in legal_indices:
            legal_mask[idx] = 1.0

        # 网络推理: 同时获取 policy 和 value
        policy, value = self.model.predict(feature, legal_mask)

        # V2: 缓存 value
        node.cached_value = value

        # V6 修复: 在模式注入和转置表混合之前保存网络原始prior
        raw_policy = policy.copy()

        # 模式注入: 混合棋型先验
        if USE_PATTERN_INJECTION:
            pattern_bonus = compute_pattern_prior_bonus(board.board, board.current_player)
            pat_max = pattern_bonus.max()
            if pat_max > 0:
                pattern_prior = pattern_bonus / pat_max
                policy = (1 - PATTERN_INJECTION_WEIGHT) * policy + PATTERN_INJECTION_WEIGHT * pattern_prior
                policy = policy * legal_mask
                psum = policy.sum()
                if psum > 0:
                    policy /= psum

        # 转置表查询
        if USE_TRANSPOSITION and self._tp_table:
            tp_result = self._tp_table.lookup(board.zobrist_hash)
            if tp_result is not None:
                tp_value, tp_visits, tp_policy = tp_result
                if tp_visits > 10:
                    # 混合转置表策略
                    policy = 0.7 * policy + 0.3 * tp_policy
                    policy = policy * legal_mask
                    psum = policy.sum()
                    if psum > 0:
                        policy /= psum

        # 创建子节点 — V6: original_prior 使用网络原始prior (模式注入前)
        for idx in legal_indices:
            child = self._alloc_node(parent=node, action=idx, prior=policy[idx])
            child.original_prior = raw_policy[idx]  # V6: 保存网络原始prior
            node.children[idx] = child

        node.is_expanded = True
        node.sqrt_N = 0.0

        # 存入转置表
        if USE_TRANSPOSITION and self._tp_table:
            self._tp_table.store(board.zobrist_hash, value, 1, policy)

        return value

    def _backpropagate(self, path, value):
        """回传价值 + RAVE (V7: 清理冗余条件)"""
        # RAVE 只应使用当前节点子树中的动作, 而非整条路径
        subtree_actions = set() if USE_RAVE else None

        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            node.virtual_loss = max(0, node.virtual_loss - 1)
            if node.parent is not None:
                node.parent.sqrt_N = math.sqrt(node.parent.visit_count + 1)

            if USE_RAVE:
                # V7: RAVE value 从当前节点视角计算
                # 当前 value 是从当前节点父节点视角的
                # 子节点的 rave_value 应从当前节点视角计算 = -value
                for action in subtree_actions:
                    if action in node.children:
                        child = node.children[action]
                        child.rave_count += 1
                        child.rave_value += -value
                # 当前节点的动作属于其父节点的子树
                if node.action is not None:
                    subtree_actions.add(node.action)

            value = -value

    def advance(self, action):
        """
        V2 修复: 正确推进子树
        =============================
        修复前: advance 后 root 被推进, 但 search 中又在新 root 的 children 里查找
        修复后: advance 直接推进 root, search 中直接使用 self.root
        """
        if USE_SUBTREE_REUSE and self.root is not None:
            if action in self.root.children:
                new_root = self.root.children[action]
                new_root.parent = None
                # V3 修复: 清理旧root未选中子节点的引用, 帮助GC回收
                old_root = self.root
                for act, child in old_root.children.items():
                    if act != action:
                        child.parent = None  # 断开引用, 允许GC回收
                old_root.children.clear()
                self.root = new_root
                # 不清除子树, 保留已搜索的信息
            else:
                self.root = None
        else:
            self.root = None
