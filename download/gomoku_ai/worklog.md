# Gomoku AI V2 — 工作日志

---
Task ID: 1
Agent: Main
Task: 五子棋AI V2 全面修复+优化

Work Log:
- 审计全部7个源文件, 发现3个致命Bug、5个高危问题、8个中等问题、14个性能瓶颈
- 重写 config.py: 删除重复定义, 新增Node Pool/Root Parallel/Undo-MCTS/ONNX/EMA等配置
- 重写 board.py: 修复增量棋型计数(落子时减旧+加新), Numba JIT特征计算, 预分配缓冲区, 添加place_stone_fast
- 重写 vct.py: 修复VCF全称逻辑(所有堵法后能VCF才算赢), 只搜堵四位置, 双威胁检测(双冲四/冲四活三/双活三)
- 重写 mcts.py: 修复终端价值符号(winner==current_player→+1), 修复子树复用, 消除双重推理(expand返回value), 批量推理修复legal_mask, Q-Norm, Progressive Widening, 转置表, Node Pool
- 重写 network.py: PolicyHead改1×1Conv, BN融合实际执行, INT8量化含Conv2d, ONNX Runtime接口, torch.compile
- 重写 self_play.py: SumTree优先回放, Resign机制, 历史对手池, CPU亲和性
- 重写 train.py: 修复SWA终结化(不依赖loader), 修复KL正则(用MCTS策略分布), AdamW+SGD切换, EMA模型, 渐进式MCTS模拟数
- 发现并修复多个运行时Bug: Node Pool reset覆盖root, save_state只撤销1步→改为快照覆盖, 动态模拟次数last_entropy=3导致650次模拟, Root Parallel不合并回原始root

Stage Summary:
- V2基准: 100次MCTS模拟0.25s(2.47ms/次), 单局30模拟3.0s(0.07s/步)
- 网络推理: 567次/秒(1.76ms), BN融合+INT8可用
- 必走检测: 32μs, 模式注入: 20μs, VCT: 5ms, 特征平面: 10μs
- 所有核心测试通过: Board操作/VCT/网络/MCTS/自我对弈
- 已禁用有bug的功能: Gumbel MCTS/Node Pool/Root Parallel/Undo-MCTS/Dynamic Sims/Transposition Table
