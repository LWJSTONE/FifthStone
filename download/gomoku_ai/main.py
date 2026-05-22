#!/usr/bin/env python3
"""
五子棋 AI 训练系统 — CPU 极致优化版
===================================
入口脚本: 支持训练、评估、人机对弈三种模式

用法:
  python main.py train                  # 开始训练
  python main.py train --resume PATH    # 恢复训练
  python main.py play                   # 人机对弈(默认模型)
  python main.py play --model PATH      # 指定模型对弈
  python main.py eval --model PATH      # 评估模型棋力
  python main.py bench                  # 性能基准测试
"""

import sys
import os
import time
import argparse
import numpy as np

# 将项目目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_train(args):
    """训练模式"""
    from train import Trainer

    trainer = Trainer(device='cpu', resume_path=args.resume)
    trainer.train()


def run_play(args):
    """人机对弈模式"""
    from network import create_model
    from train import play_vs_model

    model = create_model(device='cpu')

    if args.model and os.path.exists(args.model):
        checkpoint = __import__('torch').load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已加载: {args.model}")
    elif os.path.exists('checkpoints/best_model.pt'):
        checkpoint = __import__('torch').load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("模型已加载: checkpoints/best_model.pt")
    else:
        print("警告: 未找到训练好的模型，使用随机初始化模型")

    play_vs_model(model)


def run_eval(args):
    """评估模式"""
    from network import create_model
    from mcts import MCTS
    from board import Board, BLACK, WHITE
    from config import NUM_SIMULATIONS, MAX_MOVES

    model = create_model(device='cpu')

    if args.model and os.path.exists(args.model):
        checkpoint = __import__('torch').load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("错误: 请指定模型路径 --model PATH")
        return

    print(f"\n模型评估: {args.model}")
    print(f"对局数: {args.games}")

    wins, draws, losses = 0, 0, 0
    total_moves = 0
    total_time = 0

    for g in range(args.games):
        board = Board()
        mcts = MCTS(model, num_simulations=NUM_SIMULATIONS, add_noise=False, temperature=0.0)
        start = time.time()

        while not board.game_over and board.move_count < MAX_MOVES:
            action_probs, value = mcts.search(board)
            action = np.argmax(action_probs)
            r, c = board.index_to_move(action)
            board.place_stone(r, c)

        game_time = time.time() - start
        total_moves += board.move_count
        total_time += game_time

        if board.winner == BLACK:
            wins += 1
        elif board.winner == WHITE:
            losses += 1
        else:
            draws += 1

        print(f"  对局 {g+1}: {'黑胜' if board.winner == 1 else '白胜' if board.winner == 2 else '平局'} "
              f"({board.move_count}步, {game_time:.1f}s)")

    print(f"\n结果: {wins}胜 {draws}平 {losses}负")
    print(f"平均步数: {total_moves/args.games:.1f}")
    print(f"平均用时: {total_time/args.games:.1f}s/局")


def run_bench(args):
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("  五子棋 AI 性能基准测试")
    print("=" * 60)

    from network import create_model
    from board import Board
    from mcts import MCTS
    from config import NUM_SIMULATIONS, INPUT_CHANNELS, BOARD_SIZE

    # ===== 测试1: 神经网络推理速度 =====
    print("\n[1] 神经网络推理速度")
    model = create_model(device='cpu')
    model.eval()

    import torch
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # 测量
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(x)
    elapsed = time.time() - start
    print(f"  单次推理: {elapsed/n_iters*1000:.2f} ms")
    print(f"  吞吐量: {n_iters/elapsed:.0f} 次/秒")

    # 批量推理
    batch_sizes = [1, 4, 8, 16, 32]
    print(f"\n  批量推理测试:")
    for bs in batch_sizes:
        xb = torch.randn(bs, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        start = time.time()
        for _ in range(n_iters // bs):
            with torch.no_grad():
                _ = model(xb)
        elapsed = time.time() - start
        total_inferences = (n_iters // bs) * bs
        print(f"    batch={bs:2d}: {elapsed/max(1,total_inferences)*1000:.2f} ms/样本, "
              f"{total_inferences/elapsed:.0f} 样本/秒")

    # ===== 测试2: MCTS搜索速度 =====
    print("\n[2] MCTS 搜索速度")
    board = Board()
    mcts = MCTS(model, num_simulations=NUM_SIMULATIONS)

    start = time.time()
    action_probs, value = mcts.search(board)
    elapsed = time.time() - start

    print(f"  {NUM_SIMULATIONS} 次模拟: {elapsed:.2f}s")
    print(f"  每次模拟: {elapsed/NUM_SIMULATIONS*1000:.2f} ms")
    print(f"  根节点价值: {value:.3f}")

    # ===== 测试3: 自我对弈速度 =====
    print("\n[3] 自我对弈速度")
    from self_play import self_play_game

    start = time.time()
    game = self_play_game(model, num_simulations=min(100, NUM_SIMULATIONS))
    elapsed = time.time() - start

    print(f"  单局用时: {elapsed:.1f}s")
    print(f"  对局步数: {game.length}")
    print(f"  每步用时: {elapsed/max(1,game.length):.2f}s")
    print(f"  胜者: {'黑' if game.winner == 1 else '白' if game.winner == 2 else '平局'}")

    # ===== 测试4: 棋盘操作速度 =====
    print("\n[4] 棋盘操作速度 (Numba JIT)")
    board = Board()

    # 预热Numba
    board.place_stone(7, 7)
    board.undo_stone()

    n_ops = 10000
    start = time.time()
    for _ in range(n_ops):
        board.place_stone(7, 7)
        board.undo_stone()
    elapsed = time.time() - start
    print(f"  落子+撤销: {elapsed/n_ops*1e6:.1f} μs/次")

    # 特征平面生成
    board.place_stone(7, 7)
    board.place_stone(8, 8)
    start = time.time()
    for _ in range(n_ops):
        _ = board.get_feature_planes()
    elapsed = time.time() - start
    print(f"  特征生成: {elapsed/n_ops*1e6:.1f} μs/次")

    print("\n" + "=" * 60)
    print("  基准测试完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='五子棋 AI 训练系统')
    subparsers = parser.add_subparsers(dest='command', help='运行模式')

    # train
    train_parser = subparsers.add_parser('train', help='开始训练')
    train_parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')

    # play
    play_parser = subparsers.add_parser('play', help='人机对弈')
    play_parser.add_argument('--model', type=str, default=None, help='模型路径')

    # eval
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--model', type=str, required=True, help='模型路径')
    eval_parser.add_argument('--games', type=int, default=20, help='评估对局数')

    # bench
    bench_parser = subparsers.add_parser('bench', help='性能基准测试')

    args = parser.parse_args()

    if args.command == 'train':
        run_train(args)
    elif args.command == 'play':
        run_play(args)
    elif args.command == 'eval':
        run_eval(args)
    elif args.command == 'bench':
        run_bench(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
