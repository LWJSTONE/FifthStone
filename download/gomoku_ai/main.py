#!/usr/bin/env python3
"""
五子棋 AI 训练系统 — V2 全面修复版
===================================
用法:
  python main.py train                  # 开始训练
  python main.py train --resume PATH    # 恢复训练
  python main.py play                   # 人机对弈
  python main.py play --model PATH      # 指定模型
  python main.py eval --model PATH      # 评估
  python main.py bench                  # 性能基准
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_train(args):
    from train import Trainer
    trainer = Trainer(device='cpu', resume_path=args.resume)
    trainer.train()


def run_play(args):
    from network import create_model
    from train import play_vs_model

    model = create_model(device='cpu')
    if args.model and os.path.exists(args.model):
        ckpt = __import__('torch').load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"模型已加载: {args.model}")
    elif os.path.exists('checkpoints/best_model.pt'):
        ckpt = __import__('torch').load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print("模型已加载: best_model.pt")
    else:
        print("警告: 无训练模型, 使用随机初始化")

    play_vs_model(model)


def run_eval(args):
    from network import create_model
    from mcts import MCTS
    from board import Board, BLACK, WHITE
    from config import NUM_SIMULATIONS, MAX_MOVES

    model = create_model(device='cpu')
    if args.model and os.path.exists(args.model):
        ckpt = __import__('torch').load(args.model, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("错误: 请指定模型 --model PATH")
        return

    print(f"\n评估: {args.model}")
    wins = draws = losses = 0
    total_moves = total_time = 0

    for g in range(args.games):
        board = Board()
        mcts = MCTS(model, num_simulations=NUM_SIMULATIONS, add_noise=False, temperature=0.0)
        start = time.time()
        while not board.game_over and board.move_count < MAX_MOVES:
            probs, val = mcts.search(board)
            action = np.argmax(probs)
            board.place_stone(*board.index_to_move(action))
            mcts.advance(action)
        t = time.time() - start
        total_moves += board.move_count
        total_time += t

        if board.winner == BLACK:
            wins += 1
        elif board.winner == WHITE:
            losses += 1
        else:
            draws += 1
        print(f"  对局{g+1}: {'黑胜' if board.winner==1 else '白胜' if board.winner==2 else '平局'} "
              f"({board.move_count}步, {t:.1f}s)")

    print(f"\n结果: {wins}胜 {draws}平 {losses}负")
    print(f"平均: {total_moves/args.games:.1f}步, {total_time/args.games:.1f}s/局")


def run_bench(args):
    print("\n" + "=" * 60)
    print("  五子棋 AI 性能基准测试 (V2)")
    print("=" * 60)

    from network import create_model, create_inference_model
    from board import Board, BLACK, WHITE
    from mcts import MCTS
    from vct import find_must_move, vct_search, compute_pattern_prior_bonus
    from config import NUM_SIMULATIONS, INPUT_CHANNELS, BOARD_SIZE, BOARD_SQUARES

    import torch

    # 1. 网络推理
    print("\n[1] 网络推理速度")
    model = create_model(device='cpu')
    model.eval()
    x = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    n = 200
    start = time.time()
    for _ in range(n):
        with torch.no_grad():
            _ = model(x)
    t = time.time() - start
    print(f"  原始推理: {t/n*1000:.2f} ms/次, {n/t:.0f} 次/秒")

    # 推理优化模型
    try:
        opt_model = create_inference_model(model)
        if opt_model is not model:
            for _ in range(5):
                with torch.no_grad():
                    try:
                        _ = opt_model(x)
                    except Exception:
                        break
            start = time.time()
            for _ in range(n):
                with torch.no_grad():
                    try:
                        _ = opt_model(x)
                    except Exception:
                        break
            t = time.time() - start
            print(f"  优化推理: {t/n*1000:.2f} ms/次")
    except Exception as e:
        print(f"  优化推理: 跳过 ({e})")

    # 2. 必走着法检测
    print("\n[2] 必走着法检测速度")
    board = Board()
    board.place_stone(7, 7)
    board.place_stone(7, 8)
    n2 = 1000
    start = time.time()
    for _ in range(n2):
        find_must_move(board.board, board.current_player)
    t = time.time() - start
    print(f"  {t/n2*1e6:.1f} μs/次")

    # 3. 模式注入
    print("\n[3] 模式先验计算速度")
    start = time.time()
    for _ in range(n2):
        compute_pattern_prior_bonus(board.board, board.current_player)
    t = time.time() - start
    print(f"  {t/n2*1000:.2f} ms/次")

    # 4. MCTS (标准)
    print("\n[4] MCTS搜索速度")
    mcts = MCTS(model, num_simulations=100, add_noise=True)
    board = Board()
    board.place_stone(7, 7)
    start = time.time()
    probs, val = mcts.search(board)
    t = time.time() - start
    print(f"  100次模拟: {t:.2f}s, 每次模拟: {t/100*1000:.2f}ms")

    # 5. VCT
    print("\n[5] VCT搜索速度")
    start = time.time()
    result = vct_search(board.board, BLACK, 12)
    t = time.time() - start
    print(f"  VCT: {t:.3f}s, 结果: {'找到' if result >= 0 else '未找到'}")

    # 6. 特征平面计算
    print("\n[6] 特征平面计算速度 (Numba JIT)")
    start = time.time()
    for _ in range(100):
        board.get_feature_planes()
    t = time.time() - start
    print(f"  {t/100*1000:.2f} ms/次")

    # 7. 自我对弈
    print("\n[7] 自我对弈速度 (30次模拟)")
    from self_play import self_play_game
    start = time.time()
    game = self_play_game(model, num_simulations=30)
    t = time.time() - start
    print(f"  单局: {t:.1f}s, {game.length}步, {t/game.length:.2f}s/步")

    # 8. Board Undo vs Copy
    print("\n[8] Board Undo vs Copy 速度")
    board = Board()
    board.place_stone(7, 7)
    board.place_stone(7, 8)
    board.place_stone(6, 6)
    # Copy
    start = time.time()
    for _ in range(10000):
        b2 = board.copy()
    t_copy = time.time() - start
    # Undo
    start = time.time()
    for _ in range(10000):
        board.save_state()
        board.place_stone(8, 8)
        board.restore_state()
    t_undo = time.time() - start
    print(f"  Board.copy(): {t_copy/10000*1e6:.1f} μs/次")
    print(f"  save/restore: {t_undo/10000*1e6:.1f} μs/次")
    print(f"  加速比: {t_copy/t_undo:.2f}×")

    print("\n" + "=" * 60)
    print("  基准测试完成")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='五子棋 AI (V2 全面修复版)')
    sub = parser.add_subparsers(dest='command')

    tp = sub.add_parser('train')
    tp.add_argument('--resume', type=str, default=None)

    pp = sub.add_parser('play')
    pp.add_argument('--model', type=str, default=None)

    ep = sub.add_parser('eval')
    ep.add_argument('--model', type=str, required=True)
    ep.add_argument('--games', type=int, default=20)

    sub.add_parser('bench')

    args = parser.parse_args()
    if args.command == 'train': run_train(args)
    elif args.command == 'play': run_play(args)
    elif args.command == 'eval': run_eval(args)
    elif args.command == 'bench': run_bench(args)
    else: parser.print_help()


if __name__ == '__main__':
    main()
