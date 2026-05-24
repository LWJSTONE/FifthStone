'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Separator } from '@/components/ui/separator'
import GomokuBoard from './gomoku-board'
import {
  Swords, Play, Square, Circle, CircleDot, Clock, Loader2, Trophy, Gauge,
} from 'lucide-react'

interface ModelInfo {
  name: string
  path: string
  size: number
  modified_time: number
}

interface BattleMove {
  player: number
  row: number
  col: number
  value: number
  move_number: number
}

const EMPTY_BOARD = Array.from({ length: 15 }, () => Array(15).fill(0))

export default function AIVsAI() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model1, setModel1] = useState<string>('')
  const [model2, setModel2] = useState<string>('')
  const [numSims, setNumSims] = useState(200)
  const [speed, setSpeed] = useState(500) // ms delay between moves
  const [board, setBoard] = useState<number[][]>(EMPTY_BOARD)
  const [lastMove, setLastMove] = useState<[number, number] | null>(null)
  const [moveHistory, setMoveHistory] = useState<BattleMove[]>([])
  const [battleRunning, setBattleRunning] = useState(false)
  const [currentEval, setCurrentEval] = useState<number | null>(null)
  const [timePerMove, setTimePerMove] = useState<number | null>(null)
  const [gameResult, setGameResult] = useState<string>('')
  const [totalMoves, setTotalMoves] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)
  const moveTimerRef = useRef<number | null>(null)

  // Fetch models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/models?XTransformPort=8000')
        if (res.ok) {
          const data = await res.json()
          setModels(data)
          if (data.length > 0) {
            if (!model1) setModel1(data[0].path)
            if (!model2) setModel2(data[Math.min(1, data.length - 1)].path)
          }
        }
      } catch { /* silent */ }
    }
    fetchModels()
  }, [model1, model2])

  // Cleanup
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close()
      if (moveTimerRef.current) clearTimeout(moveTimerRef.current)
    }
  }, [])

  const startBattle = useCallback(async () => {
    try {
      setBoard(EMPTY_BOARD.map(row => [...row]))
      setLastMove(null)
      setMoveHistory([])
      setCurrentEval(null)
      setTimePerMove(null)
      setGameResult('')
      setTotalMoves(0)
      setBattleRunning(true)

      if (wsRef.current) wsRef.current.close()

      const ws = new WebSocket(`ws://${window.location.host}/ws/battle?XTransformPort=8000`)
      wsRef.current = ws

      ws.onopen = () => {
        ws.send(JSON.stringify({
          type: 'start',
          model1_path: model1 || undefined,
          model2_path: model2 || undefined,
          num_simulations: numSims,
        }))
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.type === 'move') {
          const move: BattleMove = {
            player: data.player,
            row: data.row,
            col: data.col,
            value: data.value,
            move_number: data.move_number,
          }

          // Delay for visual effect
          const delay = speed
          const startTime = Date.now()

          setTimeout(() => {
            setBoard(prev => {
              const newBoard = prev.map(row => [...row])
              newBoard[data.row][data.col] = data.player
              return newBoard
            })
            setLastMove([data.row, data.col])
            setMoveHistory(prev => [...prev, move])
            setCurrentEval(data.value)
            setTotalMoves(data.move_number)
            const elapsed = Date.now() - startTime
            setTimePerMove(elapsed > 0 ? elapsed / 1000 : null)
          }, delay > 50 ? 50 : 0)
        } else if (data.type === 'game_over') {
          setBattleRunning(false)
          setTotalMoves(data.total_moves)
          if (data.winner === 0) {
            setGameResult('平局！')
          } else if (data.winner === 1) {
            setGameResult('⬛ 黑棋 (模型1) 获胜！')
          } else {
            setGameResult('⬜ 白棋 (模型2) 获胜！')
          }
        } else if (data.type === 'error') {
          setBattleRunning(false)
          setGameResult(`错误: ${data.message}`)
        }
      }

      ws.onclose = () => {
        setBattleRunning(false)
      }

      ws.onerror = () => {
        setBattleRunning(false)
        setGameResult('连接失败，请检查后端服务')
      }
    } catch {
      setBattleRunning(false)
      setGameResult('连接失败')
    }
  }, [model1, model2, numSims, speed])

  const stopBattle = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }))
      wsRef.current.close()
    }
    setBattleRunning(false)
  }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Board */}
      <div className="lg:col-span-2 flex flex-col gap-3">
        <GomokuBoard
          board={board}
          lastMove={lastMove}
          interactive={false}
        />
        {gameResult && (
          <div className="text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30">
              <Trophy className="h-4 w-4 text-amber-500" />
              <span className="text-sm font-medium">{gameResult}</span>
            </div>
          </div>
        )}
      </div>

      {/* Side Panel */}
      <div className="space-y-3">
        {/* Battle Controls */}
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Swords className="h-3.5 w-3.5" /> 对战设置
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">黑棋模型 (模型1)</label>
              <Select value={model1} onValueChange={setModel1} disabled={battleRunning}>
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue placeholder="选择模型" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(m => (
                    <SelectItem key={m.path} value={m.path}>{m.name}</SelectItem>
                  ))}
                  {models.length === 0 && <SelectItem value="default">默认模型</SelectItem>}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">白棋模型 (模型2)</label>
              <Select value={model2} onValueChange={setModel2} disabled={battleRunning}>
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue placeholder="选择模型" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(m => (
                    <SelectItem key={m.path} value={m.path}>{m.name}</SelectItem>
                  ))}
                  {models.length === 0 && <SelectItem value="default">默认模型</SelectItem>}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">
                MCTS 模拟次数: {numSims}
              </label>
              <Slider
                value={[numSims]}
                onValueChange={v => setNumSims(v[0])}
                min={50}
                max={1000}
                step={50}
                disabled={battleRunning}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">
                播放速度: {speed}ms
              </label>
              <Slider
                value={[speed]}
                onValueChange={v => setSpeed(v[0])}
                min={50}
                max={2000}
                step={50}
                disabled={battleRunning}
              />
            </div>
            <div className="flex gap-2">
              <Button
                onClick={startBattle}
                disabled={battleRunning}
                size="sm"
                className="gap-1 flex-1"
              >
                <Play className="h-3.5 w-3.5" /> 开始对战
              </Button>
              <Button
                onClick={stopBattle}
                disabled={!battleRunning}
                variant="destructive"
                size="sm"
                className="gap-1"
              >
                <Square className="h-3.5 w-3.5" /> 停止
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Live Stats */}
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Gauge className="h-3.5 w-3.5" /> 实时统计
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">状态</span>
              {battleRunning ? (
                <Badge variant="default" className="text-xs gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" /> 进行中
                </Badge>
              ) : (
                <Badge variant="secondary" className="text-xs">空闲</Badge>
              )}
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">当前步数</span>
              <span className="text-sm font-medium">{totalMoves}</span>
            </div>
            {currentEval !== null && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">当前评估</span>
                <span className="text-sm font-medium">{currentEval.toFixed(4)}</span>
              </div>
            )}
            {timePerMove !== null && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">每步用时</span>
                <span className="text-sm font-medium">{timePerMove.toFixed(2)}s</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Move History */}
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">落子记录</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <ScrollArea className="h-48">
              {moveHistory.length === 0 ? (
                <p className="text-xs text-muted-foreground">尚无落子</p>
              ) : (
                <div className="space-y-1">
                  {moveHistory.map((m, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs font-mono">
                      <span className="text-muted-foreground w-6 shrink-0">{m.move_number}.</span>
                      <span className="shrink-0">
                        {m.player === 1 ? (
                          <Circle className="h-2.5 w-2.5" />
                        ) : (
                          <CircleDot className="h-2.5 w-2.5" />
                        )}
                      </span>
                      <span>({m.row},{m.col})</span>
                      <span className="text-muted-foreground ml-auto">v={m.value.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
