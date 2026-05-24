'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import GomokuBoard from './gomoku-board'
import {
  Circle, CircleDot, Undo2, RotateCcw, Loader2, Bot, User, Trophy,
} from 'lucide-react'

interface ModelInfo {
  name: string
  path: string
  size: number
  modified_time: number
}

interface GameState {
  board: number[][]
  current_player: number
  game_over: boolean
  winner: number
  last_move: [number, number] | null
}

interface MoveRecord {
  number: number
  player: number
  row: number
  col: number
  value?: number
  thinkingTime?: number
}

const EMPTY_BOARD = Array.from({ length: 15 }, () => Array(15).fill(0))

export default function HumanVsAI() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [humanColor, setHumanColor] = useState<1 | 2>(1) // 1=black, 2=white
  const [gameState, setGameState] = useState<GameState>({
    board: EMPTY_BOARD,
    current_player: 1,
    game_over: false,
    winner: 0,
    last_move: null,
  })
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([])
  const [aiThinking, setAiThinking] = useState(false)
  const [aiValue, setAiValue] = useState<number | null>(null)
  const [aiThinkingTime, setAiThinkingTime] = useState<number | null>(null)
  const [gameStarted, setGameStarted] = useState(false)
  const [gameResult, setGameResult] = useState<string>('')
  const wsRef = useRef<WebSocket | null>(null)

  // Fetch models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('/api/models?XTransformPort=8000')
        if (res.ok) {
          const data = await res.json()
          setModels(data)
          if (data.length > 0 && !selectedModel) {
            setSelectedModel(data[0].path)
          }
        }
      } catch { /* silent */ }
    }
    fetchModels()
    const interval = setInterval(fetchModels, 10000)
    return () => clearInterval(interval)
  }, [selectedModel])

  // Cleanup WebSocket
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  const connectWebSocket = useCallback((): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      const ws = new WebSocket(`ws://${window.location.host}/ws/play?XTransformPort=8000`)
      wsRef.current = ws

      ws.onopen = () => resolve(ws)

      ws.onerror = () => reject(new Error('连接失败'))

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)

        if (data.type === 'state') {
          setGameState({
            board: data.board,
            current_player: data.current_player,
            game_over: data.game_over,
            winner: data.winner,
            last_move: data.last_move,
          })
        } else if (data.type === 'ai_move') {
          setAiThinking(false)
          setAiValue(data.value)
          setAiThinkingTime(data.thinking_time)
          setMoveHistory(prev => [...prev, {
            number: prev.length + 1,
            player: gameState.current_player,
            row: data.row,
            col: data.col,
            value: data.value,
            thinkingTime: data.thinking_time,
          }])
        } else if (data.type === 'game_over') {
          setAiThinking(false)
          setGameStarted(false)
          const winnerName = data.winner === 1 ? '黑棋' : data.winner === 2 ? '白棋' : '平局'
          const playerColor = humanColor === 1 ? '黑棋' : '白棋'
          if (data.winner === 0) {
            setGameResult('平局！')
          } else if (data.winner === humanColor) {
            setGameResult(`🎉 你 (${playerColor}) 获胜！`)
          } else {
            setGameResult(`AI (${winnerName}) 获胜！`)
          }
        } else if (data.type === 'error') {
          setAiThinking(false)
          console.error('WS Error:', data.message)
        }
      }

      ws.onclose = () => {
        setGameStarted(false)
      }

      setTimeout(() => reject(new Error('连接超时')), 5000)
    })
  }, [humanColor, gameState.current_player])

  const startNewGame = async () => {
    try {
      setMoveHistory([])
      setAiValue(null)
      setAiThinkingTime(null)
      setGameResult('')
      setGameState({
        board: EMPTY_BOARD,
        current_player: 1,
        game_over: false,
        winner: 0,
        last_move: null,
      })

      const ws = await connectWebSocket()
      ws.send(JSON.stringify({
        type: 'start',
        human_color: humanColor,
        model_path: selectedModel || undefined,
      }))
      setGameStarted(true)
    } catch (err) {
      setGameResult('连接失败，请检查后端服务')
    }
  }

  const handleCellClick = useCallback((row: number, col: number) => {
    if (!gameStarted || gameState.game_over || aiThinking) return
    if (gameState.current_player !== humanColor) return
    if (gameState.board[row][col] !== 0) return

    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    setMoveHistory(prev => [...prev, {
      number: prev.length + 1,
      player: humanColor,
      row,
      col,
    }])
    setAiThinking(true)
    ws.send(JSON.stringify({ type: 'move', row, col }))
  }, [gameStarted, gameState, aiThinking, humanColor])

  const handleUndo = useCallback(() => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    if (!gameStarted || gameState.game_over) return

    ws.send(JSON.stringify({ type: 'undo' }))
    // Remove last two moves (human + AI) from history
    setMoveHistory(prev => prev.slice(0, Math.max(0, prev.length - 2)))
  }, [gameStarted, gameState.game_over])

  const formatMove = (row: number, col: number) => `(${row},${col})`

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Board */}
      <div className="lg:col-span-2 flex flex-col gap-3">
        <GomokuBoard
          board={gameState.board}
          lastMove={gameState.last_move}
          onCellClick={handleCellClick}
          interactive={gameStarted && !gameState.game_over && !aiThinking && gameState.current_player === humanColor}
        />
        {/* Game Result Overlay */}
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
        {/* Game Controls */}
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Bot className="h-3.5 w-3.5" /> 游戏控制
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">选择模型</label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue placeholder="选择模型" />
                </SelectTrigger>
                <SelectContent>
                  {models.map(m => (
                    <SelectItem key={m.path} value={m.path}>
                      {m.name}
                    </SelectItem>
                  ))}
                  {models.length === 0 && (
                    <SelectItem value="default">默认模型</SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">玩家颜色</label>
              <div className="flex gap-2">
                <Button
                  variant={humanColor === 1 ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setHumanColor(1)}
                  disabled={gameStarted}
                  className="gap-1 flex-1"
                >
                  <Circle className="h-3.5 w-3.5" /> 执黑
                </Button>
                <Button
                  variant={humanColor === 2 ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setHumanColor(2)}
                  disabled={gameStarted}
                  className="gap-1 flex-1"
                >
                  <CircleDot className="h-3.5 w-3.5" /> 执白
                </Button>
              </div>
            </div>
            <div className="flex gap-2">
              <Button onClick={startNewGame} size="sm" className="gap-1 flex-1">
                <RotateCcw className="h-3.5 w-3.5" /> 新游戏
              </Button>
              <Button onClick={handleUndo} size="sm" variant="outline" className="gap-1" disabled={!gameStarted}>
                <Undo2 className="h-3.5 w-3.5" /> 悔棋
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Game Info */}
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <User className="h-3.5 w-3.5" /> 对局信息
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">当前回合</span>
              <Badge variant="outline" className="text-xs gap-1">
                {gameState.current_player === 1 ? (
                  <><Circle className="h-2.5 w-2.5" /> 黑棋</>
                ) : (
                  <><CircleDot className="h-2.5 w-2.5" /> 白棋</>
                )}
              </Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">步数</span>
              <span className="text-sm font-medium">{moveHistory.length}</span>
            </div>
            <Separator />
            <div className="flex justify-between items-center">
              <span className="text-xs text-muted-foreground">AI 状态</span>
              {aiThinking ? (
                <Badge variant="secondary" className="text-xs gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" /> 思考中
                </Badge>
              ) : (
                <Badge variant="outline" className="text-xs">等待中</Badge>
              )}
            </div>
            {aiValue !== null && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">AI 评估</span>
                <span className="text-sm font-medium">{aiValue.toFixed(3)}</span>
              </div>
            )}
            {aiThinkingTime !== null && (
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">AI 用时</span>
                <span className="text-sm font-medium">{aiThinkingTime.toFixed(1)}s</span>
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
                      <span className="text-muted-foreground w-6 shrink-0">{m.number}.</span>
                      <span className="shrink-0">
                        {m.player === 1 ? (
                          <Circle className="h-2.5 w-2.5" />
                        ) : (
                          <CircleDot className="h-2.5 w-2.5" />
                        )}
                      </span>
                      <span>{formatMove(m.row, m.col)}</span>
                      {m.value !== undefined && (
                        <span className="text-muted-foreground ml-auto">v={m.value.toFixed(2)}</span>
                      )}
                      {m.thinkingTime !== undefined && (
                        <span className="text-muted-foreground">{m.thinkingTime.toFixed(1)}s</span>
                      )}
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
