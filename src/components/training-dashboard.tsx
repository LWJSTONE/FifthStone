'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import {
  Play, Square, Pause, RotateCcw, Activity, TrendingUp, Brain, Clock, Zap,
} from 'lucide-react'

interface TrainingStatus {
  running: boolean
  iteration: number
  total_steps: number
  best_elo: number
  current_stats: Record<string, number>
}

interface TrainingHistory {
  policy_loss: number[]
  value_loss: number[]
  elo: number[]
}

interface LogEntry {
  time: string
  message: string
  type: 'info' | 'success' | 'warning' | 'error'
}

export default function TrainingDashboard() {
  const [status, setStatus] = useState<TrainingStatus>({
    running: false, iteration: 0, total_steps: 0, best_elo: 0, current_stats: {},
  })
  const [history, setHistory] = useState<TrainingHistory>({
    policy_loss: [], value_loss: [], elo: [],
  })
  const [resumePath, setResumePath] = useState('')
  const [iterations, setIterations] = useState('')
  const [loading, setLoading] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([])

  const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const time = new Date().toLocaleTimeString('zh-CN')
    setLogs(prev => [{ time, message, type }, ...prev].slice(0, 100))
  }, [])

  // Data fetching helpers (not called directly in effects)
  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/train/status?XTransformPort=8000')
      if (res.ok) setStatus(await res.json())
    } catch { /* silent */ }
  }, [])

  const refreshHistory = useCallback(async () => {
    try {
      const res = await fetch('/api/train/history?XTransformPort=8000')
      if (res.ok) setHistory(await res.json())
    } catch { /* silent */ }
  }, [])

  // Initial fetch + polling — fetch inline so setState is in async callback
  useEffect(() => {
    const loadData = async () => {
      try {
        const [statusRes, historyRes] = await Promise.all([
          fetch('/api/train/status?XTransformPort=8000'),
          fetch('/api/train/history?XTransformPort=8000'),
        ])
        if (statusRes.ok) setStatus(await statusRes.json())
        if (historyRes.ok) setHistory(await historyRes.json())
      } catch { /* silent */ }
    }
    loadData()

    const id = setInterval(() => { loadData() }, 2000)
    return () => clearInterval(id)
  }, [])

  const startTraining = async () => {
    setLoading(true)
    addLog('正在启动训练...', 'info')
    try {
      const body: Record<string, unknown> = {}
      if (resumePath) body.resume_path = resumePath
      if (iterations) body.iterations = parseInt(iterations)
      const res = await fetch('/api/train/start?XTransformPort=8000', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (res.ok) {
        addLog('训练已启动', 'success')
        refreshStatus()
      } else {
        addLog(`启动失败: ${data.detail || '未知错误'}`, 'error')
      }
    } catch (err) {
      addLog(`连接错误: ${err}`, 'error')
    }
    setLoading(false)
  }

  const stopTraining = async () => {
    setLoading(true)
    addLog('正在停止训练...', 'warning')
    try {
      const res = await fetch('/api/train/stop?XTransformPort=8000', { method: 'POST' })
      const data = await res.json()
      if (res.ok) {
        addLog('训练将在当前迭代后停止', 'success')
        refreshStatus()
      } else {
        addLog(`停止失败: ${data.detail || '未知错误'}`, 'error')
      }
    } catch (err) {
      addLog(`连接错误: ${err}`, 'error')
    }
    setLoading(false)
  }

  // Build chart data
  const policyLossData = history.policy_loss.map((v, i) => ({ iteration: i + 1, value: v }))
  const valueLossData = history.value_loss.map((v, i) => ({ iteration: i + 1, value: v }))
  const eloData = history.elo.map((v, i) => ({ iteration: i + 1, value: v }))

  const currentStats = status.current_stats || {}

  return (
    <div className="space-y-4">
      {/* Training Controls */}
      <Card className="border-border/50">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Zap className="h-4 w-4 text-amber-500" />
            训练控制
          </CardTitle>
          <CardDescription>启动、停止或恢复模型训练</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-3">
            <div className="flex-1 min-w-[180px]">
              <label className="text-xs text-muted-foreground mb-1 block">恢复路径</label>
              <Input
                placeholder="checkpoints/model_iter_10.pt"
                value={resumePath}
                onChange={e => setResumePath(e.target.value)}
                disabled={status.running}
                className="h-8 text-sm"
              />
            </div>
            <div className="w-28">
              <label className="text-xs text-muted-foreground mb-1 block">迭代次数</label>
              <Input
                placeholder="默认"
                type="number"
                value={iterations}
                onChange={e => setIterations(e.target.value)}
                disabled={status.running}
                className="h-8 text-sm"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={startTraining}
              disabled={status.running || loading}
              size="sm"
              className="gap-1"
            >
              <Play className="h-3.5 w-3.5" /> 开始训练
            </Button>
            <Button
              onClick={stopTraining}
              disabled={!status.running || loading}
              variant="destructive"
              size="sm"
              className="gap-1"
            >
              <Square className="h-3.5 w-3.5" /> 停止
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Status Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Card className="border-border/50">
          <CardContent className="p-3">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="h-3.5 w-3.5 text-emerald-500" />
              <span className="text-xs text-muted-foreground">状态</span>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={status.running ? 'default' : 'secondary'} className="text-xs">
                {status.running ? '训练中' : '空闲'}
              </Badge>
            </div>
          </CardContent>
        </Card>
        <Card className="border-border/50">
          <CardContent className="p-3">
            <div className="flex items-center gap-2 mb-1">
              <RotateCcw className="h-3.5 w-3.5 text-blue-500" />
              <span className="text-xs text-muted-foreground">当前迭代</span>
            </div>
            <p className="text-xl font-bold">{status.iteration}</p>
          </CardContent>
        </Card>
        <Card className="border-border/50">
          <CardContent className="p-3">
            <div className="flex items-center gap-2 mb-1">
              <Clock className="h-3.5 w-3.5 text-purple-500" />
              <span className="text-xs text-muted-foreground">总步数</span>
            </div>
            <p className="text-xl font-bold">{status.total_steps.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card className="border-border/50">
          <CardContent className="p-3">
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp className="h-3.5 w-3.5 text-amber-500" />
              <span className="text-xs text-muted-foreground">最佳 ELO</span>
            </div>
            <p className="text-xl font-bold">{status.best_elo.toFixed(0)}</p>
          </CardContent>
        </Card>
      </div>

      {/* Current Stats */}
      {Object.keys(currentStats).length > 0 && (
        <Card className="border-border/50">
          <CardContent className="p-3">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="h-3.5 w-3.5 text-rose-500" />
              <span className="text-xs text-muted-foreground">当前统计</span>
            </div>
            <div className="flex flex-wrap gap-3 text-sm">
              {currentStats.policy_loss !== undefined && (
                <span>策略损失: <strong>{currentStats.policy_loss.toFixed(4)}</strong></span>
              )}
              {currentStats.value_loss !== undefined && (
                <span>价值损失: <strong>{currentStats.value_loss.toFixed(4)}</strong></span>
              )}
              {currentStats.lr !== undefined && (
                <span>学习率: <strong>{currentStats.lr.toExponential(3)}</strong></span>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">策略损失 (Policy Loss)</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {policyLossData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={policyLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="iteration" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <YAxis tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="value" stroke="#EF4444" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                  暂无数据
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">价值损失 (Value Loss)</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {valueLossData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={valueLossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="iteration" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <YAxis tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="value" stroke="#8B5CF6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                  暂无数据
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">ELO 评分</CardTitle>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {eloData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={eloData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="iteration" tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <YAxis tick={{ fontSize: 10 }} stroke="var(--muted-foreground)" />
                    <Tooltip contentStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="value" stroke="#F59E0B" strokeWidth={2} dot />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
                  暂无数据
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* Training Log */}
      <Card className="border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-3.5 w-3.5" /> 训练日志
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <ScrollArea className="h-40">
            {logs.length === 0 ? (
              <p className="text-sm text-muted-foreground">暂无日志记录</p>
            ) : (
              <div className="space-y-1">
                {logs.map((log, i) => (
                  <div key={i} className="flex gap-2 text-xs font-mono">
                    <span className="text-muted-foreground shrink-0">{log.time}</span>
                    <span className={
                      log.type === 'error' ? 'text-red-500' :
                      log.type === 'success' ? 'text-emerald-500' :
                      log.type === 'warning' ? 'text-amber-500' :
                      'text-foreground'
                    }>
                      {log.message}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
