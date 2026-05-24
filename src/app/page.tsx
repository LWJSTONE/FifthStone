'use client'

import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import TrainingDashboard from '@/components/training-dashboard'
import HumanVsAI from '@/components/human-vs-ai'
import AIVsAI from '@/components/ai-vs-ai'
import ModelManagement from '@/components/model-management'
import ConfigurationTab from '@/components/configuration-tab'
import {
  Activity, Gamepad2, Swords, HardDrive, Settings, Wifi, WifiOff, BrainCircuit,
} from 'lucide-react'

export default function Home() {
  const [activeTab, setActiveTab] = useState('training')
  const [backendOnline, setBackendOnline] = useState(false)
  const [checkingBackend, setCheckingBackend] = useState(true)

  // Check backend health
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('/api/health?XTransformPort=8000', { signal: AbortSignal.timeout(3000) })
        setBackendOnline(res.ok)
      } catch {
        setBackendOnline(false)
      }
      setCheckingBackend(false)
    }
    checkHealth()
    const interval = setInterval(checkHealth, 15000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BrainCircuit className="h-6 w-6 text-amber-500" />
              <div>
                <h1 className="text-lg font-bold tracking-tight">五子棋 AI 训练平台</h1>
                <p className="text-xs text-muted-foreground">AlphaZero-based Gomoku AI</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {checkingBackend ? (
                <Badge variant="secondary" className="text-xs gap-1">
                  <Activity className="h-3 w-3 animate-pulse" /> 检测中...
                </Badge>
              ) : backendOnline ? (
                <Badge variant="default" className="text-xs gap-1 bg-emerald-600 hover:bg-emerald-700">
                  <Wifi className="h-3 w-3" /> 后端在线
                </Badge>
              ) : (
                <Badge variant="destructive" className="text-xs gap-1">
                  <WifiOff className="h-3 w-3" /> 后端离线
                </Badge>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-4">
        {!backendOnline && !checkingBackend && (
          <Card className="mb-4 border-amber-500/30 bg-amber-500/5">
            <CardContent className="p-3">
              <div className="flex items-center gap-2 text-sm">
                <WifiOff className="h-4 w-4 text-amber-500" />
                <span className="text-amber-600">
                  无法连接到 AI 后端服务 (端口 8000)。请确保后端服务已启动：
                  <code className="mx-1 px-1 py-0.5 bg-muted rounded text-xs">
                    uvicorn api_server:app --host 0.0.0.0 --port 8000
                  </code>
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="grid w-full grid-cols-5 h-auto">
            <TabsTrigger value="training" className="gap-1.5 text-xs sm:text-sm py-2">
              <Activity className="h-3.5 w-3.5 hidden sm:block" /> 训练仪表盘
            </TabsTrigger>
            <TabsTrigger value="play" className="gap-1.5 text-xs sm:text-sm py-2">
              <Gamepad2 className="h-3.5 w-3.5 hidden sm:block" /> 人机对弈
            </TabsTrigger>
            <TabsTrigger value="battle" className="gap-1.5 text-xs sm:text-sm py-2">
              <Swords className="h-3.5 w-3.5 hidden sm:block" /> AI对战
            </TabsTrigger>
            <TabsTrigger value="models" className="gap-1.5 text-xs sm:text-sm py-2">
              <HardDrive className="h-3.5 w-3.5 hidden sm:block" /> 模型管理
            </TabsTrigger>
            <TabsTrigger value="config" className="gap-1.5 text-xs sm:text-sm py-2">
              <Settings className="h-3.5 w-3.5 hidden sm:block" /> 配置
            </TabsTrigger>
          </TabsList>

          <TabsContent value="training" className="mt-4">
            <TrainingDashboard />
          </TabsContent>

          <TabsContent value="play" className="mt-4">
            <HumanVsAI />
          </TabsContent>

          <TabsContent value="battle" className="mt-4">
            <AIVsAI />
          </TabsContent>

          <TabsContent value="models" className="mt-4">
            <ModelManagement />
          </TabsContent>

          <TabsContent value="config" className="mt-4">
            <ConfigurationTab />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 bg-card/30 mt-auto">
        <div className="max-w-7xl mx-auto px-4 py-3 text-center text-xs text-muted-foreground">
          Gomoku AI Training Platform · 基于 AlphaZero 算法 · MCTS + 残差网络
        </div>
      </footer>
    </div>
  )
}
