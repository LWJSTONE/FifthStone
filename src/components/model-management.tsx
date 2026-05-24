'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import {
  FolderOpen, Trash2, Download, RefreshCw, HardDrive, CheckCircle2,
} from 'lucide-react'

interface ModelInfo {
  name: string
  path: string
  size: number
  modified_time: number
}

interface CurrentModel {
  loaded: boolean
  model: {
    path: string
    name: string
    iteration: number
    best_elo: number
    total_steps: number
  } | null
}

export default function ModelManagement() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [currentModel, setCurrentModel] = useState<CurrentModel>({ loaded: false, model: null })
  const [loading, setLoading] = useState(false)
  const [loadPath, setLoadPath] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/models?XTransformPort=8000')
      if (res.ok) {
        const data = await res.json()
        setModels(data)
      }
    } catch { /* silent */ }
    setLoading(false)
  }, [])

  const fetchCurrentModel = useCallback(async () => {
    try {
      const res = await fetch('/api/models/current?XTransformPort=8000')
      if (res.ok) {
        const data = await res.json()
        setCurrentModel(data)
      }
    } catch { /* silent */ }
  }, [])

  // Use refs to avoid calling setState-bearing functions directly in effect
  useEffect(() => {
    const load = async () => {
      try {
        const [modelsRes, currentRes] = await Promise.all([
          fetch('/api/models?XTransformPort=8000'),
          fetch('/api/models/current?XTransformPort=8000'),
        ])
        if (modelsRes.ok) setModels(await modelsRes.json())
        if (currentRes.ok) setCurrentModel(await currentRes.json())
      } catch { /* silent */ }
    }
    load()
  }, [])

  const loadModel = async (path: string) => {
    setLoadPath(path)
    try {
      const res = await fetch('/api/models/load?XTransformPort=8000', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })
      if (res.ok) {
        fetchCurrentModel()
      } else {
        const data = await res.json()
        alert(`加载失败: ${data.detail || '未知错误'}`)
      }
    } catch {
      alert('连接失败')
    }
    setLoadPath(null)
  }

  const deleteModel = async (path: string) => {
    if (!confirm('确认删除此模型文件？此操作不可恢复。')) return
    try {
      // Note: No delete API in backend, this is a placeholder
      alert('删除功能需后端支持')
    } catch {
      alert('操作失败')
    }
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString('zh-CN')
  }

  return (
    <div className="space-y-4">
      {/* Current Model */}
      <Card className="border-border/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> 当前模型
          </CardTitle>
        </CardHeader>
        <CardContent>
          {currentModel.loaded && currentModel.model ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <p className="text-xs text-muted-foreground">模型名称</p>
                <p className="text-sm font-medium">{currentModel.model.name}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">迭代</p>
                <p className="text-sm font-medium">{currentModel.model.iteration}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">最佳 ELO</p>
                <p className="text-sm font-medium">{currentModel.model.best_elo.toFixed(0)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">总步数</p>
                <p className="text-sm font-medium">{currentModel.model.total_steps.toLocaleString()}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">未加载任何模型</p>
          )}
        </CardContent>
      </Card>

      {/* Model List */}
      <Card className="border-border/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm flex items-center gap-2">
                <HardDrive className="h-3.5 w-3.5" /> 可用模型
              </CardTitle>
              <CardDescription>检查点目录中的所有模型文件</CardDescription>
            </div>
            <Button onClick={fetchModels} size="sm" variant="outline" className="gap-1" disabled={loading}>
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /> 刷新
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <div className="text-center py-8">
              <FolderOpen className="h-10 w-10 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">暂无可用模型</p>
              <p className="text-xs text-muted-foreground">开始训练后将自动保存模型检查点</p>
            </div>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs">模型名称</TableHead>
                    <TableHead className="text-xs">大小</TableHead>
                    <TableHead className="text-xs hidden md:table-cell">修改时间</TableHead>
                    <TableHead className="text-xs text-right">操作</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {models.map(m => (
                    <TableRow key={m.path}>
                      <TableCell className="text-sm font-mono">{m.name}</TableCell>
                      <TableCell className="text-sm">{formatSize(m.size)}</TableCell>
                      <TableCell className="text-sm hidden md:table-cell">{formatDate(m.modified_time)}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex gap-1 justify-end">
                          <Button
                            onClick={() => loadModel(m.path)}
                            size="sm"
                            variant="outline"
                            className="h-7 text-xs gap-1"
                            disabled={loadPath === m.path}
                          >
                            {loadPath === m.path ? (
                              <RefreshCw className="h-3 w-3 animate-spin" />
                            ) : (
                              <Download className="h-3 w-3" />
                            )}
                            加载
                          </Button>
                          <Button
                            onClick={() => deleteModel(m.path)}
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs gap-1 text-destructive hover:text-destructive"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
