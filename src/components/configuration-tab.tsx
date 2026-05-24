'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Settings, Save, RefreshCw, Info } from 'lucide-react'

interface ConfigData {
  [key: string]: string | number | boolean
}

const CONFIG_GROUPS: { title: string; keys: string[]; editable: string[] }[] = [
  {
    title: '棋盘设置',
    keys: ['BOARD_SIZE', 'WIN_LENGTH'],
    editable: [],
  },
  {
    title: '模型架构',
    keys: ['NUM_RES_BLOCKS', 'NUM_FILTERS', 'INPUT_CHANNELS'],
    editable: ['NUM_RES_BLOCKS', 'NUM_FILTERS'],
  },
  {
    title: 'MCTS 参数',
    keys: ['NUM_SIMULATIONS', 'C_PUCT', 'DIRICHLET_ALPHA'],
    editable: ['NUM_SIMULATIONS', 'C_PUCT', 'DIRICHLET_ALPHA'],
  },
  {
    title: '训练参数',
    keys: ['LEARNING_RATE', 'TOTAL_ITERATIONS', 'BATCH_SIZE', 'NUM_ACTORS', 'NUM_GAMES_PER_ITER'],
    editable: ['LEARNING_RATE', 'TOTAL_ITERATIONS'],
  },
  {
    title: 'VCT 搜索',
    keys: ['USE_VCT', 'VCT_DEPTH_LIMIT', 'VCF_DEPTH_LIMIT', 'USE_MUST_MOVE', 'USE_PATTERN_INJECTION'],
    editable: [],
  },
  {
    title: '存储路径',
    keys: ['CHECKPOINT_DIR', 'BEST_MODEL_PATH'],
    editable: [],
  },
]

export default function ConfigurationTab() {
  const [config, setConfig] = useState<ConfigData>({})
  const [editedValues, setEditedValues] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState<string>('')

  const fetchConfig = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/config?XTransformPort=8000')
      if (res.ok) {
        const data = await res.json()
        setConfig(data)
        setEditedValues({})
      }
    } catch { /* silent */ }
    setLoading(false)
  }, [])

  // Use ref to avoid calling setState-bearing function directly in effect
  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await fetch('/api/config?XTransformPort=8000')
        if (res.ok) {
          const data = await res.json()
          setConfig(data)
          setEditedValues({})
        }
      } catch { /* silent */ }
      setLoading(false)
    }
    load()
  }, [])

  const isEditable = (key: string): boolean => {
    for (const group of CONFIG_GROUPS) {
      if (group.editable.includes(key)) return true
    }
    return false
  }

  const handleEdit = (key: string, value: string) => {
    setEditedValues(prev => ({ ...prev, [key]: value }))
  }

  const getDisplayValue = (key: string): string => {
    if (editedValues[key] !== undefined) return editedValues[key]
    const val = config[key]
    if (val === undefined) return ''
    return String(val)
  }

  const saveConfig = async () => {
    setSaving(true)
    setSaveMessage('')
    try {
      const body: Record<string, unknown> = {}
      for (const [key, val] of Object.entries(editedValues)) {
        const originalVal = config[key]
        if (typeof originalVal === 'number') {
          const parsed = Number(val)
          if (!isNaN(parsed)) {
            // Map to API field names
            const apiKey = key.toLowerCase()
            body[apiKey] = parsed
          }
        }
      }

      if (Object.keys(body).length === 0) {
        setSaveMessage('没有需要保存的更改')
        setSaving(false)
        return
      }

      const res = await fetch('/api/config?XTransformPort=8000', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (res.ok) {
        setSaveMessage('✅ 配置已保存，将在下次训练时生效')
        setEditedValues({})
        fetchConfig()
      } else {
        const data = await res.json()
        setSaveMessage(`❌ 保存失败: ${data.detail || '未知错误'}`)
      }
    } catch {
      setSaveMessage('❌ 连接失败')
    }
    setSaving(false)
    setTimeout(() => setSaveMessage(''), 5000)
  }

  const formatConfigKey = (key: string): string => {
    const labels: Record<string, string> = {
      BOARD_SIZE: '棋盘大小',
      WIN_LENGTH: '连珠数',
      NUM_SIMULATIONS: 'MCTS 模拟次数',
      NUM_RES_BLOCKS: '残差块数量',
      NUM_FILTERS: '卷积滤波器数',
      INPUT_CHANNELS: '输入通道数',
      C_PUCT: 'PUCT 探索常数',
      DIRICHLET_ALPHA: 'Dirichlet 噪声 α',
      LEARNING_RATE: '学习率',
      TOTAL_ITERATIONS: '总迭代次数',
      BATCH_SIZE: '批大小',
      USE_VCT: '启用 VCT 搜索',
      VCT_DEPTH_LIMIT: 'VCT 深度限制',
      VCF_DEPTH_LIMIT: 'VCF 深度限制',
      USE_MUST_MOVE: '启用必走检测',
      USE_PATTERN_INJECTION: '启用模式注入',
      CHECKPOINT_DIR: '检查点目录',
      BEST_MODEL_PATH: '最佳模型路径',
      NUM_ACTORS: '自对弈 Actor 数',
      NUM_GAMES_PER_ITER: '每迭代局数',
    }
    return labels[key] || key
  }

  return (
    <div className="space-y-4">
      {/* Save Bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs gap-1">
            <Info className="h-3 w-3" /> 修改将在下次训练时生效
          </Badge>
          {saveMessage && (
            <span className="text-xs">{saveMessage}</span>
          )}
        </div>
        <div className="flex gap-2">
          <Button onClick={fetchConfig} size="sm" variant="outline" className="gap-1" disabled={loading}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /> 刷新
          </Button>
          <Button onClick={saveConfig} size="sm" className="gap-1" disabled={saving || Object.keys(editedValues).length === 0}>
            <Save className="h-3.5 w-3.5" /> 保存修改
          </Button>
        </div>
      </div>

      {/* Config Groups */}
      {CONFIG_GROUPS.map(group => (
        <Card key={group.title} className="border-border/50">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <Settings className="h-3.5 w-3.5" /> {group.title}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {group.keys.map(key => (
                <div key={key} className="space-y-1">
                  <label className="text-xs text-muted-foreground">
                    {formatConfigKey(key)}
                    {isEditable(key) && (
                      <span className="ml-1 text-amber-500">✎</span>
                    )}
                  </label>
                  {isEditable(key) ? (
                    <Input
                      value={getDisplayValue(key)}
                      onChange={e => handleEdit(key, e.target.value)}
                      className="h-7 text-sm"
                      type={typeof config[key] === 'number' ? 'number' : 'text'}
                      step={typeof config[key] === 'number' && !Number.isInteger(config[key]) ? '0.01' : undefined}
                    />
                  ) : (
                    <div className="h-7 px-3 rounded-md border bg-muted/50 flex items-center text-sm">
                      {String(config[key] ?? '-')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
