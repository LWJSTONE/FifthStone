'use client'

import React, { useRef, useEffect, useCallback, useState } from 'react'

const BOARD_SIZE = 15
const STAR_POINTS = [
  [3, 3], [3, 7], [3, 11],
  [7, 3], [7, 7], [7, 11],
  [11, 3], [11, 7], [11, 11],
]

interface GomokuBoardProps {
  board: number[][]            // 0=empty, 1=black, 2=white
  lastMove?: [number, number] | null
  onCellClick?: (row: number, col: number) => void
  interactive?: boolean
  highlightMoves?: [number, number][]  // optional moves to highlight
  size?: number
}

export default function GomokuBoard({
  board,
  lastMove,
  onCellClick,
  interactive = false,
  highlightMoves = [],
  size,
}: GomokuBoardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  // Use size prop directly when provided; otherwise track via state
  const [autoSize, setAutoSize] = useState(480)
  const canvasSize = size ?? autoSize

  // Responsive sizing via ResizeObserver (only when no size prop)
  useEffect(() => {
    if (size) return
    const updateSize = () => {
      if (containerRef.current) {
        const w = containerRef.current.offsetWidth
        setAutoSize(Math.min(w, 560))
      }
    }
    const observer = new ResizeObserver(updateSize)
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [size])

  const getCellSize = useCallback(() => {
    const padding = canvasSize * 0.04
    return (canvasSize - padding * 2) / (BOARD_SIZE - 1)
  }, [canvasSize])

  const getPadding = useCallback(() => {
    return canvasSize * 0.04
  }, [canvasSize])

  // Draw board
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = canvasSize * dpr
    canvas.height = canvasSize * dpr
    ctx.scale(dpr, dpr)

    const cellSize = getCellSize()
    const padding = getPadding()

    // Wood background
    const gradient = ctx.createLinearGradient(0, 0, canvasSize, canvasSize)
    gradient.addColorStop(0, '#DEB887')
    gradient.addColorStop(0.5, '#D2A96A')
    gradient.addColorStop(1, '#C8A25C')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, canvasSize, canvasSize)

    // Subtle wood grain texture
    ctx.save()
    ctx.globalAlpha = 0.06
    for (let i = 0; i < 40; i++) {
      const y = Math.random() * canvasSize
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.bezierCurveTo(
        canvasSize * 0.3, y + (Math.random() - 0.5) * 8,
        canvasSize * 0.7, y + (Math.random() - 0.5) * 8,
        canvasSize, y + (Math.random() - 0.5) * 4
      )
      ctx.strokeStyle = '#8B6914'
      ctx.lineWidth = Math.random() * 2 + 0.5
      ctx.stroke()
    }
    ctx.restore()

    // Grid lines
    ctx.strokeStyle = '#5C4033'
    ctx.lineWidth = 1
    for (let i = 0; i < BOARD_SIZE; i++) {
      const pos = padding + i * cellSize
      // Horizontal
      ctx.beginPath()
      ctx.moveTo(padding, pos)
      ctx.lineTo(padding + (BOARD_SIZE - 1) * cellSize, pos)
      ctx.stroke()
      // Vertical
      ctx.beginPath()
      ctx.moveTo(pos, padding)
      ctx.lineTo(pos, padding + (BOARD_SIZE - 1) * cellSize)
      ctx.stroke()
    }

    // Star points
    for (const [r, c] of STAR_POINTS) {
      const x = padding + c * cellSize
      const y = padding + r * cellSize
      ctx.beginPath()
      ctx.arc(x, y, cellSize * 0.1, 0, Math.PI * 2)
      ctx.fillStyle = '#5C4033'
      ctx.fill()
    }

    // Coordinate labels
    ctx.fillStyle = '#5C4033'
    ctx.font = `${Math.max(10, cellSize * 0.3)}px sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    for (let i = 0; i < BOARD_SIZE; i++) {
      // Top labels
      const labelX = padding + i * cellSize
      ctx.fillText(i.toString(), labelX, padding * 0.35)
      // Left labels
      const labelY = padding + i * cellSize
      ctx.fillText(i.toString(), padding * 0.35, labelY)
    }

    // Stones
    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        const piece = board[r]?.[c]
        if (piece === 0) continue
        const x = padding + c * cellSize
        const y = padding + r * cellSize
        const radius = cellSize * 0.42

        if (piece === 1) {
          // Black stone
          const grad = ctx.createRadialGradient(
            x - radius * 0.3, y - radius * 0.3, radius * 0.1,
            x, y, radius
          )
          grad.addColorStop(0, '#666')
          grad.addColorStop(0.6, '#222')
          grad.addColorStop(1, '#111')
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fillStyle = grad
          ctx.fill()
          // Shadow
          ctx.save()
          ctx.globalAlpha = 0.15
          ctx.beginPath()
          ctx.arc(x + 2, y + 2, radius, 0, Math.PI * 2)
          ctx.fillStyle = '#000'
          ctx.fill()
          ctx.restore()
          // Re-draw stone on top of shadow
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fillStyle = grad
          ctx.fill()
        } else {
          // White stone
          const grad = ctx.createRadialGradient(
            x - radius * 0.3, y - radius * 0.3, radius * 0.1,
            x, y, radius
          )
          grad.addColorStop(0, '#fff')
          grad.addColorStop(0.6, '#f0f0f0')
          grad.addColorStop(1, '#d8d8d8')
          // Shadow
          ctx.save()
          ctx.globalAlpha = 0.2
          ctx.beginPath()
          ctx.arc(x + 2, y + 2, radius, 0, Math.PI * 2)
          ctx.fillStyle = '#000'
          ctx.fill()
          ctx.restore()
          // Stone
          ctx.beginPath()
          ctx.arc(x, y, radius, 0, Math.PI * 2)
          ctx.fillStyle = grad
          ctx.fill()
          ctx.strokeStyle = '#bbb'
          ctx.lineWidth = 0.5
          ctx.stroke()
        }
      }
    }

    // Last move indicator
    if (lastMove) {
      const [r, c] = lastMove
      const x = padding + c * cellSize
      const y = padding + r * cellSize
      ctx.beginPath()
      ctx.arc(x, y, cellSize * 0.12, 0, Math.PI * 2)
      ctx.fillStyle = '#EF4444'
      ctx.fill()
    }

    // Highlight moves
    for (const [r, c] of highlightMoves) {
      const x = padding + c * cellSize
      const y = padding + r * cellSize
      ctx.beginPath()
      ctx.arc(x, y, cellSize * 0.15, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.3)'
      ctx.fill()
    }
  }, [board, lastMove, canvasSize, getCellSize, getPadding, highlightMoves])

  // Handle click
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!interactive || !onCellClick) return
    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const scaleX = canvasSize / rect.width
    const scaleY = canvasSize / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    const cellSize = getCellSize()
    const padding = getPadding()

    const col = Math.round((x - padding) / cellSize)
    const row = Math.round((y - padding) / cellSize)

    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
      onCellClick(row, col)
    }
  }, [interactive, onCellClick, canvasSize, getCellSize, getPadding])

  return (
    <div ref={containerRef} className="w-full flex justify-center">
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        style={{
          width: canvasSize,
          height: canvasSize,
          cursor: interactive ? 'pointer' : 'default',
        }}
        className="rounded-lg shadow-xl"
      />
    </div>
  )
}
