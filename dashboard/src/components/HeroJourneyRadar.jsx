import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { Map, TrendingUp } from 'lucide-react'

const CAMPBELL_STAGES = [
  { id: 1, name: 'Ordinary World', shortName: 'Ordinary', color: '#6B7280' },
  { id: 2, name: 'Call to Adventure', shortName: 'Call', color: '#10B981' },
  { id: 3, name: 'Refusal of Call', shortName: 'Refusal', color: '#F59E0B' },
  { id: 4, name: 'Meeting Mentor', shortName: 'Mentor', color: '#3B82F6' },
  { id: 5, name: 'Crossing Threshold', shortName: 'Threshold', color: '#8B5CF6' },
  { id: 6, name: 'Tests & Allies', shortName: 'Tests', color: '#EC4899' },
  { id: 7, name: 'Inmost Cave', shortName: 'Cave', color: '#F59E0B' },
  { id: 8, name: 'Ordeal', shortName: 'Ordeal', color: '#DC2626' },
  { id: 9, name: 'Reward', shortName: 'Reward', color: '#10B981' },
  { id: 10, name: 'Road Back', shortName: 'Return', color: '#3B82F6' },
  { id: 11, name: 'Resurrection', shortName: 'Rebirth', color: '#8B5CF6' },
  { id: 12, name: 'Return with Elixir', shortName: 'Elixir', color: '#F59E0B' }
]

export default function HeroJourneyRadar({ journeyMetrics = {}, currentStage, domain }) {
  const canvasRef = useRef(null)
  const [hoveredStage, setHoveredStage] = useState(null)
  const [animationProgress, setAnimationProgress] = useState(0)

  // Animate on mount
  useEffect(() => {
    let frame
    let start = Date.now()

    const animate = () => {
      const progress = Math.min((Date.now() - start) / 1000, 1) // 1 second animation
      setAnimationProgress(progress)

      if (progress < 1) {
        frame = requestAnimationFrame(animate)
      }
    }

    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [journeyMetrics])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const maxRadius = Math.min(width, height) * 0.35

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw concentric circles (grid)
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.15)'
    ctx.lineWidth = 1
    for (let i = 1; i <= 5; i++) {
      ctx.beginPath()
      ctx.arc(centerX, centerY, (maxRadius / 5) * i, 0, Math.PI * 2)
      ctx.stroke()
    }

    // Draw axes for each stage
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.2)'
    ctx.lineWidth = 1
    CAMPBELL_STAGES.forEach((stage, index) => {
      const angle = (index / CAMPBELL_STAGES.length) * Math.PI * 2 - Math.PI / 2
      const x = centerX + Math.cos(angle) * maxRadius
      const y = centerY + Math.sin(angle) * maxRadius

      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(x, y)
      ctx.stroke()

      // Draw stage labels
      const labelRadius = maxRadius + 25
      const labelX = centerX + Math.cos(angle) * labelRadius
      const labelY = centerY + Math.sin(angle) * labelRadius

      ctx.fillStyle = stage.name === currentStage ? stage.color : '#9CA3AF'
      ctx.font = stage.name === currentStage ? 'bold 11px sans-serif' : '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(stage.shortName, labelX, labelY)
    })

    // Get metrics for each stage (default to 0.1 if not provided)
    const metrics = CAMPBELL_STAGES.map(stage => {
      const metric = journeyMetrics[stage.name] || { intensity: 0.1, completion: 0, relevance: 0.1 }
      // Apply animation progress
      return {
        intensity: metric.intensity * animationProgress,
        completion: metric.completion * animationProgress,
        relevance: metric.relevance * animationProgress
      }
    })

    // Draw data polygon (intensity metric)
    const intensityPoints = metrics.map((metric, index) => {
      const angle = (index / CAMPBELL_STAGES.length) * Math.PI * 2 - Math.PI / 2
      const radius = maxRadius * metric.intensity
      return {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      }
    })

    // Fill
    ctx.fillStyle = 'rgba(249, 115, 22, 0.2)'
    ctx.beginPath()
    intensityPoints.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y)
      else ctx.lineTo(point.x, point.y)
    })
    ctx.closePath()
    ctx.fill()

    // Stroke
    ctx.strokeStyle = 'rgba(249, 115, 22, 0.8)'
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw completion overlay (second metric)
    const completionPoints = metrics.map((metric, index) => {
      const angle = (index / CAMPBELL_STAGES.length) * Math.PI * 2 - Math.PI / 2
      const radius = maxRadius * metric.completion
      return {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      }
    })

    ctx.fillStyle = 'rgba(139, 92, 246, 0.15)'
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.6)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    completionPoints.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y)
      else ctx.lineTo(point.x, point.y)
    })
    ctx.closePath()
    ctx.fill()
    ctx.stroke()
    ctx.setLineDash([])

    // Draw points for each stage
    CAMPBELL_STAGES.forEach((stage, index) => {
      const angle = (index / CAMPBELL_STAGES.length) * Math.PI * 2 - Math.PI / 2
      const metric = metrics[index]
      const radius = maxRadius * metric.intensity
      const x = centerX + Math.cos(angle) * radius
      const y = centerY + Math.sin(angle) * radius

      // Point
      ctx.beginPath()
      ctx.arc(x, y, stage.name === currentStage ? 6 : 4, 0, Math.PI * 2)
      ctx.fillStyle = stage.name === currentStage ? stage.color : '#F97316'
      ctx.fill()

      if (stage.name === currentStage) {
        ctx.strokeStyle = stage.color
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })

    // Draw center point
    ctx.beginPath()
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2)
    ctx.fillStyle = '#F97316'
    ctx.fill()

  }, [journeyMetrics, currentStage, hoveredStage, animationProgress])

  // Calculate overall journey progress
  const overallProgress = Object.values(journeyMetrics).length > 0
    ? Object.values(journeyMetrics).reduce((sum, m) => sum + (m.completion || 0), 0) / 12
    : 0

  // Find dominant stage
  const dominantStage = Object.entries(journeyMetrics).reduce((max, [stage, metrics]) => {
    const score = (metrics.intensity || 0) * 0.5 + (metrics.relevance || 0) * 0.5
    return score > max.score ? { stage, score } : max
  }, { stage: null, score: 0 })

  return (
    <motion.div
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-orange-300 flex items-center">
          <Map className="w-5 h-5 mr-2" />
          Hero's Journey Radar
        </h2>
        <div className="text-xs text-gray-400">
          {Object.keys(journeyMetrics).length}/12 stages active
        </div>
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={400}
          height={400}
          className="w-full max-w-md mx-auto"
        />

        {/* Center overlay for current stage */}
        {currentStage && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center pointer-events-none"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-2 border border-orange-500/50">
              <div className="text-xs text-orange-300 font-semibold text-center">
                {currentStage}
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
          <span className="text-gray-400">Intensity (solid)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 border-2 border-purple-500 border-dashed rounded-full"></div>
          <span className="text-gray-400">Completion (dash)</span>
        </div>
      </div>

      {/* Stats */}
      <div className="mt-4 pt-4 border-t border-orange-500/20 space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">Overall Progress</span>
          <div className="flex items-center space-x-2">
            <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-orange-500 to-purple-500"
                initial={{ width: 0 }}
                animate={{ width: `${overallProgress * 100}%` }}
                transition={{ duration: 1, ease: 'easeOut' }}
              />
            </div>
            <span className="text-orange-300 font-semibold w-10 text-right">
              {(overallProgress * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {dominantStage.stage && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Dominant Stage</span>
            <span className="text-purple-300 font-semibold">
              {dominantStage.stage}
            </span>
          </div>
        )}

        {currentStage && journeyMetrics[currentStage] && (
          <motion.div
            className="mt-3 p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg"
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="text-xs text-orange-300 font-semibold mb-2">
              {currentStage} Metrics
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <div className="text-gray-500">Intensity</div>
                <div className="text-orange-300 font-bold">
                  {(journeyMetrics[currentStage].intensity * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-gray-500">Complete</div>
                <div className="text-purple-300 font-bold">
                  {(journeyMetrics[currentStage].completion * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-gray-500">Relevance</div>
                <div className="text-blue-300 font-bold">
                  {(journeyMetrics[currentStage].relevance * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      <div className="mt-3 text-xs text-gray-500 text-center italic">
        Real-time narrative stage tracking across 12 archetypal phases
      </div>
    </motion.div>
  )
}
