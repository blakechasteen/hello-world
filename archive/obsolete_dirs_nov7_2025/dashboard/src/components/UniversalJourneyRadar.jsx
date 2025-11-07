import { motion, AnimatePresence } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { Map, Layers, TrendingUp, Zap, Eye, EyeOff } from 'lucide-react'

// Journey configurations
const JOURNEY_CONFIGS = {
  hero: { name: "Hero's Journey", color: '#F97316', icon: '⚔️' },
  business: { name: 'Business Journey', color: '#10B981', icon: '💼' },
  learning: { name: 'Learning Journey', color: '#3B82F6', icon: '📚' },
  scientific: { name: 'Scientific Journey', color: '#8B5CF6', icon: '🔬' },
  personal: { name: 'Personal Journey', color: '#EC4899', icon: '💖' },
  product: { name: 'Product Journey', color: '#06B6D4', icon: '🚀' }
}

export default function UniversalJourneyRadar({
  multiJourneyData,  // { hero: {...}, business: {...}, ... }
  mode = 'overlay',  // 'single' or 'overlay'
  activeJourneys = ['hero'],
  onJourneyToggle,
  universalPatterns = [],
  showResonance = true
}) {
  const canvasRef = useRef(null)
  const [hoveredStage, setHoveredStage] = useState(null)
  const [hoveredJourney, setHoveredJourney] = useState(null)
  const [animationProgress, setAnimationProgress] = useState(0)

  // Animation on mount or data change
  useEffect(() => {
    let frame
    let start = Date.now()

    const animate = () => {
      const progress = Math.min((Date.now() - start) / 1000, 1)
      setAnimationProgress(progress)

      if (progress < 1) {
        frame = requestAnimationFrame(animate)
      }
    }

    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [multiJourneyData, activeJourneys])

  // Draw the radar chart
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !multiJourneyData) return

    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height
    const centerX = width / 2
    const centerY = height / 2
    const maxRadius = Math.min(width, height) * 0.35

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw concentric circles (grid)
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)'
    ctx.lineWidth = 1
    for (let i = 1; i <= 5; i++) {
      ctx.beginPath()
      ctx.arc(centerX, centerY, (maxRadius / 5) * i, 0, Math.PI * 2)
      ctx.stroke()
    }

    // Get stage count (assume all journeys have 12 stages)
    const stageCount = 12

    // Draw axes and labels
    ctx.strokeStyle = 'rgba(139, 92, 246, 0.15)'
    ctx.lineWidth = 1

    for (let i = 0; i < stageCount; i++) {
      const angle = (i / stageCount) * Math.PI * 2 - Math.PI / 2
      const x = centerX + Math.cos(angle) * maxRadius
      const y = centerY + Math.sin(angle) * maxRadius

      // Axis line
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(x, y)
      ctx.stroke()

      // Stage number label
      const labelRadius = maxRadius + 25
      const labelX = centerX + Math.cos(angle) * labelRadius
      const labelY = centerY + Math.sin(angle) * labelRadius

      ctx.fillStyle = '#9CA3AF'
      ctx.font = '10px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText((i + 1).toString(), labelX, labelY)
    }

    // Draw each active journey's polygon
    activeJourneys.forEach((journeyId, journeyIndex) => {
      const journeyData = multiJourneyData[journeyId]
      if (!journeyData || !journeyData.stage_metrics) return

      const config = JOURNEY_CONFIGS[journeyId]
      if (!config) return

      // Get stage metrics in order
      const stages = Object.keys(journeyData.stage_metrics)
      if (stages.length !== 12) return // Ensure 12 stages

      // Calculate points for this journey's polygon
      const points = stages.map((stageName, index) => {
        const angle = (index / stageCount) * Math.PI * 2 - Math.PI / 2
        const metrics = journeyData.stage_metrics[stageName]
        const intensity = (metrics?.intensity || 0) * animationProgress
        const radius = maxRadius * intensity

        return {
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
          intensity,
          metrics
        }
      })

      // Determine opacity based on mode and hover
      let opacity = mode === 'overlay' ? 0.4 : 0.7
      if (hoveredJourney && hoveredJourney !== journeyId) {
        opacity = 0.15
      } else if (hoveredJourney === journeyId) {
        opacity = 0.9
      }

      // Draw filled polygon
      ctx.fillStyle = config.color + Math.floor(opacity * 0.3 * 255).toString(16).padStart(2, '0')
      ctx.beginPath()
      points.forEach((point, index) => {
        if (index === 0) ctx.moveTo(point.x, point.y)
        else ctx.lineTo(point.x, point.y)
      })
      ctx.closePath()
      ctx.fill()

      // Draw stroke
      ctx.strokeStyle = config.color + Math.floor(opacity * 255).toString(16).padStart(2, '0')
      ctx.lineWidth = mode === 'overlay' ? 2 : 2.5
      ctx.stroke()

      // Draw completion overlay (dashed)
      if (mode !== 'overlay' || hoveredJourney === journeyId) {
        const completionPoints = stages.map((stageName, index) => {
          const angle = (index / stageCount) * Math.PI * 2 - Math.PI / 2
          const metrics = journeyData.stage_metrics[stageName]
          const completion = (metrics?.completion || 0) * animationProgress
          const radius = maxRadius * completion

          return {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius
          }
        })

        ctx.strokeStyle = config.color + '60'
        ctx.lineWidth = 1.5
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        completionPoints.forEach((point, index) => {
          if (index === 0) ctx.moveTo(point.x, point.y)
          else ctx.lineTo(point.x, point.y)
        })
        ctx.closePath()
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Draw points at each stage
      points.forEach((point, index) => {
        if (point.intensity > 0.1) {
          ctx.beginPath()
          ctx.arc(point.x, point.y, mode === 'overlay' ? 3 : 4, 0, Math.PI * 2)
          ctx.fillStyle = config.color
          ctx.fill()

          // Highlight if hovered journey
          if (hoveredJourney === journeyId && point.intensity > 0.5) {
            ctx.strokeStyle = config.color
            ctx.lineWidth = 2
            ctx.stroke()
          }
        }
      })
    })

    // Draw resonance zones (overlay mode only)
    if (mode === 'overlay' && showResonance && universalPatterns.length > 0) {
      // Highlight stages with high cross-journey resonance
      universalPatterns.slice(0, 3).forEach((pattern, patternIndex) => {
        if (pattern.resonance_score > 0.6) {
          // Find the stage index for this pattern
          // This is simplified - in reality you'd map pattern to stage indices
          const alpha = Math.floor(pattern.resonance_score * 80)
          ctx.strokeStyle = `rgba(255, 215, 0, ${alpha / 255})`
          ctx.lineWidth = 3
          ctx.setLineDash([10, 5])

          // Draw a circle at resonance level
          ctx.beginPath()
          ctx.arc(centerX, centerY, maxRadius * pattern.avg_intensity, 0, Math.PI * 2)
          ctx.stroke()
          ctx.setLineDash([])
        }
      })
    }

    // Center indicator
    ctx.beginPath()
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2)
    ctx.fillStyle = mode === 'overlay' ? '#8B5CF6' : '#F97316'
    ctx.fill()

  }, [multiJourneyData, activeJourneys, hoveredJourney, animationProgress, mode, showResonance, universalPatterns])

  // Calculate stats and detect comparison mode
  const isComparisonMode = activeJourneys.length === 2

  const stats = {
    totalJourneys: activeJourneys.length,
    avgProgress: activeJourneys.length > 0 ?
      activeJourneys.reduce((sum, id) => sum + (multiJourneyData[id]?.overall_progress || 0), 0) / activeJourneys.length : 0,
    highestResonance: universalPatterns[0]?.pattern_name || 'None',
    resonanceScore: universalPatterns[0]?.resonance_score || 0
  }

  // Intelligent comparison insights (when exactly 2 journeys selected)
  const comparisonInsights = isComparisonMode && activeJourneys.length === 2 ?
    generateComparisonInsights(activeJourneys, multiJourneyData, universalPatterns) : null

  function generateComparisonInsights(journeyIds, data, patterns) {
    const [id1, id2] = journeyIds
    const journey1 = data[id1]
    const journey2 = data[id2]

    if (!journey1 || !journey2) return null

    const config1 = JOURNEY_CONFIGS[id1]
    const config2 = JOURNEY_CONFIGS[id2]

    // Calculate stage-by-stage differences
    const stages1 = Object.keys(journey1.stage_metrics)
    const stages2 = Object.keys(journey2.stage_metrics)

    let totalDifference = 0
    let alignedStages = 0
    let divergentStages = []
    let stronglyAlignedStages = []

    stages1.forEach((stage1, idx) => {
      const stage2 = stages2[idx]
      const metrics1 = journey1.stage_metrics[stage1]
      const metrics2 = journey2.stage_metrics[stage2]

      const diff = Math.abs(metrics1.intensity - metrics2.intensity)
      totalDifference += diff

      if (diff < 0.2) {
        alignedStages++
        if (metrics1.intensity > 0.5 && metrics2.intensity > 0.5) {
          stronglyAlignedStages.push({ stage1, stage2, intensity: (metrics1.intensity + metrics2.intensity) / 2 })
        }
      } else if (diff > 0.4) {
        divergentStages.push({
          stage1,
          stage2,
          diff,
          stronger: metrics1.intensity > metrics2.intensity ? id1 : id2
        })
      }
    })

    const avgDifference = totalDifference / stages1.length
    const alignmentScore = 1 - avgDifference // 0 = totally different, 1 = identical

    // Progress comparison
    const progressDiff = Math.abs(journey1.overall_progress - journey2.overall_progress)
    const leadingJourney = journey1.overall_progress > journey2.overall_progress ? id1 : id2
    const leadingName = journey1.overall_progress > journey2.overall_progress ? config1.name : config2.name

    // Find which patterns align these two journeys
    const sharedPatterns = patterns.filter(p =>
      p.journeys_matched[id1] && p.journeys_matched[id2]
    )

    // Intelligence extraction
    const insights = []

    // Overall alignment
    if (alignmentScore > 0.8) {
      insights.push({
        type: 'alignment',
        icon: '🎯',
        text: `Strong alignment: These journeys follow remarkably similar paths (${(alignmentScore * 100).toFixed(0)}% aligned)`
      })
    } else if (alignmentScore < 0.5) {
      insights.push({
        type: 'divergence',
        icon: '🔀',
        text: `Divergent paths: These journeys differ significantly (${(alignmentScore * 100).toFixed(0)}% aligned)`
      })
    } else {
      insights.push({
        type: 'moderate',
        icon: '⚖️',
        text: `Moderate alignment: Similar core pattern, different expressions (${(alignmentScore * 100).toFixed(0)}% aligned)`
      })
    }

    // Progress comparison
    if (progressDiff > 0.2) {
      insights.push({
        type: 'progress',
        icon: '📊',
        text: `${leadingName} is ahead by ${(progressDiff * 100).toFixed(0)}% - leading in narrative development`
      })
    }

    // Strongly aligned stages
    if (stronglyAlignedStages.length > 0) {
      const topAligned = stronglyAlignedStages.sort((a, b) => b.intensity - a.intensity)[0]
      insights.push({
        type: 'resonance',
        icon: '✨',
        text: `Strongest resonance at stage ${stages1.indexOf(topAligned.stage1) + 1}: "${topAligned.stage1}" ≈ "${topAligned.stage2}"`
      })
    }

    // Divergent stages
    if (divergentStages.length > 0) {
      const topDivergent = divergentStages.sort((a, b) => b.diff - a.diff)[0]
      const strongerConfig = JOURNEY_CONFIGS[topDivergent.stronger]
      insights.push({
        type: 'difference',
        icon: '⚡',
        text: `Key difference at stage ${stages1.indexOf(topDivergent.stage1) + 1}: ${strongerConfig.name} emphasizes this more`
      })
    }

    // Universal patterns
    if (sharedPatterns.length > 0) {
      insights.push({
        type: 'universal',
        icon: '🌟',
        text: `${sharedPatterns.length} universal patterns connect these journeys (${sharedPatterns[0].pattern_name}, etc.)`
      })
    }

    // Cross-domain learning insight
    const learningInsight = generateLearningInsight(id1, id2, journey1, journey2)
    if (learningInsight) {
      insights.push(learningInsight)
    }

    return {
      journey1: { id: id1, name: config1.name, icon: config1.icon, color: config1.color },
      journey2: { id: id2, name: config2.name, icon: config2.icon, color: config2.color },
      alignmentScore,
      alignedStages,
      stronglyAlignedStages,
      divergentStages,
      progressDiff,
      leadingJourney,
      sharedPatterns,
      insights
    }
  }

  function generateLearningInsight(id1, id2, data1, data2) {
    // Cross-domain learning patterns
    const learningMap = {
      'hero-business': {
        icon: '💡',
        text: 'Your startup journey mirrors the hero\'s quest - every entrepreneur faces the ordeal before claiming the reward'
      },
      'business-hero': {
        icon: '💡',
        text: 'Ancient myths predict your business challenges - the hero\'s ordeal is your cash crunch moment'
      },
      'hero-learning': {
        icon: '🧠',
        text: 'Learning mastery follows the hero\'s path - breakthrough comes after confronting the crisis'
      },
      'learning-hero': {
        icon: '🧠',
        text: 'Your learning struggles are part of an ancient pattern - trust the journey through difficulty'
      },
      'business-learning': {
        icon: '📈',
        text: 'Business growth IS learning - your startup challenges are opportunities for mastery'
      },
      'learning-business': {
        icon: '📈',
        text: 'Skill development mirrors startup building - deliberate practice is like MVP iteration'
      },
      'scientific-hero': {
        icon: '🔬',
        text: 'Scientific discovery follows the hero\'s journey - failed experiments are part of the ordeal'
      },
      'hero-scientific': {
        icon: '🔬',
        text: 'Research is a quest - your hypothesis is the threshold you cross into the unknown'
      },
      'personal-hero': {
        icon: '💫',
        text: 'Personal transformation mirrors ancient myths - your dark night is the hero\'s ordeal'
      },
      'hero-personal': {
        icon: '💫',
        text: 'Inner growth follows the hero\'s path - facing shadows leads to resurrection'
      },
      'product-business': {
        icon: '🚀',
        text: 'Product development mirrors business journey - critical bugs are like cash crunches'
      },
      'business-product': {
        icon: '🚀',
        text: 'Building a business is like shipping a product - both need market validation'
      }
    }

    const key1 = `${id1}-${id2}`
    const key2 = `${id2}-${id1}`

    return learningMap[key1] || learningMap[key2] || null
  }

  return (
    <motion.div
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-orange-300 flex items-center">
          <Layers className="w-5 h-5 mr-2" />
          Universal Journey Radar
        </h2>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">
            {mode === 'overlay' ? `${activeJourneys.length} overlays` : 'Single view'}
          </span>
          {showResonance && universalPatterns.length > 0 && (
            <span className="text-xs text-yellow-400">
              ✨ {universalPatterns.length} patterns
            </span>
          )}
        </div>
      </div>

      {/* Journey Toggles */}
      <div className="mb-4 flex flex-wrap gap-2">
        {Object.entries(JOURNEY_CONFIGS).map(([id, config]) => {
          const isActive = activeJourneys.includes(id)
          const hasData = multiJourneyData && multiJourneyData[id]

          return (
            <motion.button
              key={id}
              onClick={() => onJourneyToggle && onJourneyToggle(id)}
              onMouseEnter={() => setHoveredJourney(id)}
              onMouseLeave={() => setHoveredJourney(null)}
              disabled={!hasData}
              className={`
                px-3 py-1.5 rounded-lg text-xs font-medium border transition-all
                ${isActive
                  ? 'border-current opacity-100'
                  : 'border-gray-600 opacity-50'
                }
                ${!hasData && 'cursor-not-allowed opacity-30'}
                ${hoveredJourney === id && 'scale-105'}
              `}
              style={{
                color: config.color,
                backgroundColor: isActive ? config.color + '20' : 'transparent'
              }}
              whileHover={hasData ? { scale: 1.05 } : {}}
              whileTap={hasData ? { scale: 0.95 } : {}}
            >
              {config.icon} {config.name}
              {isActive && <Eye className="w-3 h-3 inline ml-1" />}
              {!isActive && hasData && <EyeOff className="w-3 h-3 inline ml-1" />}
            </motion.button>
          )
        })}
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={500}
          height={500}
          className="w-full max-w-lg mx-auto"
        />
      </div>

      {/* Legend */}
      {mode === 'overlay' && (
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 border-2 border-orange-500"></div>
            <span className="text-gray-400">Intensity (solid)</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 border-2 border-purple-500 border-dashed"></div>
            <span className="text-gray-400">Completion (dash)</span>
          </div>
          {showResonance && (
            <div className="flex items-center space-x-2 col-span-2">
              <div className="w-3 h-3 border-2 border-yellow-400 border-dashed"></div>
              <span className="text-gray-400">Universal resonance</span>
            </div>
          )}
        </div>
      )}

      {/* Stats */}
      <div className="mt-4 pt-4 border-t border-orange-500/20 space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">Average Progress</span>
          <div className="flex items-center space-x-2">
            <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-orange-500 to-purple-500"
                initial={{ width: 0 }}
                animate={{ width: `${stats.avgProgress * 100}%` }}
                transition={{ duration: 1, ease: 'easeOut' }}
              />
            </div>
            <span className="text-orange-300 font-semibold w-10 text-right">
              {(stats.avgProgress * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {showResonance && universalPatterns.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg"
          >
            <div className="text-xs text-yellow-300 font-semibold mb-1 flex items-center">
              <Zap className="w-3 h-3 mr-1" />
              Highest Resonance Pattern
            </div>
            <div className="text-sm text-yellow-200">
              {stats.highestResonance}
            </div>
            <div className="text-xs text-yellow-400 mt-1">
              {(stats.resonanceScore * 100).toFixed(0)}% resonance across {activeJourneys.length} journeys
            </div>

            {/* Show which journeys match */}
            {universalPatterns[0]?.journeys_matched && (
              <div className="mt-2 flex flex-wrap gap-1">
                {Object.keys(universalPatterns[0].journeys_matched).map(journeyId => (
                  <span
                    key={journeyId}
                    className="text-xs px-2 py-0.5 rounded"
                    style={{
                      backgroundColor: JOURNEY_CONFIGS[journeyId]?.color + '30',
                      color: JOURNEY_CONFIGS[journeyId]?.color
                    }}
                  >
                    {JOURNEY_CONFIGS[journeyId]?.icon}
                  </span>
                ))}
              </div>
            )}
          </motion.div>
        )}

        {/* Comparison Mode Intelligence */}
        {isComparisonMode && comparisonInsights && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-4 pt-4 border-t border-purple-500/30"
          >
            <div className="text-sm font-semibold text-purple-300 mb-3 flex items-center justify-between">
              <span>🔍 Comparison Insights</span>
              <span className="text-xs font-normal text-gray-400">
                {comparisonInsights.journey1.icon} vs {comparisonInsights.journey2.icon}
              </span>
            </div>

            {/* Alignment Score */}
            <div className="mb-3 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-purple-300">Journey Alignment</span>
                <span className="text-sm font-bold text-purple-200">
                  {(comparisonInsights.alignmentScore * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${comparisonInsights.alignmentScore * 100}%` }}
                  transition={{ duration: 1 }}
                />
              </div>
              <div className="mt-2 text-xs text-gray-400">
                {comparisonInsights.alignedStages}/12 stages aligned •
                {comparisonInsights.stronglyAlignedStages.length} strong resonances
              </div>
            </div>

            {/* Intelligent Insights */}
            <div className="space-y-2">
              {comparisonInsights.insights.map((insight, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`
                    p-3 rounded-lg text-xs
                    ${insight.type === 'alignment' ? 'bg-green-500/10 border border-green-500/30' :
                      insight.type === 'divergence' ? 'bg-red-500/10 border border-red-500/30' :
                      insight.type === 'resonance' ? 'bg-yellow-500/10 border border-yellow-500/30' :
                      insight.type === 'universal' ? 'bg-purple-500/10 border border-purple-500/30' :
                      'bg-blue-500/10 border border-blue-500/30'}
                  `}
                >
                  <span className="mr-2">{insight.icon}</span>
                  <span className="text-gray-200">{insight.text}</span>
                </motion.div>
              ))}
            </div>

            {/* Journey Progress Bars */}
            <div className="mt-4 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span>{comparisonInsights.journey1.icon}</span>
                  <span className="text-gray-400">{comparisonInsights.journey1.name}</span>
                </div>
                <span style={{ color: comparisonInsights.journey1.color }} className="font-semibold">
                  {(multiJourneyData[comparisonInsights.journey1.id]?.overall_progress * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full"
                  style={{ backgroundColor: comparisonInsights.journey1.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${multiJourneyData[comparisonInsights.journey1.id]?.overall_progress * 100}%` }}
                  transition={{ duration: 1 }}
                />
              </div>

              <div className="flex items-center justify-between text-xs mt-2">
                <div className="flex items-center space-x-2">
                  <span>{comparisonInsights.journey2.icon}</span>
                  <span className="text-gray-400">{comparisonInsights.journey2.name}</span>
                </div>
                <span style={{ color: comparisonInsights.journey2.color }} className="font-semibold">
                  {(multiJourneyData[comparisonInsights.journey2.id]?.overall_progress * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  className="h-full"
                  style={{ backgroundColor: comparisonInsights.journey2.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${multiJourneyData[comparisonInsights.journey2.id]?.overall_progress * 100}%` }}
                  transition={{ duration: 1 }}
                />
              </div>
            </div>
          </motion.div>
        )}

        {mode === 'overlay' && activeJourneys.length > 2 && (
          <div className="text-xs text-gray-500 text-center italic mt-4">
            Select exactly 2 journeys to see intelligent comparison • Hover badges to highlight
          </div>
        )}

        {mode === 'overlay' && activeJourneys.length === 1 && (
          <div className="text-xs text-gray-500 text-center italic mt-4">
            Select another journey to enable comparison mode
          </div>
        )}
      </div>
    </motion.div>
  )
}
