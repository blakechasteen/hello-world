import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Sparkles, Brain, TrendingUp, Users, BookOpen, 
  Zap, Target, Award, Activity, Lock, Unlock 
} from 'lucide-react'
import DomainSelector from './components/DomainSelector'
import MatryoshkaGates from './components/MatryoshkaGates'
import StreamingInput from './components/StreamingInput'
import ComplexityChart from './components/ComplexityChart'
import CharacterTimeline from './components/CharacterTimeline'
import CosmicTruthReveal from './components/CosmicTruthReveal'
import UniversalJourneyRadar from './components/UniversalJourneyRadar'
import LoopControl from './components/LoopControl'

function App() {
  const [selectedDomain, setSelectedDomain] = useState('mythology')
  const [streamingActive, setStreamingActive] = useState(false)
  const [loopActive, setLoopActive] = useState(false)
  const [loopStats, setLoopStats] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [unlockedGates, setUnlockedGates] = useState(new Set())
  const [complexityHistory, setComplexityHistory] = useState([])
  const [characters, setCharacters] = useState([])
  const [currentStage, setCurrentStage] = useState(null)
  const [journeyMetrics, setJourneyMetrics] = useState({})
  const [narrativeShifts, setNarrativeShifts] = useState([])

  // Multi-journey state
  const [multiJourneyData, setMultiJourneyData] = useState({})
  const [activeJourneys, setActiveJourneys] = useState(['hero', 'business', 'learning', 'scientific', 'personal', 'product'])
  const [universalPatterns, setUniversalPatterns] = useState([])
  
  const domains = [
    { id: 'mythology', name: 'Mythology', icon: Sparkles, color: '#FF4500' },
    { id: 'business', name: 'Business', icon: TrendingUp, color: '#FF6347' },
    { id: 'science', name: 'Science', icon: Brain, color: '#FF8C00' },
    { id: 'personal', name: 'Personal', icon: Users, color: '#FFA500' },
    { id: 'product', name: 'Product', icon: Target, color: '#FF7F50' },
    { id: 'history', name: 'History', icon: BookOpen, color: '#FFD700' }
  ]

  // Poll loop status when active
  useEffect(() => {
    if (!loopActive) return
    
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8000/api/loop/status')
        const data = await response.json()
        setLoopStats(data.stats)
        if (!data.active) {
          setLoopActive(false)
        }
      } catch (error) {
        console.error('Failed to fetch loop status:', error)
      }
    }, 1000)
    
    return () => clearInterval(interval)
  }, [loopActive])

  const handleLoopToggle = async () => {
    if (loopActive) {
      // Stop loop
      try {
        await fetch('http://localhost:8000/api/loop/stop', { method: 'POST' })
        setLoopActive(false)
        setLoopStats(null)
      } catch (error) {
        console.error('Failed to stop loop:', error)
      }
    } else {
      // Start loop
      try {
        const response = await fetch('http://localhost:8000/api/loop/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: 'batch',
            rate_limit: 5,
            auto_detect: true
          })
        })
        const data = await response.json()
        if (data.status === 'started') {
          setLoopActive(true)
        }
      } catch (error) {
        console.error('Failed to start loop:', error)
      }
    }
  }

  const handleStreamEvent = (event) => {
    console.log('Stream event:', event)
    
    switch(event.event_type) {
      case 'gate_unlocked':
        setUnlockedGates(prev => new Set([...prev, event.data.gate]))
        break
      case 'complexity_update':
        setComplexityHistory(prev => [...prev, {
          time: event.timestamp,
          complexity: event.data.complexity,
          confidence: event.data.confidence,
          depth: event.data.max_depth
        }])
        break
      case 'character_detected':
        setCharacters(prev => [...prev, {
          name: event.data.character,
          archetype: event.data.archetype,
          timestamp: event.timestamp,
          position: event.cumulative_text_length
        }])
        break
      case 'stage_transition':
        setCurrentStage(event.data.stage)
        break
      case 'journey_metrics':
        setJourneyMetrics(event.data.metrics)
        break
      case 'multi_journey_update':
        setMultiJourneyData(event.data.multi_journey_data)
        if (event.data.universal_patterns) {
          setUniversalPatterns(event.data.universal_patterns)
        }
        break
      case 'narrative_shift':
        setNarrativeShifts(prev => [...prev, {
          timestamp: event.timestamp,
          position: event.cumulative_text_length,
          description: event.data.description
        }])
        break
      case 'analysis_complete':
        setAnalysis(event.data.final_analysis)
        setStreamingActive(false)
        break
    }
  }

  const handleAnalyze = async (text, streaming = false) => {
    // Reset state
    setUnlockedGates(new Set())
    setComplexityHistory([])
    setCharacters([])
    setNarrativeShifts([])
    setAnalysis(null)
    setCurrentStage(null)
    setJourneyMetrics({})
    setMultiJourneyData({})
    setUniversalPatterns([])
    setStreamingActive(streaming)
    
    if (streaming) {
      // TODO: Connect to WebSocket for real-time streaming
      // For now, simulate streaming analysis
      simulateStreaming(text)
    } else {
      // Direct API call
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, domain: selectedDomain })
        })
        const result = await response.json()
        setAnalysis(result)
      } catch (error) {
        console.error('Analysis error:', error)
      }
    }
  }

  const handleJourneyToggle = (journeyId) => {
    setActiveJourneys(prev => {
      if (prev.includes(journeyId)) {
        return prev.filter(id => id !== journeyId)
      } else {
        return [...prev, journeyId]
      }
    })
  }

  const simulateStreaming = (text) => {
    // Simulate streaming events for demo
    const words = text.split(' ')
    let cumulative = 0
    let eventIndex = 0

    // Campbell stages sequence
    const stageSequence = [
      'Ordinary World', 'Call to Adventure', 'Refusal of Call', 'Meeting Mentor',
      'Crossing Threshold', 'Tests, Allies, Enemies', 'Approach Inmost Cave',
      'Ordeal', 'Reward', 'Road Back', 'Resurrection', 'Return with Elixir'
    ]

    // Journey names for multi-journey (all 6!)
    const journeyNames = {
      hero: ['Ordinary World', 'Call to Adventure', 'Refusal of Call', 'Meeting Mentor',
             'Crossing Threshold', 'Tests, Allies, Enemies', 'Approach Inmost Cave',
             'Ordeal', 'Reward', 'Road Back', 'Resurrection', 'Return with Elixir'],
      business: ['Ideation', 'Validation', 'Doubt & Fear', 'Advisor/Investor',
                 'MVP Launch', 'Early Traction', 'Preparing to Scale',
                 'Cash Crunch', 'Product-Market Fit', 'Scaling Operations',
                 'Market Leadership', 'Exit or Legacy'],
      learning: ['Unconscious Incompetence', 'Awareness', 'Overwhelm', 'Finding Teacher',
                 'Commitment to Practice', 'Deliberate Practice', 'Plateau Preparation',
                 'Learning Crisis', 'Breakthrough', 'Application', 'Mastery', 'Teaching Others'],
      scientific: ['Observation', 'Question', 'Doubt & Skepticism', 'Literature Review',
                   'Hypothesis', 'Experimental Design', 'Preparation',
                   'Failed Experiments', 'Discovery', 'Analysis',
                   'Theory Formation', 'Publication & Impact'],
      personal: ['Comfort Zone', 'Awakening', 'Resistance', 'Seeking Guidance',
                 'Decision to Change', 'Self-Discovery', 'Facing Shadows',
                 'Dark Night', 'Breakthrough', 'Integration', 'Wholeness', 'Service'],
      product: ['Problem Space', 'Solution Hypothesis', 'Technical Doubt', 'Research & Discovery',
                'Design Decision', 'Prototyping', 'Development Sprint',
                'Critical Bug', 'Working Product', 'User Testing',
                'Product-Market Fit', 'Scale & Impact']
    }

    const interval = setInterval(() => {
      if (eventIndex >= words.length) {
        clearInterval(interval)
        handleStreamEvent({
          event_type: 'analysis_complete',
          timestamp: Date.now(),
          data: {
            final_analysis: {
              max_depth: 'COSMIC',
              complexity: 0.92,
              confidence: 0.87,
              deepest_meaning: 'The journey transforms the traveler'
            }
          }
        })
        return
      }

      cumulative += words[eventIndex].length + 1
      const progress = eventIndex / words.length
      
      // Simulate different event types
      if (eventIndex % 10 === 0) {
        handleStreamEvent({
          event_type: 'complexity_update',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: {
            complexity: Math.min(0.3 + (eventIndex / words.length) * 0.7, 1.0),
            confidence: 0.5 + (eventIndex / words.length) * 0.4,
            max_depth: eventIndex < words.length * 0.3 ? 'SURFACE' : 
                       eventIndex < words.length * 0.5 ? 'SYMBOLIC' :
                       eventIndex < words.length * 0.7 ? 'ARCHETYPAL' :
                       eventIndex < words.length * 0.9 ? 'MYTHIC' : 'COSMIC'
          }
        })
      }
      
      // Gate unlocks at specific thresholds
      if (eventIndex === Math.floor(words.length * 0.3)) {
        handleStreamEvent({
          event_type: 'gate_unlocked',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { gate: 'SYMBOLIC', total_unlocked: 1 }
        })
      }
      if (eventIndex === Math.floor(words.length * 0.5)) {
        handleStreamEvent({
          event_type: 'gate_unlocked',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { gate: 'ARCHETYPAL', total_unlocked: 2 }
        })
      }
      if (eventIndex === Math.floor(words.length * 0.7)) {
        handleStreamEvent({
          event_type: 'gate_unlocked',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { gate: 'MYTHIC', total_unlocked: 3 }
        })
      }
      if (eventIndex === Math.floor(words.length * 0.9)) {
        handleStreamEvent({
          event_type: 'gate_unlocked',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { gate: 'COSMIC', total_unlocked: 4 }
        })
      }
      
      // Stage transitions and journey metrics
      const stageIndex = Math.floor(progress * 12)
      if (eventIndex % 15 === 0 && stageIndex < stageSequence.length) {
        const currentStageName = stageSequence[stageIndex]

        // Transition to new stage
        handleStreamEvent({
          event_type: 'stage_transition',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { stage: currentStageName }
        })

        // Update journey metrics for all stages
        const metrics = {}
        stageSequence.forEach((stage, idx) => {
          if (idx <= stageIndex) {
            // Completed or current stages
            const isComplete = idx < stageIndex
            const isCurrent = idx === stageIndex

            metrics[stage] = {
              intensity: isCurrent ? 0.8 + Math.random() * 0.2 : 0.4 + Math.random() * 0.3,
              completion: isComplete ? 0.9 + Math.random() * 0.1 : progress * 0.7,
              relevance: 0.6 + Math.random() * 0.4
            }
          } else {
            // Future stages - minimal activity
            metrics[stage] = {
              intensity: 0.1 + Math.random() * 0.1,
              completion: 0,
              relevance: 0.2 + Math.random() * 0.2
            }
          }
        })

        handleStreamEvent({
          event_type: 'journey_metrics',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { metrics }
        })

        // Multi-journey update
        const multiJourney = {}
        Object.keys(journeyNames).forEach(journeyId => {
          const stages = journeyNames[journeyId]
          const journeyMetrics = {}

          stages.forEach((stageName, idx) => {
            const intensityVariance = journeyId === 'hero' ? 1.0 :
                                     journeyId === 'business' ? 1.1 : 0.9
            const isAtStage = idx === stageIndex
            const isPast = idx < stageIndex

            journeyMetrics[stageName] = {
              intensity: isAtStage ? (0.7 + Math.random() * 0.2) * intensityVariance :
                        isPast ? (0.3 + Math.random() * 0.2) * intensityVariance :
                        (0.05 + Math.random() * 0.1) * intensityVariance,
              completion: isPast ? 0.85 + Math.random() * 0.15 :
                         isAtStage ? progress * 0.6 :
                         0,
              relevance: 0.5 + Math.random() * 0.3
            }
          })

          multiJourney[journeyId] = {
            stage_metrics: journeyMetrics,
            overall_progress: (stageIndex / 12) * 0.8 + progress * 0.2,
            current_stage: stages[stageIndex],
            dominant_stage: stages[stageIndex]
          }
        })

        // Create universal patterns (across all 6 journeys!)
        const patterns = [
          {
            pattern_name: 'The Commitment',
            resonance_score: stageIndex >= 4 ? 0.85 : 0.3,
            avg_intensity: 0.75,
            journeys_matched: {
              hero: { stage: 'Crossing Threshold' },
              business: { stage: 'MVP Launch' },
              learning: { stage: 'Commitment to Practice' },
              scientific: { stage: 'Hypothesis' },
              personal: { stage: 'Decision to Change' },
              product: { stage: 'Design Decision' }
            }
          },
          {
            pattern_name: 'The Crisis',
            resonance_score: stageIndex >= 7 ? 0.92 : 0.2,
            avg_intensity: 0.85,
            journeys_matched: {
              hero: { stage: 'Ordeal' },
              business: { stage: 'Cash Crunch' },
              learning: { stage: 'Learning Crisis' },
              scientific: { stage: 'Failed Experiments' },
              personal: { stage: 'Dark Night' },
              product: { stage: 'Critical Bug' }
            }
          },
          {
            pattern_name: 'The Breakthrough',
            resonance_score: stageIndex >= 8 ? 0.88 : 0.15,
            avg_intensity: 0.8,
            journeys_matched: {
              hero: { stage: 'Reward' },
              business: { stage: 'Product-Market Fit' },
              learning: { stage: 'Breakthrough' },
              scientific: { stage: 'Discovery' },
              personal: { stage: 'Breakthrough' },
              product: { stage: 'Working Product' }
            }
          }
        ]

        handleStreamEvent({
          event_type: 'multi_journey_update',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: {
            multi_journey_data: multiJourney,
            universal_patterns: patterns
          }
        })
      }

      // Character detection
      if (eventIndex === Math.floor(words.length * 0.2)) {
        handleStreamEvent({
          event_type: 'character_detected',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { character: 'Hero', archetype: 'hero' }
        })
      }
      if (eventIndex === Math.floor(words.length * 0.4)) {
        handleStreamEvent({
          event_type: 'character_detected',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { character: 'Mentor', archetype: 'mentor' }
        })
      }
      
      // Narrative shift
      if (eventIndex === Math.floor(words.length * 0.65)) {
        handleStreamEvent({
          event_type: 'narrative_shift',
          timestamp: Date.now(),
          cumulative_text_length: cumulative,
          data: { description: 'Dramatic narrative turn detected' }
        })
      }
      
      eventIndex++
    }, 50) // 20 words/second
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black">
      {/* Header */}
      <motion.header 
        className="border-b border-orange-500/30 bg-black/80 backdrop-blur-sm sticky top-0 z-50"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: 'spring', stiffness: 100 }}
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Sparkles className="w-8 h-8 text-orange-400 animate-fire" />
              <h1 className="text-3xl font-bold text-transparent bg-gradient-to-r from-orange-400 via-red-500 to-orange-400 bg-clip-text">
                🔥 mythRL
              </h1>
              <span className="text-sm text-orange-300 font-mono">
                v1.0 • Dark Fire Mode 🔥
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              {streamingActive && (
                <motion.div 
                  className="flex items-center space-x-2 bg-orange-500/20 px-4 py-2 rounded-full border border-orange-400/30"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                >
                  <Activity className="w-4 h-4 text-orange-400" />
                  <span className="text-sm text-orange-300">🔥 Streaming...</span>
                </motion.div>
              )}
              
              <div className="text-sm text-gray-300">
                🔥 {unlockedGates.size} / 5 Gates Unlocked
              </div>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Input & Control */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <DomainSelector 
              domains={domains}
              selected={selectedDomain}
              onChange={setSelectedDomain}
            />
            
            <LoopControl 
              onToggle={handleLoopToggle}
              active={loopActive}
              stats={loopStats}
            />
            
            <StreamingInput 
              onAnalyze={handleAnalyze}
              streaming={streamingActive}
              domain={selectedDomain}
            />
            
            <MatryoshkaGates 
              unlockedGates={unlockedGates}
              currentDepth={analysis?.max_depth}
            />
          </div>

          {/* Center Column - Visualizations */}
          <div className="col-span-12 lg:col-span-5 space-y-6">
            <ComplexityChart 
              data={complexityHistory}
              shifts={narrativeShifts}
            />
            
            <CharacterTimeline 
              characters={characters}
              domain={selectedDomain}
            />
            
            <UniversalJourneyRadar
              multiJourneyData={multiJourneyData}
              mode="overlay"
              activeJourneys={activeJourneys}
              onJourneyToggle={handleJourneyToggle}
              universalPatterns={universalPatterns}
              showResonance={true}
            />
          </div>

          {/* Right Column - Insights */}
          <div className="col-span-12 lg:col-span-3 space-y-6">
            {analysis && (
              <>
                <CosmicTruthReveal 
                  truth={analysis.deepest_meaning}
                  complexity={analysis.complexity}
                  confidence={analysis.confidence}
                />
                
                <motion.div 
                  className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <h3 className="text-lg font-semibold text-orange-300 mb-4 flex items-center">
                    <Award className="w-5 h-5 mr-2" />
                    🔥 Analysis Summary
                  </h3>
                  
                  <div className="space-y-3 text-sm">
                    <div>
                      <span className="text-gray-400">Max Depth:</span>
                      <span className="float-right font-semibold text-purple-300">
                        {analysis.max_depth}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Complexity:</span>
                      <span className="float-right font-semibold text-purple-300">
                        {(analysis.complexity * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Confidence:</span>
                      <span className="float-right font-semibold text-purple-300">
                        {(analysis.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Gates Unlocked:</span>
                      <span className="float-right font-semibold text-purple-300">
                        {unlockedGates.size} / 5
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Characters:</span>
                      <span className="float-right font-semibold text-purple-300">
                        {characters.length}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Shifts:</span>
                      <span className="float-right font-semibold text-purple-300">
                        {narrativeShifts.length}
                      </span>
                    </div>
                  </div>
                </motion.div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-purple-500/30 bg-slate-900/50 backdrop-blur-sm mt-12">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div>
              Built with 🚀 mythRL • Cross-Domain Narrative Intelligence
            </div>
            <div className="flex space-x-4">
              <span>Real-time Streaming</span>
              <span>•</span>
              <span>6 Domains</span>
              <span>•</span>
              <span>5 Depth Levels</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
