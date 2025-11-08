import { motion } from 'framer-motion'
import { Lock, Unlock } from 'lucide-react'

const GATES = [
  { id: 'SURFACE', name: 'Surface', color: '#6B7280', threshold: 0.0 },
  { id: 'SYMBOLIC', name: 'Symbolic', color: '#10B981', threshold: 0.3 },
  { id: 'ARCHETYPAL', name: 'Archetypal', color: '#F59E0B', threshold: 0.5 },
  { id: 'MYTHIC', name: 'Mythic', color: '#EC4899', threshold: 0.7 },
  { id: 'COSMIC', name: 'Cosmic', color: '#8B5CF6', threshold: 0.85 },
]

export default function MatryoshkaGates({ unlockedGates, currentDepth }) {
  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <h2 className="text-xl font-semibold text-orange-300 mb-4">
        🔥 Matryoshka Gates
      </h2>

      <div className="space-y-3">
        {GATES.map((gate, index) => {
          const isUnlocked = unlockedGates.has(gate.id)
          const isCurrent = currentDepth === gate.id
          
          return (
            <motion.div
              key={gate.id}
              className={`
                relative p-4 rounded-lg border-2 transition-all
                ${isUnlocked 
                  ? 'border-orange-500 bg-orange-500/10' 
                  : 'border-gray-700 bg-black/50'
                }
                ${isCurrent ? 'ring-2 ring-orange-400' : ''}
              `}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  {isUnlocked ? (
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', stiffness: 200 }}
                    >
                      <Unlock 
                        className="w-5 h-5" 
                        style={{ color: gate.color }}
                      />
                    </motion.div>
                  ) : (
                    <Lock className="w-5 h-5 text-gray-600" />
                  )}
                  
                  <div>
                    <div className={`font-semibold ${isUnlocked ? 'text-orange-300' : 'text-gray-400'}`}>
                      {gate.name}
                    </div>
                    <div className="text-xs text-gray-600">
                      Threshold: {gate.threshold.toFixed(2)}
                    </div>
                  </div>
                </div>

                {isUnlocked && (
                  <motion.div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: gate.color }}
                    animate={{ 
                      boxShadow: [
                        `0 0 10px ${gate.color}`,
                        `0 0 20px ${gate.color}`,
                        `0 0 10px ${gate.color}`
                      ]
                    }}
                    transition={{ repeat: Infinity, duration: 2 }}
                  />
                )}
              </div>

              {isUnlocked && (
                <motion.div
                  className="absolute inset-0 rounded-lg pointer-events-none"
                  style={{ 
                    boxShadow: `0 0 20px ${gate.color}30` 
                  }}
                  animate={{ opacity: [0.3, 0.6, 0.3] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
              )}
            </motion.div>
          )
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="text-sm text-gray-300 text-center">
          {unlockedGates.size === 0 && "Begin analysis to unlock gates"}
          {unlockedGates.size > 0 && unlockedGates.size < 5 && `${unlockedGates.size}/5 gates unlocked`}
          {unlockedGates.size === 5 && "🌌 All gates unlocked! COSMIC depth achieved!"}
        </div>
      </div>
    </motion.div>
  )
}
