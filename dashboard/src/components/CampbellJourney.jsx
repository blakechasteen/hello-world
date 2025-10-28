import { motion } from 'framer-motion'
import { Map } from 'lucide-react'

const CAMPBELL_STAGES = [
  { id: 1, name: 'Ordinary World', color: '#6B7280', angle: 0 },
  { id: 2, name: 'Call to Adventure', color: '#10B981', angle: 30 },
  { id: 3, name: 'Refusal of Call', color: '#F59E0B', angle: 60 },
  { id: 4, name: 'Meeting Mentor', color: '#3B82F6', angle: 90 },
  { id: 5, name: 'Crossing Threshold', color: '#8B5CF6', angle: 120 },
  { id: 6, name: 'Tests, Allies, Enemies', color: '#EC4899', angle: 150 },
  { id: 7, name: 'Approach Inmost Cave', color: '#F59E0B', angle: 180 },
  { id: 8, name: 'Ordeal', color: '#DC2626', angle: 210 },
  { id: 9, name: 'Reward', color: '#10B981', angle: 240 },
  { id: 10, name: 'Road Back', color: '#3B82F6', angle: 270 },
  { id: 11, name: 'Resurrection', color: '#8B5CF6', angle: 300 },
  { id: 12, name: 'Return with Elixir', color: '#F59E0B', angle: 330 }
]

export default function CampbellJourney({ stage, domain }) {
  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h2 className="text-xl font-semibold text-orange-300 mb-4 flex items-center">
        <Map className="w-5 h-5 mr-2" />
        Hero's Journey
      </h2>

      <div className="relative w-full aspect-square max-w-sm mx-auto">
        {/* Center circle */}
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.div 
            className="w-20 h-20 rounded-full bg-gradient-to-br from-orange-600 to-pink-600 flex items-center justify-center"
            animate={{
              boxShadow: [
                '0 0 20px rgba(139, 92, 246, 0.5)',
                '0 0 40px rgba(139, 92, 246, 0.8)',
                '0 0 20px rgba(139, 92, 246, 0.5)'
              ]
            }}
            transition={{ repeat: Infinity, duration: 3 }}
          >
            <span className="text-2xl">🌟</span>
          </motion.div>
        </div>

        {/* Stage circles */}
        {CAMPBELL_STAGES.map((s, index) => {
          const radius = 120
          const x = 50 + radius * Math.cos((s.angle * Math.PI) / 180) / 2
          const y = 50 + radius * Math.sin((s.angle * Math.PI) / 180) / 2
          const isActive = stage === s.name

          return (
            <motion.div
              key={s.id}
              className="absolute"
              style={{
                left: `${x}%`,
                top: `${y}%`,
                transform: 'translate(-50%, -50%)'
              }}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: index * 0.05 }}
            >
              <motion.div
                className={`
                  w-10 h-10 rounded-full border-2 flex items-center justify-center
                  ${isActive 
                    ? 'border-orange-400 bg-orange-500 scale-125' 
                    : 'border-gray-600 bg-slate-800'
                  }
                `}
                style={{ 
                  borderColor: isActive ? s.color : undefined 
                }}
                whileHover={{ scale: isActive ? 1.3 : 1.1 }}
                animate={isActive ? {
                  boxShadow: [
                    `0 0 10px ${s.color}`,
                    `0 0 20px ${s.color}`,
                    `0 0 10px ${s.color}`
                  ]
                } : {}}
                transition={{ repeat: Infinity, duration: 2 }}
              >
                <span className="text-xs font-bold text-white">{s.id}</span>
              </motion.div>
            </motion.div>
          )
        })}
      </div>

      {stage && (
        <motion.div 
          className="mt-4 p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg text-center"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="text-sm text-orange-300 font-semibold">
            Current Stage: {stage}
          </div>
        </motion.div>
      )}

      <div className="mt-4 text-xs text-gray-400 text-center">
        12 stages of transformation across all domains
      </div>
    </motion.div>
  )
}
