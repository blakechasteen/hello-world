import { motion } from 'framer-motion'
import { Users } from 'lucide-react'

const ARCHETYPE_COLORS = {
  hero: '#8B5CF6',
  mentor: '#10B981',
  threshold_guardian: '#F59E0B',
  shadow: '#DC2626',
  ally: '#3B82F6',
  trickster: '#EC4899'
}

export default function CharacterTimeline({ characters, domain }) {
  if (!characters || characters.length === 0) {
    return (
      <motion.div 
        className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-xl font-semibold text-orange-300 mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2" />
          Character Detection
        </h2>
        <div className="flex items-center justify-center h-32 text-gray-400">
          Characters will appear here during analysis
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h2 className="text-xl font-semibold text-orange-300 mb-4 flex items-center">
        <Users className="w-5 h-5 mr-2" />
        Character Timeline
      </h2>

      <div className="space-y-3">
        {characters.map((char, index) => {
          const color = ARCHETYPE_COLORS[char.archetype] || '#6B7280'
          
          return (
            <motion.div
              key={index}
              className="flex items-center space-x-3 p-3 bg-black/50 rounded-lg border border-gray-700"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <motion.div
                className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-white"
                style={{ backgroundColor: color }}
                whileHover={{ scale: 1.1 }}
                animate={{
                  boxShadow: [
                    `0 0 10px ${color}40`,
                    `0 0 20px ${color}60`,
                    `0 0 10px ${color}40`
                  ]
                }}
                transition={{ repeat: Infinity, duration: 2 }}
              >
                {char.name.charAt(0)}
              </motion.div>

              <div className="flex-1">
                <div className="font-semibold text-orange-300">
                  {char.name}
                </div>
                <div className="text-xs text-gray-300">
                  {char.archetype.replace('_', ' ')}
                </div>
              </div>

              <div className="text-xs text-gray-400">
                {new Date(char.timestamp).toLocaleTimeString()}
              </div>
            </motion.div>
          )
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ARCHETYPE_COLORS.hero }}></div>
            <span className="text-gray-300">Hero</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ARCHETYPE_COLORS.mentor }}></div>
            <span className="text-gray-300">Mentor</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ARCHETYPE_COLORS.shadow }}></div>
            <span className="text-gray-300">Shadow</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ARCHETYPE_COLORS.ally }}></div>
            <span className="text-gray-300">Ally</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ARCHETYPE_COLORS.threshold_guardian }}></div>
            <span className="text-gray-300">Guardian</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ARCHETYPE_COLORS.trickster }}></div>
            <span className="text-gray-300">Trickster</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
