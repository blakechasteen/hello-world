import { motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'

export default function CosmicTruthReveal({ truth, complexity, confidence }) {
  if (!truth) return null

  const isCosmicLevel = complexity >= 0.85 && confidence >= 0.7

  return (
    <motion.div 
      className={`
        bg-gradient-to-br from-orange-900/50 to-pink-900/50 backdrop-blur-sm 
        rounded-xl p-6 border-2 relative overflow-hidden
        ${isCosmicLevel ? 'border-orange-400 animate-glow' : 'border-orange-500/30'}
      `}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 100 }}
    >
      {isCosmicLevel && (
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-orange-500/20 to-pink-500/20"
          animate={{
            opacity: [0.3, 0.6, 0.3],
            scale: [1, 1.05, 1]
          }}
          transition={{ repeat: Infinity, duration: 3 }}
        />
      )}

      <div className="relative z-10">
        <div className="flex items-center space-x-2 mb-3">
          <Sparkles className={`w-5 h-5 ${isCosmicLevel ? 'text-yellow-300' : 'text-orange-400'}`} />
          <h3 className="text-lg font-semibold text-orange-300">
            {isCosmicLevel ? '🌌 Cosmic Truth' : 'Deepest Meaning'}
          </h3>
        </div>

        <motion.p 
          className={`text-sm leading-relaxed ${isCosmicLevel ? 'text-white font-medium' : 'text-gray-300'}`}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          "{truth}"
        </motion.p>

        {isCosmicLevel && (
          <motion.div
            className="mt-4 flex items-center justify-center space-x-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="w-2 h-2 bg-yellow-300 rounded-full"
                animate={{
                  scale: [1, 1.5, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{
                  repeat: Infinity,
                  duration: 2,
                  delay: i * 0.2
                }}
              />
            ))}
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}
