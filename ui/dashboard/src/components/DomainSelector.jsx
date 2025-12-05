import { motion } from 'framer-motion'

export default function DomainSelector({ domains, selected, onChange }) {
  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h2 className="text-xl font-semibold text-orange-300 mb-4">
        🔥 Select Domain
      </h2>
      
      <div className="grid grid-cols-2 gap-3">
        {domains.map(domain => {
          const Icon = domain.icon
          const isSelected = selected === domain.id
          
          return (
            <motion.button
              key={domain.id}
              onClick={() => onChange(domain.id)}
              className={`
                relative p-4 rounded-lg border-2 transition-all
                ${isSelected 
                  ? 'border-orange-500 bg-orange-500/20' 
                  : 'border-gray-800 bg-black/50 hover:border-orange-400'
                }
              `}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="flex flex-col items-center space-y-2">
                <Icon 
                  className="w-6 h-6" 
                  style={{ color: isSelected ? domain.color : '#9CA3AF' }}
                />
                <span className={`text-sm font-medium ${isSelected ? 'text-orange-300' : 'text-gray-300'}`}>
                  {domain.name}
                </span>
              </div>
              
              {isSelected && (
                <motion.div
                  className="absolute inset-0 rounded-lg"
                  style={{ 
                    boxShadow: `0 0 20px ${domain.color}40` 
                  }}
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
              )}
            </motion.button>
          )
        })}
      </div>
    </motion.div>
  )
}
