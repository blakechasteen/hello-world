import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RefreshCw, Zap, TrendingUp } from 'lucide-react'

export default function LoopControl({ onToggle, active, stats }) {
  const [mode, setMode] = useState('batch')
  const [rateLimit, setRateLimit] = useState(5)
  const [autoDetect, setAutoDetect] = useState(true)

  const modes = [
    { id: 'batch', name: 'Batch', desc: 'Process queue then stop', icon: Zap },
    { id: 'continuous', name: 'Continuous', desc: '24/7 processing', icon: RefreshCw },
    { id: 'scheduled', name: 'Scheduled', desc: 'Process at intervals', icon: TrendingUp }
  ]

  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-orange-300 flex items-center">
          <RefreshCw className="w-5 h-5 mr-2" />
          🔥 Loop Mode
        </h2>
        
        <motion.button
          onClick={() => onToggle()}
          className={`
            px-4 py-2 rounded-lg font-medium flex items-center space-x-2
            ${active 
              ? 'bg-orange-600 hover:bg-orange-700 text-white' 
              : 'bg-orange-600 hover:bg-orange-700 text-white'
            }
            transition-colors
          `}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          {active ? (
            <>
              <Pause className="w-4 h-4" />
              <span>Stop Loop</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              <span>Start Loop</span>
            </>
          )}
        </motion.button>
      </div>

      {/* Mode Selection */}
      <div className="mb-4">
        <label className="text-sm text-gray-300 mb-2 block">Processing Mode</label>
        <div className="grid grid-cols-3 gap-2">
          {modes.map(m => {
            const Icon = m.icon
            const isSelected = mode === m.id
            
            return (
              <motion.button
                key={m.id}
                onClick={() => !active && setMode(m.id)}
                disabled={active}
                className={`
                  p-3 rounded-lg border transition-all
                  ${isSelected 
                    ? 'border-orange-500 bg-orange-500/20' 
                    : 'border-gray-700 bg-black/50 hover:border-orange-400'
                  }
                  ${active ? 'opacity-50 cursor-not-allowed' : ''}
                `}
                whileHover={!active ? { scale: 1.05 } : {}}
                whileTap={!active ? { scale: 0.95 } : {}}
              >
                <Icon className={`w-5 h-5 mx-auto mb-1 ${isSelected ? 'text-orange-400' : 'text-gray-400'}`} />
                <div className={`text-xs font-medium ${isSelected ? 'text-orange-300' : 'text-gray-400'}`}>
                  {m.name}
                </div>
              </motion.button>
            )
          })}
        </div>
        <div className="text-xs text-gray-400 mt-2">
          {modes.find(m => m.id === mode)?.desc}
        </div>
      </div>

      {/* Rate Limit */}
      <div className="mb-4">
        <label className="text-sm text-gray-300 mb-2 block flex items-center justify-between">
          <span>Rate Limit</span>
          <span className="text-orange-400 font-mono">{rateLimit} tasks/sec</span>
        </label>
        <input
          type="range"
          min="1"
          max="20"
          value={rateLimit}
          onChange={(e) => !active && setRateLimit(parseInt(e.target.value))}
          disabled={active}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
          style={{
            background: `linear-gradient(to right, #8B5CF6 0%, #8B5CF6 ${rateLimit * 5}%, #374151 ${rateLimit * 5}%, #374151 100%)`
          }}
        />
      </div>

      {/* Auto-Detect Toggle */}
      <div className="mb-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={autoDetect}
            onChange={(e) => !active && setAutoDetect(e.target.checked)}
            disabled={active}
            className="w-4 h-4 text-orange-500 rounded focus:ring-orange-500"
          />
          <span className="text-sm text-gray-300">
            Auto-detect domains (learning mode)
          </span>
        </label>
      </div>

      {/* Active Status */}
      <AnimatePresence>
        {active && (
          <motion.div 
            className="p-3 bg-orange-500/10 border border-orange-500/30 rounded-lg"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="flex items-center space-x-2 mb-2">
              <motion.div
                className="w-3 h-3 bg-green-400 rounded-full"
                animate={{ opacity: [1, 0.3, 1] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              />
              <span className="text-sm font-semibold text-orange-300">Loop Active</span>
            </div>
            
            {stats && (
              <div className="space-y-1 text-xs text-gray-300">
                <div className="flex justify-between">
                  <span>Processed:</span>
                  <span className="text-orange-300 font-mono">{stats.processed || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Queued:</span>
                  <span className="text-orange-300 font-mono">{stats.queued || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span>Rate:</span>
                  <span className="text-orange-300 font-mono">{stats.rate?.toFixed(1) || '0.0'} tasks/sec</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Time:</span>
                  <span className="text-orange-300 font-mono">{stats.avgTime?.toFixed(0) || '0'}ms</span>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info */}
      {!active && (
        <div className="mt-4 text-xs text-gray-400 bg-black/50 p-3 rounded-lg">
          <p className="mb-1">💡 <strong>Loop Mode</strong> enables continuous processing:</p>
          <ul className="ml-4 space-y-1 list-disc">
            <li><strong>Batch:</strong> Process all queued items then stop</li>
            <li><strong>Continuous:</strong> Run 24/7, process as items arrive</li>
            <li><strong>Scheduled:</strong> Process at regular intervals</li>
          </ul>
        </div>
      )}
    </motion.div>
  )
}
