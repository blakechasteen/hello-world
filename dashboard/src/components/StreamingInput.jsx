import { useState } from 'react'
import { motion } from 'framer-motion'
import { Send, Zap } from 'lucide-react'

const EXAMPLE_TEXTS = {
  mythology: "Odysseus, guided by Athena, faced the Cyclops and overcame his pride. The journey home transformed him from warrior to wise king.",
  business: "Sarah quit her corporate job to build a startup. Her advisor warned: 'You'll pivot three times.' Months of failures followed, then one customer email changed everything.",
  science: "Dr. Chen's experiment contradicted 50 years of theory. Her PI dismissed it as error. But after three replications, they couldn't ignore the paradigm shift.",
  personal: "In therapy, I finally faced what I'd avoided for years. My inner critic screamed, but as I sat with discomfort, the wound became a doorway.",
  product: "User interviews revealed we'd solved the wrong problem. The team resisted redesigning. But we fell in love with the problem, and users fell in love with our solution.",
  history: "The protesters gathered despite warnings. Each crackdown spawned ten more protests. When the masses flooded the capital, the old order crumbled."
}

export default function StreamingInput({ onAnalyze, streaming, domain }) {
  const [text, setText] = useState('')
  const [streamMode, setStreamMode] = useState(true)

  const handleSubmit = () => {
    if (!text.trim()) return
    onAnalyze(text, streamMode)
  }

  const loadExample = () => {
    setText(EXAMPLE_TEXTS[domain] || EXAMPLE_TEXTS.mythology)
  }

  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-orange-300">
          Narrative Input
        </h2>
        
        <button
          onClick={loadExample}
          className="text-sm text-orange-400 hover:text-orange-300 transition-colors"
        >
          Load Example
        </button>
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={`Enter your ${domain} narrative here...`}
        className="w-full h-48 bg-black/50 border border-gray-700 rounded-lg p-4 text-gray-300 placeholder-gray-500 focus:outline-none focus:border-orange-500 resize-none"
        disabled={streaming}
      />

      <div className="mt-4 flex items-center justify-between">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={streamMode}
            onChange={(e) => setStreamMode(e.target.checked)}
            className="w-4 h-4 text-orange-500 rounded focus:ring-orange-500"
            disabled={streaming}
          />
          <span className="text-sm text-gray-300 flex items-center">
            <Zap className="w-4 h-4 mr-1 text-yellow-400" />
            Real-time Streaming
          </span>
        </label>

        <motion.button
          onClick={handleSubmit}
          disabled={streaming || !text.trim()}
          className={`
            px-6 py-2 rounded-lg font-medium flex items-center space-x-2
            ${streaming || !text.trim()
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-orange-600 text-white hover:bg-orange-700'
            }
            transition-colors
          `}
          whileHover={!streaming && text.trim() ? { scale: 1.05 } : {}}
          whileTap={!streaming && text.trim() ? { scale: 0.95 } : {}}
        >
          <Send className="w-4 h-4" />
          <span>Analyze</span>
        </motion.button>
      </div>

      <div className="mt-3 text-xs text-gray-400">
        {text.length} characters • {Math.ceil(text.split(' ').length / 5)} seconds streaming
      </div>
    </motion.div>
  )
}
