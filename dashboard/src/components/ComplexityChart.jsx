import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { TrendingUp } from 'lucide-react'

export default function ComplexityChart({ data, shifts }) {
  if (!data || data.length === 0) {
    return (
      <motion.div 
        className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30 h-80"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-xl font-semibold text-orange-300 mb-4 flex items-center">
          <TrendingUp className="w-5 h-5 mr-2" />
          Complexity Evolution
        </h2>
        <div className="flex items-center justify-center h-48 text-gray-400">
          Start analysis to see complexity evolution
        </div>
      </motion.div>
    )
  }

  // Prepare data for chart
  const chartData = data.map((point, index) => ({
    index,
    complexity: (point.complexity * 100).toFixed(1),
    confidence: (point.confidence * 100).toFixed(1),
    depth: point.depth
  }))

  return (
    <motion.div 
      className="bg-black/70 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <h2 className="text-xl font-semibold text-orange-300 mb-4 flex items-center">
        <TrendingUp className="w-5 h-5 mr-2" />
        Complexity Evolution
      </h2>

      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={chartData}>
          <XAxis 
            dataKey="index" 
            stroke="#6B7280"
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
          />
          <YAxis 
            stroke="#6B7280"
            tick={{ fill: '#9CA3AF', fontSize: 12 }}
            domain={[0, 100]}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: '#1E293B',
              border: '1px solid #8B5CF6',
              borderRadius: '8px'
            }}
            labelStyle={{ color: '#A78BFA' }}
          />
          
          {/* Threshold lines */}
          <ReferenceLine y={30} stroke="#10B981" strokeDasharray="3 3" label="SYMBOLIC" />
          <ReferenceLine y={50} stroke="#F59E0B" strokeDasharray="3 3" label="ARCHETYPAL" />
          <ReferenceLine y={70} stroke="#EC4899" strokeDasharray="3 3" label="MYTHIC" />
          <ReferenceLine y={85} stroke="#8B5CF6" strokeDasharray="3 3" label="COSMIC" />
          
          <Line 
            type="monotone" 
            dataKey="complexity" 
            stroke="#8B5CF6" 
            strokeWidth={3}
            dot={{ fill: '#8B5CF6', r: 4 }}
            animationDuration={500}
          />
          <Line 
            type="monotone" 
            dataKey="confidence" 
            stroke="#EC4899" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ fill: '#EC4899', r: 3 }}
            animationDuration={500}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="mt-4 flex justify-around text-xs">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
          <span className="text-gray-300">Complexity</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-0.5 bg-pink-500"></div>
          <span className="text-gray-300">Confidence</span>
        </div>
        {shifts.length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-yellow-400">⚡</span>
            <span className="text-gray-300">{shifts.length} Narrative Shifts</span>
          </div>
        )}
      </div>
    </motion.div>
  )
}
