/**
 * Main App Component - Elle AR Client
 *
 * Root component managing AR session and UI overlays.
 *
 * Created: 2025-11-22
 */

import { useState, useCallback } from 'react'
import { Canvas } from '@react-three/fiber'
import { XR, Controllers, Hands } from '@react-three/xr'
import ARScene from './components/ARScene'
import VoiceUI from './components/VoiceUI'
import StatusBar from './components/StatusBar'
import { useElleConnection } from './hooks/useElleConnection'
import { useARSession } from './hooks/useARSession'

export default function App() {
  const [isARActive, setIsARActive] = useState(false)
  const [transcript, setTranscript] = useState<string>('')

  // Connect to Elle backend
  const {
    connected,
    sessionId,
    sendQuery,
    visualizations,
  } = useElleConnection('ws://localhost:8000/ws/ar')

  // AR session management
  const { context, updateContext } = useARSession()

  // Handle voice input
  const handleVoiceCommand = useCallback(async (text: string) => {
    setTranscript(text)

    if (connected && context) {
      await sendQuery(text, context)
    }
  }, [connected, context, sendQuery])

  // Handle AR session start
  const handleARStart = useCallback(() => {
    setIsARActive(true)
  }, [])

  // Handle AR session end
  const handleAREnd = useCallback(() => {
    setIsARActive(false)
  }, [])

  return (
    <div className="no-select">
      {/* Status Bar */}
      <StatusBar
        connected={connected}
        sessionId={sessionId}
        arActive={isARActive}
      />

      {/* 3D AR Canvas */}
      <Canvas
        camera={{ position: [0, 1.6, 0], fov: 75 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <XR onSessionStart={handleARStart} onSessionEnd={handleAREnd}>
          {/* Hand tracking (Quest 3+) */}
          <Hands />

          {/* Controller support */}
          <Controllers />

          {/* Main AR scene */}
          <ARScene
            visualizations={visualizations}
            onContextUpdate={updateContext}
          />
        </XR>
      </Canvas>

      {/* Voice UI Overlay */}
      <VoiceUI
        onVoiceCommand={handleVoiceCommand}
        transcript={transcript}
        isProcessing={false}
      />
    </div>
  )
}
