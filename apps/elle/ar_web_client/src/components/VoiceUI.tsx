/**
 * VoiceUI Component - Voice input interface with Web Speech API
 *
 * Features:
 * - Push-to-talk voice recording
 * - Real-time transcript display
 * - Visual feedback
 * - Automatic intent classification (future)
 *
 * Created: 2025-11-22
 */

import { useState, useCallback, useRef, useEffect } from 'react'

interface VoiceUIProps {
  onVoiceCommand: (text: string) => void
  transcript: string
  isProcessing: boolean
}

export default function VoiceUI({
  onVoiceCommand,
  transcript,
  isProcessing,
}: VoiceUIProps) {
  const [isListening, setIsListening] = useState(false)
  const [interimTranscript, setInterimTranscript] = useState('')

  const recognitionRef = useRef<any>(null)

  // Initialize Web Speech API
  useEffect(() => {
    // Check for browser support
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition

    if (!SpeechRecognition) {
      console.error('Speech recognition not supported')
      return
    }

    // Create recognition instance
    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => {
      setIsListening(true)
      setInterimTranscript('')
    }

    recognition.onresult = (event: any) => {
      let interim = ''
      let final = ''

      for (let i = 0; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript

        if (event.results[i].isFinal) {
          final += transcript
        } else {
          interim += transcript
        }
      }

      setInterimTranscript(interim)

      if (final) {
        onVoiceCommand(final)
        setInterimTranscript('')
      }
    }

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error)
      setIsListening(false)
    }

    recognition.onend = () => {
      setIsListening(false)
    }

    recognitionRef.current = recognition

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort()
      }
    }
  }, [onVoiceCommand])

  // Toggle listening
  const toggleListening = useCallback(() => {
    if (!recognitionRef.current) {
      console.error('Speech recognition not initialized')
      return
    }

    if (isListening) {
      recognitionRef.current.stop()
    } else {
      try {
        recognitionRef.current.start()
      } catch (error) {
        console.error('Failed to start recognition:', error)
      }
    }
  }, [isListening])

  return (
    <div className="ui-overlay">
      {/* Voice button */}
      <button
        className={`voice-button ${isListening ? 'active' : ''}`}
        onClick={toggleListening}
        disabled={isProcessing}
        aria-label={isListening ? 'Stop listening' : 'Start listening'}
      >
        {isListening ? (
          <svg width="32" height="32" viewBox="0 0 24 24" fill="white">
            <rect x="6" y="6" width="12" height="12" />
          </svg>
        ) : (
          <svg width="32" height="32" viewBox="0 0 24 24" fill="white">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
          </svg>
        )}
      </button>

      {/* Transcript display */}
      {(transcript || interimTranscript || isProcessing) && (
        <div className="transcript">
          {transcript && (
            <p className="elle">
              <strong>Elle:</strong> {transcript}
            </p>
          )}
          {interimTranscript && (
            <p className="user" style={{ opacity: 0.7 }}>
              <strong>You:</strong> {interimTranscript}
            </p>
          )}
          {isProcessing && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div className="spinner" style={{ width: '16px', height: '16px' }} />
              <span style={{ fontSize: '0.875rem', color: '#999' }}>Processing...</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
