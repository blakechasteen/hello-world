#!/usr/bin/env python3
"""
Script to add voice integration to agentic_dashboard.html

Adds:
1. Voice CSS styles
2. Voice HTML elements (mic button, recording indicator, audio player)
3. Voice JavaScript (recording, transcription, auto-speak)

Usage:
    python add_voice_to_dashboard.py
"""

import re
from pathlib import Path

# Voice CSS to add before </style>
VOICE_CSS = """
        /* ========================================
           VOICE INTEGRATION STYLES
           ======================================== */

        /* Voice Button (Floating Action Button) */
        .voice-button {
            position: fixed;
            bottom: 90px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            border: none;
            color: #fff;
            font-size: 1.5em;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(255, 107, 107, 0.4);
            transition: all 0.3s;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .voice-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
        }

        .voice-button.recording {
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            animation: voicePulse 1.5s infinite;
        }

        @keyframes voicePulse {
            0%, 100% {
                transform: scale(1);
                box-shadow: 0 4px 12px rgba(255, 68, 68, 0.4);
            }
            50% {
                transform: scale(1.05);
                box-shadow: 0 8px 24px rgba(255, 68, 68, 0.8);
            }
        }

        /* Recording Indicator */
        .recording-indicator {
            position: fixed;
            bottom: 160px;
            right: 30px;
            background: rgba(255, 68, 68, 0.95);
            color: white;
            padding: 12px 20px;
            border-radius: 24px;
            font-size: 0.9em;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(255, 68, 68, 0.4);
            display: none;
            align-items: center;
            gap: 8px;
            z-index: 999;
        }

        .recording-indicator.show {
            display: flex;
            animation: slideIn 0.3s ease-out;
        }

        .recording-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: white;
            animation: blink 1s infinite;
        }

        @keyframes blink {
            0%, 50%, 100% { opacity: 1; }
            25%, 75% { opacity: 0.3; }
        }

        @keyframes slideIn {
            from {
                transform: translateY(10px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        /* Speak Response Button */
        .speak-button {
            display: inline-block;
            margin-left: 8px;
            padding: 4px 10px;
            background: rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            color: #667eea;
            font-size: 0.75em;
            cursor: pointer;
            transition: all 0.2s;
        }

        .speak-button:hover {
            background: rgba(102, 126, 234, 0.2);
            transform: translateY(-1px);
        }

        .speak-button.playing {
            background: rgba(102, 126, 234, 0.3);
            color: #fff;
        }

        /* Audio Player (hidden) */
        #audioPlayer {
            display: none;
        }
"""

# Voice HTML to add before </body>
VOICE_HTML = """
    <!-- ========================================
         VOICE INTEGRATION UI
         ======================================== -->

    <!-- Voice Button (Floating Action Button) -->
    <button class="voice-button" id="voiceButton" title="Click to record voice">
        🎤
    </button>

    <!-- Recording Indicator -->
    <div class="recording-indicator" id="recordingIndicator">
        <div class="recording-dot"></div>
        <span>Recording...</span>
    </div>

    <!-- Hidden Audio Player for auto-speak -->
    <audio id="audioPlayer"></audio>
"""

# Voice JavaScript to add in script section (simplified for integration)
VOICE_JS = """
        // ========================================
        // VOICE INTEGRATION JAVASCRIPT
        // ========================================

        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];

        // Voice Recording
        document.getElementById('voiceButton')?.addEventListener('click', async () => {
            if (isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        });

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await transcribeAudio(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                isRecording = true;

                document.getElementById('voiceButton').classList.add('recording');
                document.getElementById('recordingIndicator').classList.add('show');

                console.log('Recording started...');
            } catch (error) {
                console.error('Microphone access denied:', error);
                alert('Please allow microphone access to use voice input');
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;

                document.getElementById('voiceButton').classList.remove('recording');
                document.getElementById('recordingIndicator').classList.remove('show');

                console.log('Recording stopped');
            }
        }

        async function transcribeAudio(audioBlob) {
            console.log('Transcribing audio...');

            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');

                const response = await fetch('/api/voice/transcribe', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success && data.transcript) {
                    console.log('Transcript:', data.transcript);

                    // Auto-submit the transcribed query
                    const messageInput = document.getElementById('messageInput');
                    if (messageInput) {
                        messageInput.value = data.transcript;

                        // Trigger send
                        const sendButton = document.getElementById('sendButton') || document.querySelector('.send-button');
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                } else {
                    console.error('Transcription failed:', data);
                }
            } catch (error) {
                console.error('Transcription error:', error);
            }
        }

        // Auto-speak response (conversational mode)
        async function speakResponseAuto(responseText, confidence, mode) {
            console.log('Auto-speaking response...', { confidence, mode });

            try {
                const formData = new FormData();
                formData.append('response', responseText);
                formData.append('confidence', (confidence || 0.5).toString());
                formData.append('mode', mode || 'direct');

                const response = await fetch('/api/voice/speak_response', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    console.error('Auto-speak failed:', response.statusText);
                    return;
                }

                const audioBlob = await response.blob();
                const audioUrl = URL.createObjectURL(audioBlob);

                const audio = document.getElementById('audioPlayer');
                if (audio) {
                    audio.src = audioUrl;
                    audio.play().catch(err => {
                        console.error('Audio playback failed:', err);
                    });
                }
            } catch (error) {
                console.error('Auto-speak error:', error);
            }
        }

        console.log('Voice integration loaded ✓');
"""

def add_voice_integration():
    """Add voice integration to agentic_dashboard.html"""

    dashboard_path = Path(__file__).parent / "agentic_dashboard.html"

    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False

    print(f"Reading {dashboard_path}...")
    html_content = dashboard_path.read_text(encoding='utf-8')

    # Check if already added
    if '<!-- VOICE INTEGRATION UI -->' in html_content or 'VOICE INTEGRATION STYLES' in html_content:
        print("⚠️  Voice integration already added!")
        return False

    # 1. Add CSS before </style>
    print("Adding voice CSS...")
    html_content = html_content.replace('    </style>', VOICE_CSS + '\n    </style>', 1)

    # 2. Add HTML before </body>
    print("Adding voice HTML...")
    html_content = html_content.replace('</body>', VOICE_HTML + '\n</body>', 1)

    # 3. Add JavaScript before closing </script>
    print("Adding voice JavaScript...")
    # Find the last </script> tag
    script_pattern = r'(</script>\s*</body>)'
    html_content = re.sub(script_pattern, VOICE_JS + '\n    \\1', html_content, count=1)

    # Write updated file
    print(f"Writing updated dashboard...")
    dashboard_path.write_text(html_content, encoding='utf-8')

    print("✅ Voice integration added successfully!")
    print()
    print("Next steps:")
    print("1. Server is already running with voice backend")
    print("2. Refresh http://localhost:8002 in your browser")
    print("3. Click the 🎤 button to record voice")
    print("4. Responses will speak automatically!")
    print()
    print("Note: Update WebSocket handler to trigger speakResponseAuto() for full auto-speak")

    return True


if __name__ == '__main__':
    add_voice_integration()
