// ========================================
// VOICE INTEGRATION JAVASCRIPT
// ========================================

console.log('Loading voice integration...');

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing voice...');

    const voiceButton = document.getElementById('voiceButton');
    if (!voiceButton) {
        console.error('Voice button not found!');
        return;
    }

    // Voice Recording
    voiceButton.addEventListener('click', async () => {
        console.log('Voice button clicked, isRecording:', isRecording);
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    console.log('Voice integration initialized ✓');
});

async function startRecording() {
    console.log('Starting recording...');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log('Microphone access granted');

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            console.log('Audio data available:', event.data.size, 'bytes');
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            console.log('Recording stopped, processing...');
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            console.log('Audio blob created:', audioBlob.size, 'bytes');
            await transcribeAudio(audioBlob);
            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;

        const voiceBtn = document.getElementById('voiceButton');
        const recordingIndicator = document.getElementById('recordingIndicator');

        if (voiceBtn) voiceBtn.classList.add('recording');
        if (recordingIndicator) recordingIndicator.classList.add('show');

        console.log('Recording started ✓');
    } catch (error) {
        console.error('Microphone access denied:', error);
        alert('Please allow microphone access to use voice input.\n\nClick the lock icon in the address bar and allow microphone access.');
    }
}

function stopRecording() {
    console.log('Stopping recording...');
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        const voiceBtn = document.getElementById('voiceButton');
        const recordingIndicator = document.getElementById('recordingIndicator');

        if (voiceBtn) voiceBtn.classList.remove('recording');
        if (recordingIndicator) recordingIndicator.classList.remove('show');

        console.log('Recording stopped ✓');
    }
}

async function transcribeAudio(audioBlob) {
    console.log('Transcribing audio...');

    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        console.log('Sending to /api/voice/transcribe...');
        const response = await fetch('/api/voice/transcribe', {
            method: 'POST',
            body: formData
        });

        console.log('Transcription response:', response.status);
        const data = await response.json();
        console.log('Transcription data:', data);

        if (data.success && data.transcript) {
            console.log('Transcript:', data.transcript);

            // Auto-submit the transcribed query
            const messageInput = document.getElementById('messageInput');
            if (messageInput) {
                messageInput.value = data.transcript;
                console.log('Set message input to:', data.transcript);

                // Trigger send - try multiple methods
                const sendButton = document.getElementById('sendButton') ||
                                   document.querySelector('.send-button') ||
                                   document.querySelector('[onclick*="send"]');

                if (sendButton) {
                    console.log('Clicking send button');
                    sendButton.click();
                } else {
                    // Fallback: try pressing Enter
                    console.log('Send button not found, trying Enter key');
                    const event = new KeyboardEvent('keydown', {
                        key: 'Enter',
                        code: 'Enter',
                        keyCode: 13,
                        which: 13,
                        bubbles: true
                    });
                    messageInput.dispatchEvent(event);
                }
            } else {
                console.error('Message input not found!');
                alert('Could not find message input field. Transcript: ' + data.transcript);
            }
        } else {
            console.error('Transcription failed:', data);
            alert('Transcription failed. Please try again.');
        }
    } catch (error) {
        console.error('Transcription error:', error);
        alert('Error transcribing audio: ' + error.message);
    }
}

// Auto-speak response (conversational mode)
async function speakResponseAuto(responseText, confidence, mode) {
    console.log('Auto-speaking response...', {
        text: responseText.substring(0, 50) + '...',
        confidence,
        mode
    });

    try {
        const formData = new FormData();
        formData.append('response', responseText);
        formData.append('confidence', (confidence || 0.5).toString());
        formData.append('mode', mode || 'direct');

        console.log('Sending to /api/voice/speak_response...');
        const response = await fetch('/api/voice/speak_response', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            console.error('Auto-speak failed:', response.status, response.statusText);
            return;
        }

        console.log('Got audio response');
        const audioBlob = await response.blob();
        console.log('Audio blob size:', audioBlob.size);

        const audioUrl = URL.createObjectURL(audioBlob);

        const audio = document.getElementById('audioPlayer');
        if (audio) {
            audio.src = audioUrl;
            console.log('Playing audio...');
            audio.play().catch(err => {
                console.error('Audio playback failed:', err);
                alert('Could not play audio. Check browser console for details.');
            });
        } else {
            console.error('Audio player not found!');
        }
    } catch (error) {
        console.error('Auto-speak error:', error);
    }
}

console.log('Voice integration script loaded ✓');
