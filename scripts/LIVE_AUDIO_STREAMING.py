"""
LIVE AUDIO STREAMING PIPELINE
==============================

Extend the bee inspection pipeline for REAL-TIME audio processing.

THREE APPROACHES TO LIVE AUDIO
===============================

Approach 1: Stream → Chunk → Process (Low Latency)
Approach 2: Continuous Recording → Periodic Analysis (High Accuracy)
Approach 3: Hybrid (Real-time + Post-processing)


IMPLEMENTATION
==============
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, List
from datetime import datetime
import queue
import threading

# For audio capture
try:
    import pyaudio
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  Install: pip install pyaudio sounddevice")

# Reuse existing components
from bee_inspection_standalone import (
    AudioTranscriber,
    LLMExtractor, 
    SimpleEmbedder,
    Neo4jStorage
)


@dataclass
class StreamChunk:
    """Audio chunk from live stream"""
    audio_data: np.ndarray
    timestamp: float
    chunk_id: int
    sample_rate: int = 16000


class LiveAudioCapture:
    """Capture live audio from microphone"""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 3.0):
        """
        Args:
            sample_rate: Audio sample rate (16kHz for Whisper)
            chunk_duration: Seconds per chunk (3-10s recommended)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def _audio_callback(self, indata, frames, time, status):
        """Called for each audio block"""
        if status:
            print(f"⚠️  Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    async def stream_chunks(self) -> AsyncGenerator[StreamChunk, None]:
        """Stream audio chunks asynchronously"""
        
        if not AUDIO_AVAILABLE:
            raise RuntimeError("Audio libraries not available")
        
        chunk_id = 0
        buffer = []
        
        print(f"🎤 Starting live capture ({self.sample_rate}Hz, {self.chunk_duration}s chunks)")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        ):
            self.is_recording = True
            
            while self.is_recording:
                try:
                    # Get audio block from queue (non-blocking)
                    audio_block = await asyncio.get_event_loop().run_in_executor(
                        None, self.audio_queue.get, True, 0.1
                    )
                    buffer.append(audio_block)
                    
                    # Check if we have enough for a chunk
                    buffer_samples = sum(len(b) for b in buffer)
                    if buffer_samples >= self.chunk_samples:
                        # Concatenate and yield chunk
                        chunk_audio = np.concatenate(buffer)[:self.chunk_samples]
                        buffer = []
                        
                        yield StreamChunk(
                            audio_data=chunk_audio,
                            timestamp=datetime.now().timestamp(),
                            chunk_id=chunk_id,
                            sample_rate=self.sample_rate
                        )
                        chunk_id += 1
                        
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"❌ Audio error: {e}")
                    break
    
    def stop(self):
        """Stop recording"""
        self.is_recording = False


class LiveTranscriber:
    """Real-time transcription with Whisper"""
    
    def __init__(self, model_size: str = "base"):
        self.transcriber = AudioTranscriber(model_size=model_size)
        self.context_window = []  # Keep recent chunks for context
        self.max_context = 3  # Remember last 3 chunks
        
    async def transcribe_chunk(self, chunk: StreamChunk) -> dict:
        """Transcribe single chunk with context"""
        
        # Whisper expects 16kHz audio
        if chunk.sample_rate != 16000:
            # Resample if needed (simplified)
            pass
        
        # Transcribe this chunk
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._transcribe_sync,
            chunk.audio_data
        )
        
        # Add to context window
        self.context_window.append(result)
        if len(self.context_window) > self.max_context:
            self.context_window.pop(0)
        
        return {
            'chunk_id': chunk.chunk_id,
            'timestamp': chunk.timestamp,
            'text': result['text'],
            'segments': result.get('segments', []),
            'context': ' '.join(c['text'] for c in self.context_window)
        }
    
    def _transcribe_sync(self, audio_data: np.ndarray) -> dict:
        """Synchronous transcription (runs in executor)"""
        # Flatten if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()
        
        # Normalize
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0
        
        # Transcribe with Whisper
        result = self.transcriber.model.transcribe(audio_data)
        return result


class LiveEventDetector:
    """Detect events in live transcription stream"""
    
    def __init__(self, event_keywords: List[str], buffer_size: int = 10):
        """
        Args:
            event_keywords: Words/phrases that trigger event detection
            buffer_size: Number of recent chunks to keep
        """
        self.keywords = [kw.lower() for kw in event_keywords]
        self.transcript_buffer = []
        self.buffer_size = buffer_size
        
    def check_for_event(self, transcript_chunk: dict) -> Optional[dict]:
        """Check if chunk contains event trigger"""
        
        text = transcript_chunk['text'].lower()
        self.transcript_buffer.append(transcript_chunk)
        
        # Keep buffer size limited
        if len(self.transcript_buffer) > self.buffer_size:
            self.transcript_buffer.pop(0)
        
        # Check for keywords
        triggered_keywords = [kw for kw in self.keywords if kw in text]
        
        if triggered_keywords:
            # Event detected! Combine recent context
            context = ' '.join(c['text'] for c in self.transcript_buffer)
            
            return {
                'event_type': 'keyword_match',
                'keywords': triggered_keywords,
                'trigger_chunk': transcript_chunk['chunk_id'],
                'timestamp': transcript_chunk['timestamp'],
                'context': context,
                'full_buffer': self.transcript_buffer.copy()
            }
        
        return None


class LiveBeeInspectionPipeline:
    """Real-time bee inspection audio processing"""
    
    def __init__(
        self,
        use_llm: bool = True,
        use_neo4j: bool = False,
        event_keywords: Optional[List[str]] = None
    ):
        """
        Args:
            use_llm: Use LLM for real-time extraction
            use_neo4j: Store events in Neo4j immediately
            event_keywords: Keywords that trigger event processing
        """
        
        # Core components (reused from batch pipeline!)
        self.capturer = LiveAudioCapture(chunk_duration=5.0)
        self.transcriber = LiveTranscriber(model_size="base")
        self.embedder = SimpleEmbedder() if use_llm else None
        self.extractor = LLMExtractor() if use_llm else None
        self.storage = Neo4jStorage() if use_neo4j else None
        
        # Live-specific components
        self.event_detector = LiveEventDetector(
            event_keywords=event_keywords or [
                "queen", "swarm", "aggressive", "disease",
                "dead bees", "problem", "issue", "concern"
            ]
        )
        
        # State
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events_detected = []
        self.full_transcript = []
        
    async def process_live(self, duration_seconds: Optional[float] = None):
        """
        Process live audio stream
        
        Args:
            duration_seconds: How long to record (None = infinite)
        """
        
        print(f"🐝 Live Bee Inspection Session: {self.session_id}")
        print(f"🎤 Recording live audio...")
        print(f"📝 Will detect events with keywords: {self.event_detector.keywords}")
        print(f"⏹️  Press Ctrl+C to stop\n")
        
        start_time = datetime.now()
        
        try:
            # Start streaming audio chunks
            async for chunk in self.capturer.stream_chunks():
                
                # Check duration limit
                if duration_seconds:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > duration_seconds:
                        print(f"⏰ Reached {duration_seconds}s limit")
                        break
                
                # Step 1: Transcribe chunk
                print(f"🎤 Chunk {chunk.chunk_id}: ", end='', flush=True)
                transcript = await self.transcriber.transcribe_chunk(chunk)
                print(f"'{transcript['text'][:60]}...'")
                
                self.full_transcript.append(transcript)
                
                # Step 2: Check for events
                event = self.event_detector.check_for_event(transcript)
                
                if event:
                    print(f"\n🚨 EVENT DETECTED!")
                    print(f"   Keywords: {event['keywords']}")
                    print(f"   Context: {event['context'][:100]}...")
                    
                    # Step 3: Extract structured data (for events only)
                    if self.extractor:
                        extracted = await self._extract_event(event)
                        
                        # Step 4: Generate embeddings
                        if self.embedder:
                            embedding = self.embedder.encode([event['context']])[0]
                            extracted['embedding'] = embedding.tolist()
                        
                        # Step 5: Store immediately
                        if self.storage:
                            self.storage.store_event(extracted)
                            print(f"   ✅ Stored to Neo4j")
                        
                        self.events_detected.append(extracted)
                    
                    print()  # Blank line after event
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Stopped by user")
        finally:
            self.capturer.stop()
            await self._finalize_session()
    
    async def _extract_event(self, event: dict) -> dict:
        """Extract structured data from detected event"""
        
        # Use LLM extractor with event context
        result = await self.extractor.extract_async(
            transcript=event['context'],
            date=datetime.fromtimestamp(event['timestamp']).strftime('%Y-%m-%d')
        )
        
        # Add event metadata
        result['event_id'] = f"{self.session_id}_{event['trigger_chunk']}"
        result['trigger_keywords'] = event['keywords']
        result['timestamp'] = event['timestamp']
        
        return result
    
    async def _finalize_session(self):
        """Process full session after recording ends"""
        
        print(f"\n{'='*70}")
        print(f"SESSION SUMMARY: {self.session_id}")
        print(f"{'='*70}")
        print(f"📝 Total chunks: {len(self.full_transcript)}")
        print(f"🚨 Events detected: {len(self.events_detected)}")
        print(f"⏱️  Duration: {len(self.full_transcript) * 5}s")
        
        if self.events_detected:
            print(f"\nDetected Events:")
            for i, event in enumerate(self.events_detected, 1):
                print(f"  {i}. {event.get('trigger_keywords', [])} at chunk {event.get('event_id', 'N/A')}")
        
        # Option: Process full transcript as batch
        full_text = ' '.join(t['text'] for t in self.full_transcript)
        print(f"\n📄 Full transcript length: {len(full_text)} chars")
        
        # Save transcript
        transcript_file = f"live_session_{self.session_id}.txt"
        with open(transcript_file, 'w') as f:
            f.write(f"Live Bee Inspection Session: {self.session_id}\n")
            f.write(f"{'='*70}\n\n")
            for t in self.full_transcript:
                f.write(f"[Chunk {t['chunk_id']}] {t['text']}\n")
        
        print(f"💾 Saved transcript: {transcript_file}")
        print(f"{'='*70}\n")


"""
USAGE EXAMPLES
==============
"""

async def example_1_simple_live():
    """Example 1: Simple live transcription"""
    
    pipeline = LiveBeeInspectionPipeline(
        use_llm=False,  # No LLM, just transcribe
        use_neo4j=False  # No storage, just print
    )
    
    # Record for 60 seconds
    await pipeline.process_live(duration_seconds=60)


async def example_2_full_pipeline():
    """Example 2: Full pipeline with LLM + Neo4j"""
    
    pipeline = LiveBeeInspectionPipeline(
        use_llm=True,  # Extract schema on events
        use_neo4j=True,  # Store to graph immediately
        event_keywords=[
            "queen", "swarm", "aggressive", "disease",
            "dead bees", "problem", "varroa", "mite"
        ]
    )
    
    # Record until Ctrl+C
    await pipeline.process_live()


async def example_3_custom_events():
    """Example 3: Custom event detection"""
    
    pipeline = LiveBeeInspectionPipeline(
        use_llm=True,
        event_keywords=[
            # Critical events
            "queen lost", "swarming", "emergency",
            # Health indicators
            "foulbrood", "nosema", "varroa",
            # Behavior patterns
            "aggressive", "defensive", "robbing"
        ]
    )
    
    await pipeline.process_live(duration_seconds=300)  # 5 minutes


"""
ADVANCED: WEBSOCKET STREAMING
==============================

For web interfaces or remote monitoring:
"""

class WebSocketStreamer:
    """Stream live audio to web client"""
    
    def __init__(self, websocket):
        self.ws = websocket
        self.pipeline = LiveBeeInspectionPipeline()
    
    async def stream_to_client(self):
        """Send real-time updates to web client"""
        
        async for chunk in self.pipeline.capturer.stream_chunks():
            # Transcribe
            transcript = await self.pipeline.transcriber.transcribe_chunk(chunk)
            
            # Send to client
            await self.ws.send_json({
                'type': 'transcript',
                'chunk_id': chunk.chunk_id,
                'text': transcript['text'],
                'timestamp': transcript['timestamp']
            })
            
            # Check for events
            event = self.pipeline.event_detector.check_for_event(transcript)
            if event:
                await self.ws.send_json({
                    'type': 'event',
                    'keywords': event['keywords'],
                    'context': event['context']
                })


"""
PERFORMANCE CONSIDERATIONS
===========================

Latency Budget (per 5s chunk):
- Audio capture: ~0ms (streaming)
- Whisper base: ~500ms (CPU) or ~100ms (GPU)
- LLM extraction: ~1-2s (only for events)
- Embedding: ~50ms
- Neo4j write: ~10ms
Total: ~2.5s processing for 5s audio = OK!

Optimization strategies:
1. Use Whisper "tiny" for lowest latency (~200ms)
2. Only run LLM on detected events (not every chunk)
3. Use GPU if available (5-10x faster)
4. Buffer audio locally, process async
5. Batch embed multiple events together
"""


"""
REAL-WORLD DEPLOYMENT
=====================

Scenario 1: Field Inspection with Mobile
-----------------------------------------
- Beekeeper uses phone/tablet
- App records live during inspection
- Events detected in real-time
- Full transcript saved for later
- Critical events trigger alerts

Scenario 2: Remote Monitoring
------------------------------
- Microphone in bee yard
- Continuous recording
- Detect sounds: swarming, distress, robbing
- Send alerts to beekeeper
- Historical analysis of audio patterns

Scenario 3: Teaching/Training
------------------------------
- Record experienced beekeeper
- Real-time subtitles for students
- Highlight important observations
- Build training dataset
- Create annotated examples

Scenario 4: Multi-Hive Operation
---------------------------------
- Multiple inspectors with devices
- Central server processes all streams
- Live dashboard shows all activity
- Coordinator monitors for issues
- Aggregate statistics in real-time
"""


"""
COMPARISON: LIVE vs BATCH
==========================

LIVE (this file):
✅ Real-time feedback
✅ Immediate event detection
✅ Lower memory (streaming)
✅ Can stop/alert during recording
❌ Higher complexity
❌ Network dependency (if remote)

BATCH (bee_inspection_standalone.py):
✅ Higher accuracy (full context)
✅ Simpler implementation
✅ Can use larger models
✅ Better for analysis
❌ No real-time feedback
❌ Higher memory (full file)

RECOMMENDATION: Use both!
- Live: During inspection for real-time guidance
- Batch: After inspection for thorough analysis
"""


if __name__ == "__main__":
    print(__doc__)
    
    print("\n" + "="*70)
    print("LIVE AUDIO STREAMING READY!")
    print("="*70)
    print("\nInstallation:")
    print("  pip install pyaudio sounddevice")
    print("\nQuick Start:")
    print("  python3 LIVE_AUDIO_STREAMING.py")
    print("\nOr in code:")
    print("  await LiveBeeInspectionPipeline().process_live()")
    print("\n" + "="*70)
    
    # Uncomment to run:
    # asyncio.run(example_2_full_pipeline())
