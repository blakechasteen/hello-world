#!/usr/bin/env python3
"""
🌊 REAL-TIME STREAMING NARRATIVE DEPTH
======================================
Progressive narrative analysis for streaming text input.

Features:
1. Incremental Complexity Scoring - Updates as text arrives
2. Progressive Gate Unlocking - Matryoshka gates open in real-time
3. Character Detection Timeline - Track when characters appear
4. Narrative Shift Detection - Identify dramatic turns
5. WebSocket-Ready - Designed for live streaming scenarios

Use Cases:
- Live chat analysis (Discord, Slack, Matrix)
- Reading together (book clubs, study groups)
- Real-time transcription (speeches, lectures, podcasts)
- Progressive document analysis (as you type)
- Live storytelling events

Architecture:
- Sliding window analysis (configurable chunk size)
- Momentum-based shift detection
- Incremental gate unlocking
- Event-driven progress updates
"""

import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

from hololoom_narrative.matryoshka_depth import MatryoshkaNarrativeDepth, DepthLevel
from hololoom_narrative.intelligence import CampbellStage


class StreamEventType(Enum):
    """Types of streaming events."""
    CHUNK_ADDED = "chunk_added"
    COMPLEXITY_UPDATE = "complexity_update"
    GATE_UNLOCKED = "gate_unlocked"
    CHARACTER_DETECTED = "character_detected"
    STAGE_TRANSITION = "stage_transition"
    NARRATIVE_SHIFT = "narrative_shift"
    ANALYSIS_COMPLETE = "analysis_complete"


@dataclass
class StreamEvent:
    """Event emitted during streaming analysis."""
    event_type: StreamEventType
    timestamp: float
    data: Dict[str, Any]
    cumulative_text_length: int


@dataclass
class StreamWindow:
    """Sliding window for incremental analysis."""
    chunks: deque = field(default_factory=lambda: deque(maxlen=10))
    total_length: int = 0
    last_analysis: Optional[Dict] = None
    unlocked_gates: set = field(default_factory=set)
    detected_characters: List[str] = field(default_factory=list)
    current_stage: Optional[CampbellStage] = None


class NarrativeShiftDetector:
    """Detects dramatic narrative shifts in streaming text."""
    
    def __init__(self, sensitivity: float = 0.3):
        self.sensitivity = sensitivity
        self.complexity_history = deque(maxlen=5)
        self.emotional_history = deque(maxlen=5)
    
    def update(self, complexity: float, emotional_intensity: float) -> bool:
        """
        Update with new metrics and detect if shift occurred.
        
        Returns:
            True if narrative shift detected
        """
        self.complexity_history.append(complexity)
        self.emotional_history.append(emotional_intensity)
        
        if len(self.complexity_history) < 3:
            return False
        
        # Calculate momentum (derivative)
        complexity_delta = self.complexity_history[-1] - self.complexity_history[-2]
        emotional_delta = self.emotional_history[-1] - self.emotional_history[-2]
        
        # Shift detected if sudden change in either dimension
        return (abs(complexity_delta) > self.sensitivity or 
                abs(emotional_delta) > self.sensitivity)


class StreamingNarrativeAnalyzer:
    """
    Real-time streaming narrative depth analyzer.
    
    Processes text incrementally, emitting events as narrative elements emerge.
    """
    
    def __init__(
        self,
        chunk_size: int = 100,
        update_interval: float = 0.5,
        enable_shift_detection: bool = True
    ):
        """
        Initialize streaming analyzer.
        
        Args:
            chunk_size: Characters per chunk (default 100 = ~20 words)
            update_interval: Seconds between analysis updates
            enable_shift_detection: Detect narrative shifts
        """
        self.chunk_size = chunk_size
        self.update_interval = update_interval
        self.enable_shift_detection = enable_shift_detection
        
        self.depth_analyzer = MatryoshkaNarrativeDepth()
        self.window = StreamWindow()
        self.shift_detector = NarrativeShiftDetector() if enable_shift_detection else None
        
        self.event_callbacks: List[Callable] = []
    
    def on_event(self, callback: Callable[[StreamEvent], None]):
        """Register callback for streaming events."""
        self.event_callbacks.append(callback)
    
    async def _emit_event(self, event: StreamEvent):
        """Emit event to all registered callbacks."""
        for callback in self.event_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
    
    async def stream_analyze(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[StreamEvent]:
        """
        Analyze text stream incrementally.
        
        Args:
            text_stream: Async iterator yielding text chunks
            
        Yields:
            StreamEvent objects as analysis progresses
        """
        buffer = ""
        last_update = time.time()
        
        async for incoming_text in text_stream:
            buffer += incoming_text
            self.window.total_length += len(incoming_text)
            
            # Emit chunk added event
            event = StreamEvent(
                event_type=StreamEventType.CHUNK_ADDED,
                timestamp=time.time(),
                data={'chunk_size': len(incoming_text)},
                cumulative_text_length=self.window.total_length
            )
            await self._emit_event(event)
            yield event
            
            # Check if we have enough for a chunk
            if len(buffer) >= self.chunk_size:
                # Extract chunk
                chunk = buffer[:self.chunk_size]
                buffer = buffer[self.chunk_size:]
                self.window.chunks.append(chunk)
                
                # Check if time for update
                now = time.time()
                if now - last_update >= self.update_interval:
                    # Analyze current window
                    async for event in self._analyze_window():
                        await self._emit_event(event)
                        yield event
                    last_update = now
        
        # Process remaining buffer
        if buffer:
            self.window.chunks.append(buffer)
        
        # Final analysis
        async for event in self._analyze_window(final=True):
            await self._emit_event(event)
            yield event
        
        # Emit completion
        completion_event = StreamEvent(
            event_type=StreamEventType.ANALYSIS_COMPLETE,
            timestamp=time.time(),
            data={
                'total_chunks': len(self.window.chunks),
                'total_length': self.window.total_length,
                'final_analysis': self.window.last_analysis
            },
            cumulative_text_length=self.window.total_length
        )
        await self._emit_event(completion_event)
        yield completion_event
    
    async def _analyze_window(self, final: bool = False) -> AsyncIterator[StreamEvent]:
        """Analyze current window and yield events."""
        # Combine chunks
        current_text = "".join(self.window.chunks)
        
        if not current_text.strip():
            return
        
        # Perform depth analysis
        depth_result = await self.depth_analyzer.analyze_depth(current_text)
        
        # Check for new gate unlocks
        new_gates = set()
        if depth_result.max_depth_achieved == DepthLevel.SYMBOLIC and DepthLevel.SYMBOLIC not in self.window.unlocked_gates:
            new_gates.add(DepthLevel.SYMBOLIC)
        if depth_result.max_depth_achieved == DepthLevel.ARCHETYPAL and DepthLevel.ARCHETYPAL not in self.window.unlocked_gates:
            new_gates.add(DepthLevel.ARCHETYPAL)
        if depth_result.max_depth_achieved == DepthLevel.MYTHIC and DepthLevel.MYTHIC not in self.window.unlocked_gates:
            new_gates.add(DepthLevel.MYTHIC)
        if depth_result.max_depth_achieved == DepthLevel.COSMIC and DepthLevel.COSMIC not in self.window.unlocked_gates:
            new_gates.add(DepthLevel.COSMIC)
        
        # Emit gate unlock events
        for gate in new_gates:
            self.window.unlocked_gates.add(gate)
            event = StreamEvent(
                event_type=StreamEventType.GATE_UNLOCKED,
                timestamp=time.time(),
                data={
                    'gate': gate.name,
                    'total_unlocked': len(self.window.unlocked_gates)
                },
                cumulative_text_length=self.window.total_length
            )
            yield event
        
        # Emit complexity update
        complexity_event = StreamEvent(
            event_type=StreamEventType.COMPLEXITY_UPDATE,
            timestamp=time.time(),
            data={
                'complexity': depth_result.total_complexity,
                'confidence': depth_result.bayesian_confidence,
                'max_depth': depth_result.max_depth_achieved.name
            },
            cumulative_text_length=self.window.total_length
        )
        yield complexity_event
        
        # Check for narrative shift
        if self.shift_detector:
            # Use confidence as emotional intensity proxy
            emotional_intensity = depth_result.bayesian_confidence
            if self.shift_detector.update(depth_result.total_complexity, emotional_intensity):
                shift_event = StreamEvent(
                    event_type=StreamEventType.NARRATIVE_SHIFT,
                    timestamp=time.time(),
                    data={
                        'complexity_delta': self.shift_detector.complexity_history[-1] - self.shift_detector.complexity_history[-2],
                        'description': 'Dramatic narrative turn detected'
                    },
                    cumulative_text_length=self.window.total_length
                )
                yield shift_event
        
        # Detect new characters (simple keyword-based for now)
        text_lower = current_text.lower()
        character_keywords = [
            ('Hero', ['hero', 'protagonist', 'main character']),
            ('Mentor', ['mentor', 'advisor', 'guide', 'teacher']),
            ('Shadow', ['enemy', 'villain', 'antagonist', 'rival']),
            ('Ally', ['friend', 'ally', 'helper', 'companion'])
        ]
        
        for char_name, keywords in character_keywords:
            if any(kw in text_lower for kw in keywords):
                if char_name not in self.window.detected_characters:
                    self.window.detected_characters.append(char_name)
                    char_event = StreamEvent(
                        event_type=StreamEventType.CHARACTER_DETECTED,
                        timestamp=time.time(),
                        data={
                            'character': char_name,
                            'archetype': char_name.lower()
                        },
                        cumulative_text_length=self.window.total_length
                    )
                    yield char_event
        
        # Store last analysis
        self.window.last_analysis = {
            'max_depth': depth_result.max_depth_achieved.name,
            'complexity': depth_result.total_complexity,
            'confidence': depth_result.bayesian_confidence,
            'deepest_meaning': depth_result.deepest_meaning
        }
    
    async def analyze_text_stream(
        self,
        text: str,
        words_per_second: float = 10.0
    ) -> AsyncIterator[StreamEvent]:
        """
        Simulate streaming analysis of complete text.
        
        Useful for testing and demos - simulates typing or reading speed.
        
        Args:
            text: Complete text to analyze
            words_per_second: Simulation speed (default 10 = comfortable reading)
            
        Yields:
            StreamEvent objects as if text was arriving in real-time
        """
        # Split into words
        words = text.split()
        
        async def word_stream():
            """Generate word stream at specified rate."""
            for word in words:
                yield word + " "
                await asyncio.sleep(1.0 / words_per_second)
        
        async for event in self.stream_analyze(word_stream()):
            yield event


async def demonstrate_streaming():
    """Demonstrate real-time streaming analysis."""
    print("🌊 REAL-TIME STREAMING NARRATIVE DEPTH")
    print("=" * 80)
    print()
    
    # Sample narrative (Odyssey-inspired business story)
    narrative = """
    Sarah sat in her corporate office, comfortable but unfulfilled. The call came from 
    an unexpected place - her own frustration with tools that didn't work. What if she 
    could build something better? The idea haunted her nights.
    
    At first, she refused. Too risky. Too hard. The mortgage, the stability, the fear. 
    But her mentor, a grizzled entrepreneur, asked one question: "What will you regret 
    more - trying and failing, or never knowing?" That night, Sarah wrote her resignation.
    
    The threshold crossed, there was no going back. Early days were tests - building the 
    MVP, finding the first customer, assembling a team. Some helped; others doubted. The 
    competition noticed, threatened, copied. But Sarah pressed on.
    
    Then came the ordeal. Runway down to three months. Key engineer quit. Largest prospect 
    went with competitor. Sarah faced the abyss - bankruptcy, failure, shame. In that 
    darkness, she found clarity: pivot or die. The painful decision to abandon months of 
    work and rebuild.
    
    The reward came slowly, then suddenly. One customer became ten, ten became a hundred. 
    Product-market fit felt like magic but was earned through iteration and pain. The 
    journey back meant scaling - hiring, systems, structure. Growing pains everywhere.
    
    The final test came with acquisition offers. Sell now for comfort, or push for impact? 
    Sarah chose impact. Today, thousands use her product. But the real gift wasn't success - 
    it was the transformation from employee to entrepreneur, from fear to courage, from 
    ordinary to extraordinary. She returns to mentor other founders, sharing the elixir 
    of hard-won wisdom: the startup journey transforms the founder before it transforms 
    the world.
    """
    
    # Create streaming analyzer
    analyzer = StreamingNarrativeAnalyzer(
        chunk_size=50,
        update_interval=1.0,
        enable_shift_detection=True
    )
    
    # Track events
    events_by_type = {t: [] for t in StreamEventType}
    
    # Setup event callback
    def print_event(event: StreamEvent):
        """Print event in real-time."""
        events_by_type[event.event_type].append(event)
        
        if event.event_type == StreamEventType.CHUNK_ADDED:
            # Silent for chunks
            pass
        elif event.event_type == StreamEventType.GATE_UNLOCKED:
            gate = event.data['gate']
            print(f"🔓 GATE UNLOCKED: {gate} (at {event.cumulative_text_length} chars)")
        elif event.event_type == StreamEventType.COMPLEXITY_UPDATE:
            complexity = event.data['complexity']
            depth = event.data['max_depth']
            print(f"📊 Complexity: {complexity:.3f} | Depth: {depth}")
        elif event.event_type == StreamEventType.CHARACTER_DETECTED:
            char = event.data['character']
            archetype = event.data.get('archetype', 'unknown')
            print(f"👤 Character: {char} ({archetype})")
        elif event.event_type == StreamEventType.NARRATIVE_SHIFT:
            print(f"⚡ NARRATIVE SHIFT: {event.data['description']}")
        elif event.event_type == StreamEventType.ANALYSIS_COMPLETE:
            print(f"✅ Analysis complete!")
    
    analyzer.on_event(print_event)
    
    print("📖 STARTING STREAMING ANALYSIS")
    print("   (Simulating reading at 15 words/second)")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Stream analyze
    event_count = 0
    async for event in analyzer.analyze_text_stream(narrative, words_per_second=15):
        event_count += 1
    
    duration = time.time() - start_time
    
    print()
    print("=" * 80)
    print("📈 STREAMING ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Duration: {duration:.1f}s")
    print(f"Total events: {event_count}")
    print(f"Text length: {len(narrative)} chars")
    print()
    
    print("Event Breakdown:")
    for event_type, events in events_by_type.items():
        if events:
            print(f"  {event_type.value:25} : {len(events):3} events")
    print()
    
    # Show gate unlock timeline
    gate_events = events_by_type[StreamEventType.GATE_UNLOCKED]
    if gate_events:
        print("🔓 Gate Unlock Timeline:")
        for event in gate_events:
            gate = event.data['gate']
            position = event.cumulative_text_length / len(narrative) * 100
            print(f"   {gate:15} unlocked at {position:5.1f}% of text")
        print()
    
    # Show character detection timeline
    char_events = events_by_type[StreamEventType.CHARACTER_DETECTED]
    if char_events:
        print("👥 Character Detection Timeline:")
        for event in char_events:
            char = event.data['character']
            position = event.cumulative_text_length / len(narrative) * 100
            print(f"   {char:15} appeared at {position:5.1f}% of text")
        print()
    
    # Show narrative shifts
    shift_events = events_by_type[StreamEventType.NARRATIVE_SHIFT]
    if shift_events:
        print("⚡ Narrative Shifts Detected:")
        for i, event in enumerate(shift_events, 1):
            position = event.cumulative_text_length / len(narrative) * 100
            print(f"   Shift {i}: {position:5.1f}% of text")
        print()
    
    # Final analysis
    final_event = events_by_type[StreamEventType.ANALYSIS_COMPLETE][0]
    final_analysis = final_event.data['final_analysis']
    print("🎯 Final Analysis:")
    print(f"   Max Depth: {final_analysis['max_depth']}")
    print(f"   Complexity: {final_analysis['complexity']:.3f}")
    print(f"   Confidence: {final_analysis['confidence']:.3f}")
    print(f"   Deepest Meaning: {final_analysis['deepest_meaning']}")
    print()
    
    print("=" * 80)
    print("🌊 USE CASES:")
    print("  • Live chat analysis (Discord, Slack, Matrix)")
    print("  • Reading together (book clubs, study groups)")
    print("  • Real-time transcription (speeches, podcasts)")
    print("  • Progressive document analysis (as you type)")
    print("  • Live storytelling events (theater, improv)")
    print("=" * 80)


async def demonstrate_websocket_integration():
    """Show how to integrate with WebSocket for live streaming."""
    print()
    print("🔌 WEBSOCKET INTEGRATION EXAMPLE")
    print("=" * 80)
    print()
    
    print("```python")
    print("# Server-side FastAPI + WebSocket")
    print("""
from fastapi import FastAPI, WebSocket
from hololoom_narrative.streaming_depth import StreamingNarrativeAnalyzer

app = FastAPI()

@app.websocket("/ws/narrative")
async def narrative_websocket(websocket: WebSocket):
    await websocket.accept()
    
    analyzer = StreamingNarrativeAnalyzer(
        chunk_size=50,
        update_interval=0.5
    )
    
    async def send_event(event):
        await websocket.send_json({
            'type': event.event_type.value,
            'timestamp': event.timestamp,
            'data': event.data
        })
    
    analyzer.on_event(send_event)
    
    async def text_stream():
        while True:
            data = await websocket.receive_text()
            if data == "END":
                break
            yield data
    
    async for event in analyzer.stream_analyze(text_stream()):
        pass  # Events sent via callback

# Client-side JavaScript
const ws = new WebSocket('ws://localhost:8000/ws/narrative');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'gate_unlocked') {
        console.log(`🔓 Gate ${data.data.gate} unlocked!`);
        updateUI(data);
    }
    else if (data.type === 'complexity_update') {
        updateComplexityChart(data.data.complexity);
    }
    else if (data.type === 'character_detected') {
        addCharacterToTimeline(data.data.character);
    }
};

// Send text as user types
textarea.addEventListener('input', (e) => {
    ws.send(e.target.value.slice(-10));  // Send last 10 chars
});
    """)
    print("```")
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demonstrate_streaming())
    asyncio.run(demonstrate_websocket_integration())
