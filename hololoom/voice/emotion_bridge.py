"""
Emotion Bridge - Python ↔ Node.js Integration
==============================================

Integrates the JavaScript 110/100 Emotional Intelligence System with HoloLoom's Python ecosystem.

Architecture:
- Spawns Node.js child process running complete_emotional_pipeline.js
- Provides Python API wrapping JS emotional intelligence features
- Handles bidirectional JSON-RPC communication
- Integrates with voice_agent.py to add emotional context

Date: November 2025
"""

import asyncio
import json
import subprocess
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
    logger = structlog.get_logger()
except ImportError:
    STRUCTLOG_AVAILABLE = False
    logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class FusionStrategy(Enum):
    """Emotion fusion strategies"""
    AVERAGE = "average"
    WEIGHTED = "weighted"
    BAYESIAN = "bayesian"
    DEMPSTER_SHAFER = "dempster_shafer"
    NEURAL_FUSION = "neural_fusion"
    CONTEXT_AWARE = "context_aware"


class MetaMode(Enum):
    """Meta-interpretation modes"""
    DISABLED = "disabled"
    NUANCE_INTERPRETATION = "nuance_interpretation"
    FULL_META = "full_meta"


class PlanningStrategy(Enum):
    """Action planning strategies"""
    DISABLED = "disabled"
    IMMEDIATE = "immediate"
    ADAPTIVE = "adaptive"


@dataclass
class EmotionBridgeConfig:
    """Configuration for emotion bridge"""

    # Node.js path
    node_executable: str = "node"
    pipeline_script: Optional[str] = None  # Auto-detect if None

    # Signal detection
    enable_facial: bool = True
    enable_vocal: bool = True
    enable_text: bool = True

    # Fusion
    fusion_strategy: FusionStrategy = FusionStrategy.BAYESIAN
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        'facial': 0.4,
        'vocal': 0.35,
        'text': 0.25
    })

    # Meta-interpretation
    enable_meta_interpretation: bool = True
    meta_mode: MetaMode = MetaMode.FULL_META

    # LLM integration
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_api_key: Optional[str] = None

    # Planning
    enable_action_planning: bool = True
    planning_strategy: PlanningStrategy = PlanningStrategy.ADAPTIVE

    # Response generation
    use_templates: bool = True

    # Bridge settings
    timeout_seconds: float = 30.0
    max_retries: int = 3

    @classmethod
    def minimal(cls):
        """Minimal configuration (fastest, basic features only)"""
        return cls(
            enable_meta_interpretation=False,
            enable_action_planning=False,
            fusion_strategy=FusionStrategy.AVERAGE,
            meta_mode=MetaMode.DISABLED,
            planning_strategy=PlanningStrategy.DISABLED
        )

    @classmethod
    def standard(cls):
        """Standard configuration (balanced performance/features)"""
        return cls(
            meta_mode=MetaMode.NUANCE_INTERPRETATION,
            planning_strategy=PlanningStrategy.IMMEDIATE
        )

    @classmethod
    def advanced(cls):
        """Advanced configuration (full 110/100 features)"""
        return cls(
            meta_mode=MetaMode.FULL_META,
            planning_strategy=PlanningStrategy.ADAPTIVE,
            fusion_strategy=FusionStrategy.BAYESIAN
        )


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class EmotionalInput:
    """Input to emotional intelligence pipeline"""
    text: Optional[str] = None
    video_frame: Optional[bytes] = None  # Image bytes (JPEG/PNG)
    audio_buffer: Optional[bytes] = None  # Audio bytes (WAV)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionResult:
    """Result from emotional intelligence pipeline"""

    # Primary emotion
    emotion: str
    confidence: float
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)

    # Modality-specific results
    facial_emotion: Optional[str] = None
    facial_confidence: Optional[float] = None
    vocal_emotion: Optional[str] = None
    vocal_confidence: Optional[float] = None
    text_emotion: Optional[str] = None
    text_confidence: Optional[float] = None

    # Meta-interpretation
    meta_interpretation: Optional[str] = None
    nuances: List[str] = field(default_factory=list)
    hidden_emotions: List[str] = field(default_factory=list)

    # Action plan
    suggested_actions: List[str] = field(default_factory=list)
    conversation_strategy: Optional[str] = None

    # Metadata
    processing_time_ms: Optional[float] = None
    fusion_strategy_used: Optional[str] = None

    @classmethod
    def from_js_result(cls, js_data: Dict[str, Any]) -> 'EmotionResult':
        """Create EmotionResult from JavaScript pipeline output"""
        fused = js_data.get('fusedEmotion', {})
        meta = js_data.get('metaInterpretation', {})
        action = js_data.get('actionPlan', {})

        return cls(
            emotion=fused.get('emotion', 'neutral'),
            confidence=fused.get('confidence', 0.5),
            valence=fused.get('valence', 0.0),
            arousal=fused.get('arousal', 0.5),

            # Modality-specific
            facial_emotion=js_data.get('facialResult', {}).get('emotion'),
            facial_confidence=js_data.get('facialResult', {}).get('confidence'),
            vocal_emotion=js_data.get('vocalResult', {}).get('emotion'),
            vocal_confidence=js_data.get('vocalResult', {}).get('confidence'),
            text_emotion=js_data.get('textResult', {}).get('emotion'),
            text_confidence=js_data.get('textResult', {}).get('confidence'),

            # Meta-interpretation
            meta_interpretation=meta.get('interpretation'),
            nuances=meta.get('nuances', []),
            hidden_emotions=meta.get('hiddenEmotions', []),

            # Action plan
            suggested_actions=action.get('suggestedActions', []),
            conversation_strategy=action.get('conversationStrategy'),

            # Metadata
            processing_time_ms=js_data.get('processingTime'),
            fusion_strategy_used=fused.get('strategyUsed')
        )


# ============================================================================
# Node.js Bridge
# ============================================================================

class NodeJSBridge:
    """
    Manages Node.js child process for emotional intelligence pipeline

    Communication Protocol:
    - Python → Node.js: JSON-RPC request
    - Node.js → Python: JSON-RPC response
    - One request-response per interaction
    """

    def __init__(self, config: EmotionBridgeConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.script_path = self._find_pipeline_script()

        if STRUCTLOG_AVAILABLE:
            logger.info("nodejs_bridge_initialized",
                script_path=str(self.script_path),
                config=config)

    def _find_pipeline_script(self) -> Path:
        """Auto-detect pipeline script location"""
        if self.config.pipeline_script:
            return Path(self.config.pipeline_script)

        # Try common locations
        possible_paths = [
            Path(__file__).parent.parent / "voice_ux" / "milestone3" / "complete_emotional_pipeline.js",
            Path("hololoom/voice_ux/milestone3/complete_emotional_pipeline.js"),
            Path("voice_ux/milestone3/complete_emotional_pipeline.js")
        ]

        for path in possible_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find complete_emotional_pipeline.js. Tried: {possible_paths}"
        )

    async def start(self):
        """Start Node.js bridge process"""
        if self.process is not None:
            logger.warning("Bridge already started")
            return

        # Create bridge script wrapper
        bridge_wrapper = self._create_bridge_wrapper()

        # Start Node.js process
        try:
            self.process = subprocess.Popen(
                [self.config.node_executable, "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Send bridge wrapper to stdin
            self.process.stdin.write(bridge_wrapper)
            self.process.stdin.flush()

            if STRUCTLOG_AVAILABLE:
                logger.info("nodejs_bridge_started",
                    pid=self.process.pid,
                    script=str(self.script_path))

        except Exception as e:
            logger.error("failed_to_start_bridge", error=str(e), exc_info=True)
            raise

    def _create_bridge_wrapper(self) -> str:
        """Create Node.js wrapper script for JSON-RPC communication"""

        # Convert Python config to JS config
        js_config = {
            'enableFacial': self.config.enable_facial,
            'enableVocal': self.config.enable_vocal,
            'enableText': self.config.enable_text,
            'fusionStrategy': self.config.fusion_strategy.value,
            'modalityWeights': self.config.modality_weights,
            'enableMetaInterpretation': self.config.enable_meta_interpretation,
            'metaMode': self.config.meta_mode.value,
            'llmProvider': self.config.llm_provider,
            'llmModel': self.config.llm_model,
            'llmApiKey': self.config.llm_api_key or process.env.get('ANTHROPIC_API_KEY'),
            'enableActionPlanning': self.config.enable_action_planning,
            'planningStrategy': self.config.planning_strategy.value,
            'useTemplates': self.config.use_templates
        }

        wrapper = f"""
// Auto-generated bridge wrapper
const readline = require('readline');
const {{ EmotionalIntelligencePipeline, PipelineConfig }} = require('{self.script_path}');

// Create pipeline
const config = new PipelineConfig({json.dumps(js_config)});
const pipeline = new EmotionalIntelligencePipeline(config);

// JSON-RPC handler
const rl = readline.createInterface({{
    input: process.stdin,
    output: process.stdout,
    terminal: false
}});

rl.on('line', async (line) => {{
    try {{
        const request = JSON.parse(line);

        if (request.method === 'process') {{
            const result = await pipeline.process(request.params);
            console.log(JSON.stringify({{ id: request.id, result }}));
        }} else {{
            console.log(JSON.stringify({{
                id: request.id,
                error: {{ code: -32601, message: 'Method not found' }}
            }}));
        }}
    }} catch (error) {{
        console.log(JSON.stringify({{
            id: null,
            error: {{ code: -32603, message: error.message }}
        }}));
    }}
}});

// Keep alive
process.stdin.resume();
"""
        return wrapper

    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call Node.js method via JSON-RPC"""
        if self.process is None:
            await self.start()

        # Create JSON-RPC request
        request_id = id(params)  # Unique ID
        request = {
            'jsonrpc': '2.0',
            'id': request_id,
            'method': method,
            'params': params
        }

        # Send request
        request_json = json.dumps(request)
        self.process.stdin.write(request_json + '\n')
        self.process.stdin.flush()

        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(
                self._read_stdout_line(),
                timeout=self.config.timeout_seconds
            )

            response = json.loads(response_line)

            if 'error' in response:
                raise Exception(f"Node.js error: {response['error']}")

            return response.get('result', {})

        except asyncio.TimeoutError:
            logger.error("nodejs_call_timeout", method=method)
            raise

    async def _read_stdout_line(self) -> str:
        """Read line from stdout asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process.stdout.readline)

    async def stop(self):
        """Stop Node.js bridge process"""
        if self.process is None:
            return

        try:
            self.process.terminate()
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, self.process.wait),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            self.process.kill()

        self.process = None

        if STRUCTLOG_AVAILABLE:
            logger.info("nodejs_bridge_stopped")


# ============================================================================
# Emotion Bridge - High-Level API
# ============================================================================

class EmotionBridge:
    """
    High-level Python API for emotional intelligence

    Usage:
        async with EmotionBridge(config) as bridge:
            result = await bridge.analyze_emotion(
                text="I'm feeling frustrated with this bug",
                context={'activity': 'coding'}
            )
            print(f"Emotion: {result.emotion} ({result.confidence:.2f})")
            print(f"Suggested actions: {result.suggested_actions}")
    """

    def __init__(self, config: Optional[EmotionBridgeConfig] = None):
        self.config = config or EmotionBridgeConfig.standard()
        self.bridge = NodeJSBridge(self.config)
        self._started = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start emotion bridge"""
        if self._started:
            return

        await self.bridge.start()
        self._started = True

        if STRUCTLOG_AVAILABLE:
            logger.info("emotion_bridge_started")

    async def stop(self):
        """Stop emotion bridge"""
        if not self._started:
            return

        await self.bridge.stop()
        self._started = False

        if STRUCTLOG_AVAILABLE:
            logger.info("emotion_bridge_stopped")

    async def analyze_emotion(
        self,
        text: Optional[str] = None,
        video_frame: Optional[bytes] = None,
        audio_buffer: Optional[bytes] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionResult:
        """
        Analyze emotion from multimodal input

        Args:
            text: Text input (transcript, message, etc.)
            video_frame: Video frame bytes (JPEG/PNG)
            audio_buffer: Audio bytes (WAV)
            context: Additional context (activity, location, etc.)

        Returns:
            EmotionResult with detected emotion and action plan

        Example:
            result = await bridge.analyze_emotion(
                text="I can't figure this out",
                context={'activity': 'coding', 'session_duration': 30000}
            )
        """
        # Build input
        input_data = {}

        if text:
            input_data['text'] = text

        if video_frame:
            # Convert bytes to base64 for JSON transport
            import base64
            input_data['videoFrame'] = base64.b64encode(video_frame).decode('utf-8')

        if audio_buffer:
            import base64
            input_data['audioBuffer'] = base64.b64encode(audio_buffer).decode('utf-8')

        if context:
            input_data['context'] = context

        # Call Node.js pipeline
        try:
            result = await self.bridge.call('process', input_data)
            return EmotionResult.from_js_result(result)

        except Exception as e:
            logger.error("emotion_analysis_failed", error=str(e), exc_info=True)

            # Return neutral fallback
            return EmotionResult(
                emotion='neutral',
                confidence=0.0,
                valence=0.0,
                arousal=0.5,
                suggested_actions=['Unable to analyze emotion - check logs']
            )


# ============================================================================
# Voice Agent Integration
# ============================================================================

async def enhance_voice_agent_with_emotions(voice_agent, emotion_config: Optional[EmotionBridgeConfig] = None):
    """
    Enhance VoiceAgent with emotional intelligence

    Args:
        voice_agent: HoloLoom VoiceAgent instance
        emotion_config: Emotion bridge configuration

    Example:
        from hololoom.voice import VoiceAgent
        from hololoom.voice.emotion_bridge import enhance_voice_agent_with_emotions

        agent = VoiceAgent(orchestrator=orchestrator)
        await enhance_voice_agent_with_emotions(agent)

        # Now voice agent automatically detects emotions!
        response = await agent.process_voice_input("I'm stuck on this bug")
        # Response includes emotional context and suggestions
    """
    # Create emotion bridge
    emotion_bridge = EmotionBridge(emotion_config or EmotionBridgeConfig.standard())
    await emotion_bridge.start()

    # Store original process method
    original_process = voice_agent.process_voice_input

    # Wrap with emotion detection
    async def process_with_emotion(transcript: str) -> str:
        """Process voice input with emotional intelligence"""

        # Analyze emotion
        emotion_result = await emotion_bridge.analyze_emotion(
            text=transcript,
            context={
                'agent': voice_agent.agent_name,
                'session_id': voice_agent.conversation_memory.session_id,
                'turns_in_session': len(voice_agent.conversation_memory.turns)
            }
        )

        # Add emotional context to conversation memory
        if voice_agent.conversation_memory.turns:
            last_turn = voice_agent.conversation_memory.turns[-1]
            last_turn.metadata.update({
                'emotion': emotion_result.emotion,
                'emotion_confidence': emotion_result.confidence,
                'valence': emotion_result.valence,
                'arousal': emotion_result.arousal,
                'suggested_actions': emotion_result.suggested_actions
            })

        # Call original process (with emotional context now available)
        response = await original_process(transcript)

        # Optionally append emotional guidance to response
        if emotion_result.suggested_actions and emotion_result.confidence > 0.7:
            guidance = f"\n\n💡 {emotion_result.suggested_actions[0]}"
            response += guidance

        return response

    # Replace method
    voice_agent.process_voice_input = process_with_emotion

    # Store bridge for cleanup
    voice_agent._emotion_bridge = emotion_bridge

    if STRUCTLOG_AVAILABLE:
        logger.info("voice_agent_enhanced_with_emotions",
            agent=voice_agent.agent_name,
            config=emotion_config)

    return voice_agent


# ============================================================================
# Usage Examples
# ============================================================================

async def example_basic_emotion_detection():
    """Example: Basic emotion detection"""

    async with EmotionBridge(EmotionBridgeConfig.standard()) as bridge:
        # Analyze text emotion
        result = await bridge.analyze_emotion(
            text="I'm feeling frustrated with this bug. I've been stuck for hours!",
            context={
                'activity': 'coding',
                'session_duration': 180000  # 3 hours
            }
        )

        print(f"Emotion: {result.emotion} (confidence: {result.confidence:.2f})")
        print(f"Valence: {result.valence:.2f} (negative)")
        print(f"Arousal: {result.arousal:.2f}")
        print(f"Suggested actions: {result.suggested_actions}")
        print(f"Strategy: {result.conversation_strategy}")

        # Expected output:
        # Emotion: frustrated (confidence: 0.92)
        # Valence: -0.65 (negative)
        # Arousal: 0.78
        # Suggested actions: ['Suggest taking a break', 'Offer debugging assistance']
        # Strategy: supportive


async def example_voice_agent_integration():
    """Example: Integrate with voice agent"""

    # Import voice agent (if available)
    try:
        from hololoom.voice import VoiceAgent
        from hololoom.weaving_orchestrator import WeavingOrchestrator
        from hololoom.config import Config
    except ImportError:
        print("HoloLoom not available - skipping example")
        return

    # Create orchestrator
    config = Config.fast()
    shards = []

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Create voice agent
        agent = VoiceAgent(orchestrator=orchestrator, agent_name="Elle")

        # Enhance with emotions!
        await enhance_voice_agent_with_emotions(
            agent,
            EmotionBridgeConfig.advanced()  # Full 110/100 features
        )

        # Now agent automatically detects emotions
        response = await agent.process_voice_input(
            "I can't figure this out. Nothing works!"
        )

        print(f"Response: {response}")
        # Response includes emotional context and helpful suggestions


if __name__ == "__main__":
    print(__doc__)

    print("\n" + "="*70)
    print("EMOTION BRIDGE READY!")
    print("="*70)
    print("\nFeatures:")
    print("  ✅ Python ↔ Node.js bridge for 110/100 emotional intelligence")
    print("  ✅ Multimodal emotion detection (text, facial, vocal)")
    print("  ✅ Meta-interpretation with LLM")
    print("  ✅ Action planning and conversation strategy")
    print("  ✅ VoiceAgent integration")
    print("\nConfiguration Presets:")
    print("  - Minimal: Basic fusion only (fastest)")
    print("  - Standard: Nuance interpretation + immediate planning")
    print("  - Advanced: Full 110/100 (meta-learning + adaptive planning)")
    print("\nUsage:")
    print("  async with EmotionBridge(config) as bridge:")
    print("      result = await bridge.analyze_emotion(text='...')")
    print("\n" + "="*70)

    # Uncomment to run examples:
    # asyncio.run(example_basic_emotion_detection())
    # asyncio.run(example_voice_agent_integration())
