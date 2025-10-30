"""
Input Router

Auto-detects input type and routes to appropriate processor.
"""

from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import mimetypes

from .protocol import ProcessedInput, ModalityType, InputData
from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .structured_processor import StructuredDataProcessor
from .fusion import MultiModalFusion


class InputRouter:
    """
    Intelligent input router that auto-detects input type.
    
    Routes inputs to appropriate processors:
    - Text strings → TextProcessor
    - Image files/bytes → ImageProcessor
    - Audio files/bytes → AudioProcessor
    - JSON/CSV files → StructuredDataProcessor
    - Multiple inputs → MultiModalFusion
    """
    
    def __init__(
        self,
        text_processor: Optional[TextProcessor] = None,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        structured_processor: Optional[StructuredDataProcessor] = None,
        fusion: Optional[MultiModalFusion] = None
    ):
        """
        Initialize input router.
        
        Args:
            text_processor: Text processor instance (creates if None)
            image_processor: Image processor instance (creates if None)
            audio_processor: Audio processor instance (creates if None)
            structured_processor: Structured data processor instance (creates if None)
            fusion: Multi-modal fusion instance (creates if None)
        """
        self.text_processor = text_processor or TextProcessor(
            embedder=None,
            use_spacy=False,
            use_textblob=False
        )
        
        self.image_processor = image_processor or ImageProcessor(
            use_clip=False,  # Disabled by default
            use_ocr=False
        )
        
        self.audio_processor = audio_processor or AudioProcessor(
            use_whisper=False  # Disabled by default
        )
        
        self.structured_processor = structured_processor or StructuredDataProcessor()
        
        self.fusion = fusion or MultiModalFusion()
    
    async def process(
        self,
        input_data: Union[InputData, List[InputData]],
        **kwargs
    ) -> ProcessedInput:
        """
        Process input(s) and return unified representation.
        
        Auto-detects input type and routes to appropriate processor.
        If multiple inputs provided, fuses them together.
        
        Args:
            input_data: Single input or list of inputs
            **kwargs: Additional processor-specific arguments
        
        Returns:
            ProcessedInput with appropriate modality
        """
        # Handle list of inputs (multi-modal)
        if isinstance(input_data, list):
            processed = []
            for inp in input_data:
                result = await self._process_single(inp, **kwargs)
                processed.append(result)
            
            # Fuse if multiple modalities
            if len(processed) > 1:
                return await self.fusion.fuse(processed, strategy=kwargs.get('fusion_strategy', 'attention'))
            else:
                return processed[0]
        else:
            return await self._process_single(input_data, **kwargs)
    
    async def _process_single(
        self,
        input_data: InputData,
        **kwargs
    ) -> ProcessedInput:
        """Process single input."""
        # Detect modality
        modality = self.detect_modality(input_data)
        
        # Route to appropriate processor
        if modality == ModalityType.TEXT:
            return await self.text_processor.process(input_data, **kwargs)
        elif modality == ModalityType.IMAGE:
            return await self.image_processor.process(input_data, **kwargs)
        elif modality == ModalityType.AUDIO:
            return await self.audio_processor.process(input_data, **kwargs)
        elif modality == ModalityType.STRUCTURED:
            return await self.structured_processor.process(input_data, **kwargs)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def detect_modality(self, input_data: InputData) -> ModalityType:
        """
        Detect input modality from data.
        
        Args:
            input_data: Input data (string, path, bytes, dict)
        
        Returns:
            Detected modality type
        """
        # Dict with explicit modality key
        if isinstance(input_data, dict):
            if 'modality' in input_data:
                return ModalityType(input_data['modality'])
            
            # Check for specific keys
            if 'image' in input_data or 'image_path' in input_data:
                return ModalityType.IMAGE
            elif 'audio' in input_data or 'audio_path' in input_data:
                return ModalityType.AUDIO
            elif 'text' in input_data:
                return ModalityType.TEXT
            else:
                # Assume structured data
                return ModalityType.STRUCTURED
        
        # File path
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            
            if path.exists() and path.is_file():
                return self._detect_from_file(path)
            else:
                # Assume it's text content
                return ModalityType.TEXT
        
        # Bytes - try to detect from magic numbers
        if isinstance(input_data, bytes):
            return self._detect_from_bytes(input_data)
        
        # Default to text
        return ModalityType.TEXT
    
    def _detect_from_file(self, path: Path) -> ModalityType:
        """Detect modality from file path."""
        suffix = path.suffix.lower()
        
        # Image extensions
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
        if suffix in image_exts:
            return ModalityType.IMAGE
        
        # Audio extensions
        audio_exts = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma'}
        if suffix in audio_exts:
            return ModalityType.AUDIO
        
        # Video extensions
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        if suffix in video_exts:
            return ModalityType.VIDEO
        
        # Structured data extensions
        structured_exts = {'.json', '.csv', '.tsv', '.xml', '.yaml', '.yml'}
        if suffix in structured_exts:
            return ModalityType.STRUCTURED
        
        # Text extensions (or unknown)
        return ModalityType.TEXT
    
    def _detect_from_bytes(self, data: bytes) -> ModalityType:
        """Detect modality from byte content using magic numbers."""
        if len(data) < 4:
            return ModalityType.TEXT
        
        # Check magic numbers
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return ModalityType.IMAGE
        
        # JPEG
        if data[:2] == b'\xff\xd8':
            return ModalityType.IMAGE
        
        # GIF
        if data[:3] == b'GIF':
            return ModalityType.IMAGE
        
        # WAV
        if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
            return ModalityType.AUDIO
        
        # MP3
        if data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
            return ModalityType.AUDIO
        
        # Try to decode as text
        try:
            data.decode('utf-8')
            return ModalityType.TEXT
        except UnicodeDecodeError:
            pass
        
        # Default to text
        return ModalityType.TEXT
    
    def get_available_processors(self) -> Dict[ModalityType, bool]:
        """
        Check which processors are available.
        
        Returns:
            Dict mapping modality to availability
        """
        return {
            ModalityType.TEXT: self.text_processor.is_available(),
            ModalityType.IMAGE: self.image_processor.is_available(),
            ModalityType.AUDIO: self.audio_processor.is_available(),
            ModalityType.STRUCTURED: self.structured_processor.is_available(),
        }
    
    async def process_batch(
        self,
        inputs: List[InputData],
        fusion_strategy: str = "attention",
        **kwargs
    ) -> List[ProcessedInput]:
        """
        Process multiple inputs individually.
        
        Args:
            inputs: List of inputs to process
            fusion_strategy: Strategy if fusing multi-modal inputs
            **kwargs: Additional processor arguments
        
        Returns:
            List of ProcessedInput results
        """
        results = []
        for inp in inputs:
            result = await self.process(inp, fusion_strategy=fusion_strategy, **kwargs)
            results.append(result)
        return results
