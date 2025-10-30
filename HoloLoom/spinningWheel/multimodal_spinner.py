"""
Multi-Modal Spinner

Unified spinner that handles all input modalities using the InputRouter.
Converts raw inputs into MemoryShards with proper modality tagging.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from HoloLoom.input.protocol import ModalityType, ProcessedInput
from HoloLoom.input.router import InputRouter
from HoloLoom.input.fusion import MultiModalFusion
from HoloLoom.documentation.types import MemoryShard


class MultiModalSpinner:
    """
    Unified spinner for all input modalities.
    
    Uses InputRouter to automatically detect and process any input type,
    then converts ProcessedInput into MemoryShards for the memory system.
    
    Features:
    - Auto-detection of input modality
    - Unified processing pipeline
    - Multi-modal fusion support
    - Cross-modal shard creation
    """
    
    def __init__(self, enable_fusion: bool = True):
        """
        Initialize multi-modal spinner.
        
        Args:
            enable_fusion: Whether to fuse multi-modal inputs
        """
        self.router = InputRouter()
        self.fusion = MultiModalFusion() if enable_fusion else None
        self.enable_fusion = enable_fusion
    
    async def spin(self, raw_data: Union[str, Dict, List, Path, bytes]) -> List[MemoryShard]:
        """
        Process raw input into MemoryShards.
        
        Args:
            raw_data: Raw input of any supported type
        
        Returns:
            List of MemoryShards with modality metadata
        """
        start_time = time.time()
        
        # Process input through router
        try:
            processed = await self.router.process(raw_data)
        except Exception as e:
            print(f"Warning: MultiModalSpinner processing failed: {e}")
            # Return empty shard on error
            return [self._create_error_shard(raw_data, str(e))]
        
        # Convert to shard(s)
        if processed.modality == ModalityType.MULTIMODAL and self.enable_fusion:
            # Multi-modal input - create shards for each component
            shards = self._create_multimodal_shards(processed)
        else:
            # Single modality - create single shard
            shards = [self._create_shard(processed)]
        
        # Add processing metadata
        processing_time = (time.time() - start_time) * 1000
        for shard in shards:
            shard.metadata['processing_time_ms'] = processing_time
            shard.metadata['spinner'] = 'MultiModalSpinner'
        
        return shards
    
    def _create_shard(self, processed: ProcessedInput) -> MemoryShard:
        """Create MemoryShard from ProcessedInput."""
        # Extract entities and motifs from features if available
        entities = []
        motifs = []
        
        if isinstance(processed.features, dict):
            if 'text' in processed.features:
                text_features = processed.features['text']
                if hasattr(text_features, 'entities'):
                    entities = [e.get('text', '') for e in text_features.entities if isinstance(e, dict)]
                if hasattr(text_features, 'topics'):
                    motifs = text_features.topics
        
        return MemoryShard(
            id=f"{processed.modality.value}_{hash(processed.content[:100])}",
            text=processed.content,
            episode=processed.source,
            entities=entities,
            motifs=motifs,
            metadata={
                'modality_type': processed.modality.name,
                'confidence': processed.confidence,
                'embedding': processed.embedding.tolist() if processed.embedding is not None else None,
                'aligned_embeddings': processed.aligned_embeddings,
                'features': processed.features
            }
        )
    
    def _create_multimodal_shards(self, processed: ProcessedInput) -> List[MemoryShard]:
        """
        Create multiple shards from multi-modal input.
        
        For multi-modal inputs, we create:
        1. Individual shards for each component
        2. A fused shard combining all components
        """
        shards = []
        
        # Extract component features
        if 'components' in processed.features:
            components = processed.features['components']
            
            for i, component in enumerate(components):
                content = component.get('content', f"Component {i}")
                shard = MemoryShard(
                    id=f"multimodal_component_{i}_{hash(content[:100])}",
                    text=content,
                    episode=processed.source,
                    entities=[],
                    motifs=[],
                    metadata={
                        'component_index': i,
                        'parent_modality': ModalityType.MULTIMODAL.name,
                        'is_component': True,
                        'modality': component.get('modality', ModalityType.TEXT.value),
                        'confidence': component.get('confidence', 0.5),
                        'embedding': component.get('embedding'),
                        'features': component.get('features', {})
                    }
                )
                shards.append(shard)
        
        # Add fused shard
        fused_shard = MemoryShard(
            id=f"multimodal_fused_{hash(processed.content[:100])}",
            text=processed.content,
            episode=processed.source,
            entities=[],
            motifs=[],
            metadata={
                'modality_type': ModalityType.MULTIMODAL.name,
                'component_count': len(shards),
                'is_fused': True,
                'confidence': processed.confidence,
                'embedding': processed.embedding.tolist() if processed.embedding is not None else None,
                'features': processed.features
            }
        )
        shards.append(fused_shard)
        
        return shards
    
    def _create_error_shard(self, raw_data: Any, error: str) -> MemoryShard:
        """Create error shard when processing fails."""
        content = f"Error processing input: {str(raw_data)[:100]}"
        return MemoryShard(
            id=f"error_{hash(content)}",
            text=content,
            episode='error',
            entities=[],
            motifs=[],
            metadata={
                'processing_failed': True,
                'error_message': error,
                'confidence': 0.0
            }
        )
    
    def get_supported_modalities(self) -> List[ModalityType]:
        """Get list of supported modalities."""
        return self.router.get_available_processors()


class TextSpinner(MultiModalSpinner):
    """Specialized spinner for text inputs."""
    
    def __init__(self):
        super().__init__(enable_fusion=False)
    
    async def spin(self, raw_data: Union[str, Dict, Path]) -> List[MemoryShard]:
        """Process text input."""
        # Ensure input is text
        if isinstance(raw_data, dict):
            raw_data = raw_data.get('text', str(raw_data))
        elif isinstance(raw_data, Path):
            with open(raw_data, 'r', encoding='utf-8') as f:
                raw_data = f.read()
        else:
            raw_data = str(raw_data)
        
        return await super().spin(raw_data)


class ImageSpinner(MultiModalSpinner):
    """Specialized spinner for image inputs."""
    
    def __init__(self):
        super().__init__(enable_fusion=False)
    
    async def spin(self, raw_data: Union[str, Path, bytes]) -> List[MemoryShard]:
        """Process image input."""
        # Validate it's an image path or bytes
        if isinstance(raw_data, str):
            raw_data = Path(raw_data)
        
        return await super().spin(raw_data)


class AudioSpinner(MultiModalSpinner):
    """Specialized spinner for audio inputs."""
    
    def __init__(self):
        super().__init__(enable_fusion=False)
    
    async def spin(self, raw_data: Union[str, Path, bytes]) -> List[MemoryShard]:
        """Process audio input."""
        # Validate it's an audio path or bytes
        if isinstance(raw_data, str):
            raw_data = Path(raw_data)
        
        return await super().spin(raw_data)


class StructuredDataSpinner(MultiModalSpinner):
    """Specialized spinner for structured data inputs."""
    
    def __init__(self):
        super().__init__(enable_fusion=False)
    
    async def spin(self, raw_data: Union[Dict, List, str, Path]) -> List[MemoryShard]:
        """Process structured data input."""
        # Handle file paths
        if isinstance(raw_data, (str, Path)):
            path = Path(raw_data)
            if path.suffix.lower() == '.json':
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            elif path.suffix.lower() == '.csv':
                # Will be handled by router/processor
                pass
        
        return await super().spin(raw_data)


class CrossModalSpinner(MultiModalSpinner):
    """
    Spinner for processing multiple inputs with cross-modal fusion.
    
    Enables queries like "Show me text and images about quantum computing"
    by processing multiple modalities and fusing them.
    """
    
    def __init__(self):
        super().__init__(enable_fusion=True)
    
    async def spin_multiple(
        self,
        inputs: List[Any],
        fusion_strategy: str = "attention"
    ) -> List[MemoryShard]:
        """
        Process multiple inputs with cross-modal fusion.
        
        Args:
            inputs: List of raw inputs (different modalities)
            fusion_strategy: Fusion strategy ("attention", "concat", "average", "max")
        
        Returns:
            List of MemoryShards including individual and fused shards
        """
        start_time = time.time()
        
        # Process each input
        all_processed = []
        all_shards = []
        
        for raw_input in inputs:
            try:
                processed = await self.router.process(raw_input)
                all_processed.append(processed)
                
                # Create individual shard
                shard = self._create_shard(processed)
                shard.metadata['cross_modal_component'] = True
                all_shards.append(shard)
                
            except Exception as e:
                print(f"Warning: Failed to process input: {e}")
                continue
        
        # Fuse if multiple inputs
        if len(all_processed) > 1 and self.fusion:
            try:
                fused = await self.fusion.fuse(all_processed, strategy=fusion_strategy)
                
                # Create fused shard
                content = f"Cross-modal fusion of {len(all_processed)} inputs"
                fused_shard = MemoryShard(
                    id=f"cross_modal_fused_{hash(content)}",
                    text=content,
                    episode='cross_modal_fusion',
                    entities=[],
                    motifs=[],
                    metadata={
                        'modality_type': ModalityType.MULTIMODAL.name,
                        'fusion_strategy': fusion_strategy,
                        'component_count': len(all_processed),
                        'component_modalities': [p.modality.name for p in all_processed],
                        'is_fused': True,
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'confidence': fused.confidence,
                        'embedding': fused.embedding.tolist() if fused.embedding is not None else None,
                        'features': fused.features
                    }
                )
                all_shards.append(fused_shard)
                
            except Exception as e:
                print(f"Warning: Fusion failed: {e}")
        
        return all_shards
    
    async def spin_query(
        self,
        query: str,
        modality_filter: Optional[List[ModalityType]] = None,
        fusion_strategy: str = "attention"
    ) -> List[MemoryShard]:
        """
        Process a cross-modal query.
        
        Example: "Show me text and images about quantum computing"
        
        Args:
            query: Natural language query
            modality_filter: List of modalities to search (None = all)
            fusion_strategy: How to fuse results
        
        Returns:
            List of MemoryShards from cross-modal search
        """
        # For now, just process the query as text
        # In full implementation, this would:
        # 1. Parse query to extract modality requirements
        # 2. Search memory for matching shards
        # 3. Fuse results from different modalities
        
        query_processed = await self.router.process(query)
        
        shard = MemoryShard(
            id=f"cross_modal_query_{hash(query)}",
            text=query,
            episode='cross_modal_query',
            entities=[],
            motifs=[],
            metadata={
                'is_query': True,
                'cross_modal': True,
                'query': query,
                'modality_filter': [m.name for m in modality_filter] if modality_filter else 'all',
                'fusion_strategy': fusion_strategy,
                'confidence': query_processed.confidence,
                'embedding': query_processed.embedding.tolist() if query_processed.embedding is not None else None
            }
        )
        
        return [shard]
