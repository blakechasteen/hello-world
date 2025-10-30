"""
Image Processor

Processes images using vision models (CLIP, ResNet, etc.).
Extracts features, generates captions, detects objects.
"""

import time
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .protocol import (
    InputProcessorProtocol,
    ProcessedInput,
    ModalityType,
    ImageFeatures,
    InputMetadata
)


class ImageProcessor:
    """
    Image processor using vision models.
    
    Features:
    - CLIP embeddings for semantic understanding
    - Object detection (optional)
    - Scene classification
    - Image captioning (via CLIP text similarity)
    - OCR for text extraction (optional)
    - Color analysis
    """
    
    def __init__(self, use_clip: bool = True, use_ocr: bool = False):
        """
        Initialize image processor.
        
        Args:
            use_clip: Whether to use CLIP for embeddings
            use_ocr: Whether to use OCR for text extraction
        """
        self.use_clip = use_clip
        self.use_ocr = use_ocr
        
        # Try to load CLIP
        self.clip_model = None
        self.clip_preprocess = None
        if use_clip:
            try:
                import torch
                import clip
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                self.clip_device = device
            except ImportError:
                print("Warning: CLIP not available. Using fallback.")
                self.use_clip = False
        
        # Try to load PIL
        try:
            from PIL import Image
            self.PIL_Image = Image
        except ImportError:
            raise ImportError("PIL/Pillow is required for image processing")
        
        # Try to load OCR
        self.pytesseract = None
        if use_ocr:
            try:
                import pytesseract
                self.pytesseract = pytesseract
            except ImportError:
                print("Warning: pytesseract not available. OCR disabled.")
                self.use_ocr = False
    
    async def process(
        self,
        input_data: Union[str, Path, bytes, Dict],
        generate_caption: bool = True,
        extract_colors: bool = True,
        extract_text: bool = False,
        **kwargs
    ) -> ProcessedInput:
        """
        Process image input.
        
        Args:
            input_data: Image path, bytes, or dict with 'image' key
            generate_caption: Whether to generate caption
            extract_colors: Whether to extract dominant colors
            extract_text: Whether to extract text via OCR
        
        Returns:
            ProcessedInput with image features
        """
        start_time = time.time()
        
        # Load image
        if isinstance(input_data, dict):
            image_path = input_data.get('image')
            source = input_data.get('source')
        elif isinstance(input_data, bytes):
            import io
            image_path = io.BytesIO(input_data)
            source = None
        else:
            image_path = Path(input_data)
            source = str(image_path)
        
        image = self.PIL_Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create features
        features = ImageFeatures()
        features.dimensions = image.size
        
        # Generate embedding with CLIP
        embedding = None
        if self.use_clip and self.clip_model:
            embedding = self._generate_clip_embedding(image)
        
        # Generate caption
        caption = None
        if generate_caption and self.use_clip and self.clip_model:
            caption = self._generate_caption(image)
            features.caption = caption
        
        # Classify scene
        if self.use_clip and self.clip_model:
            features.scene = self._classify_scene(image)
        
        # Extract colors
        if extract_colors:
            features.colors = self._extract_dominant_colors(image)
        
        # Extract text via OCR
        if extract_text and self.use_ocr and self.pytesseract:
            features.ocr_text = self._extract_text(image)
        
        # Create content description
        content_parts = []
        if caption:
            content_parts.append(caption)
        if features.scene:
            content_parts.append(f"Scene: {features.scene}")
        if features.colors:
            content_parts.append(f"Colors: {', '.join(features.colors[:3])}")
        
        content = " | ".join(content_parts) if content_parts else "Image"
        
        # Calculate confidence
        confidence = 0.9 if self.use_clip else 0.7
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessedInput(
            modality=ModalityType.IMAGE,
            content=content,
            embedding=embedding,
            confidence=confidence,
            source=source,
            features={'image': features.to_dict()}
        )
    
    def _generate_clip_embedding(self, image) -> np.ndarray:
        """Generate CLIP embedding for image."""
        import torch
        
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.clip_device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def _generate_caption(self, image) -> str:
        """Generate caption using CLIP text similarity."""
        import torch
        
        # Candidate captions
        candidates = [
            "a photo of a person",
            "a photo of a dog",
            "a photo of a cat",
            "a photo of a car",
            "a photo of a building",
            "a photo of nature",
            "a photo of food",
            "a photo of an animal",
            "a landscape photo",
            "an indoor scene",
            "an outdoor scene",
            "a close-up photo",
            "a group photo",
            "a portrait",
            "an abstract image"
        ]
        
        # Encode image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.clip_device)
        
        # Encode text candidates
        import clip
        text_inputs = clip.tokenize(candidates).to(self.clip_device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T).squeeze(0)
        
        # Get best match
        best_idx = similarity.argmax().item()
        return candidates[best_idx]
    
    def _classify_scene(self, image) -> str:
        """Classify scene type."""
        import torch
        import clip
        
        # Scene categories
        scenes = [
            "indoor",
            "outdoor",
            "urban",
            "nature",
            "beach",
            "mountain",
            "forest",
            "city",
            "office",
            "home"
        ]
        
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.clip_device)
        text_inputs = clip.tokenize(scenes).to(self.clip_device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).squeeze(0)
        
        best_idx = similarity.argmax().item()
        return scenes[best_idx]
    
    def _extract_dominant_colors(self, image, n_colors: int = 5) -> List[str]:
        """Extract dominant colors from image."""
        import numpy as np
        
        # Resize for speed
        image_small = image.resize((150, 150))
        pixels = np.array(image_small).reshape(-1, 3)
        
        # Simple k-means clustering (or just use most frequent)
        # For simplicity, divide into color bins
        color_names = {
            (255, 0, 0): 'red',
            (0, 255, 0): 'green',
            (0, 0, 255): 'blue',
            (255, 255, 0): 'yellow',
            (255, 165, 0): 'orange',
            (128, 0, 128): 'purple',
            (255, 192, 203): 'pink',
            (165, 42, 42): 'brown',
            (0, 0, 0): 'black',
            (255, 255, 255): 'white',
            (128, 128, 128): 'gray'
        }
        
        # Find closest color names
        pixel_colors = []
        for pixel in pixels[::100]:  # Sample every 100th pixel
            min_dist = float('inf')
            closest_color = 'unknown'
            
            for rgb, name in color_names.items():
                dist = np.sum((pixel - np.array(rgb)) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_color = name
            
            pixel_colors.append(closest_color)
        
        # Count frequencies
        from collections import Counter
        color_counts = Counter(pixel_colors)
        
        return [color for color, _ in color_counts.most_common(n_colors)]
    
    def _extract_text(self, image) -> Optional[str]:
        """Extract text using OCR."""
        if not self.pytesseract:
            return None
        
        try:
            text = self.pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None
        except Exception:
            return None
    
    def get_modality(self) -> ModalityType:
        """Return modality type."""
        return ModalityType.IMAGE
    
    def is_available(self) -> bool:
        """Check if processor is available."""
        return self.PIL_Image is not None
