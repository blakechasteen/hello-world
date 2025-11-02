# -*- coding: utf-8 -*-
"""
Image Spinner with OCR
======================
Input adapter for ingesting images with text extraction (OCR) as MemoryShards.

Supports:
- OCR text extraction from images
- Receipt parsing
- Document scanning
- Photo metadata extraction
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import base64
from io import BytesIO

from .base import BaseSpinner, SpinnerConfig

# Import HoloLoom types
try:
    from HoloLoom.documentation.types import MemoryShard
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

# Optional OCR dependencies
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    pytesseract = None
    Image = None

# Optional AI vision for advanced extraction
try:
    import ollama
    OLLAMA_VISION_AVAILABLE = True
except ImportError:
    OLLAMA_VISION_AVAILABLE = False
    ollama = None

logger = logging.getLogger(__name__)


@dataclass
class ImageSpinnerConfig(SpinnerConfig):
    """Configuration for Image spinner with OCR."""
    ocr_engine: str = "tesseract"  # tesseract or ollama-vision
    tesseract_config: str = "--psm 6"  # Page segmentation mode
    vision_model: str = "llava:7b"  # Ollama vision model for advanced extraction
    extract_metadata: bool = True  # Extract EXIF metadata
    save_images: bool = False  # Save processed images locally
    image_storage_path: Optional[str] = None


class ImageSpinner(BaseSpinner):
    """
    Spinner for images with OCR text extraction.

    Converts images to MemoryShards with extracted text and metadata.
    """

    def __init__(self, config: ImageSpinnerConfig = None):
        if config is None:
            config = ImageSpinnerConfig()
        super().__init__(config)

        # Check OCR availability
        if config.ocr_engine == "tesseract" and not OCR_AVAILABLE:
            logger.warning(
                "pytesseract not available. Install: pip install pytesseract pillow\n"
                "Also install tesseract: https://github.com/tesseract-ocr/tesseract"
            )
        elif config.ocr_engine == "ollama-vision" and not OLLAMA_VISION_AVAILABLE:
            logger.warning("ollama not available. Install: pip install ollama")

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert image → MemoryShards with OCR text.

        Args:
            raw_data: Dict with keys:
                - 'image_path': Path to image file (str or Path)
                - 'image_data': Raw bytes or base64-encoded image
                - 'url': URL to download image from
                - 'episode': Episode/session identifier
                - 'metadata': Additional metadata

        Returns:
            List of MemoryShard objects with extracted text
        """
        shards = []
        episode = raw_data.get('episode', 'image_session')

        # Load image
        image = await self._load_image(raw_data)
        if image is None:
            logger.error("Failed to load image")
            return shards

        # Extract text via OCR
        text = await self._extract_text(image, raw_data)

        # Extract metadata
        metadata = {'type': 'image', **raw_data.get('metadata', {})}
        if self.config.extract_metadata:
            image_metadata = self._extract_image_metadata(image)
            metadata.update(image_metadata)

        # Create shard
        shard = MemoryShard(
            id=f"{episode}_image",
            text=text,
            episode=episode,
            entities=[],
            motifs=[],
            metadata=metadata
        )

        # Optionally enrich with Ollama
        if self.config.enable_enrichment and text:
            enrichment = await self.enrich(text)
            if 'ollama' in enrichment:
                ollama_data = enrichment['ollama']
                shard.entities = ollama_data.get('entities', [])
                shard.motifs = ollama_data.get('motifs', [])
                shard.metadata['enrichment'] = enrichment

        shards.append(shard)

        # Save image if configured
        if self.config.save_images and self.config.image_storage_path:
            self._save_image(image, episode)

        return shards

    async def _load_image(self, raw_data: Dict[str, Any]):
        """Load image from various sources."""
        if not OCR_AVAILABLE and self.config.ocr_engine == "tesseract":
            return None

        # From file path
        if 'image_path' in raw_data:
            path = Path(raw_data['image_path'])
            if path.exists():
                return Image.open(path)

        # From raw bytes
        if 'image_data' in raw_data:
            data = raw_data['image_data']
            if isinstance(data, str):
                # Assume base64
                import base64
                data = base64.b64decode(data)
            return Image.open(BytesIO(data))

        # From URL
        if 'url' in raw_data:
            try:
                import requests
                response = requests.get(raw_data['url'], timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                logger.error(f"Failed to download image from URL: {e}")

        return None

    async def _extract_text(self, image, raw_data: Dict[str, Any]) -> str:
        """Extract text from image using configured OCR engine."""
        if self.config.ocr_engine == "tesseract":
            return self._ocr_tesseract(image)
        elif self.config.ocr_engine == "ollama-vision":
            return await self._ocr_ollama_vision(image, raw_data)
        else:
            logger.warning(f"Unknown OCR engine: {self.config.ocr_engine}")
            return ""

    def _ocr_tesseract(self, image) -> str:
        """Extract text using Tesseract OCR."""
        if not OCR_AVAILABLE:
            return ""

        try:
            text = pytesseract.image_to_string(image, config=self.config.tesseract_config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    async def _ocr_ollama_vision(self, image, raw_data: Dict[str, Any]) -> str:
        """Extract text using Ollama vision model (multimodal LLM)."""
        if not OLLAMA_VISION_AVAILABLE:
            return ""

        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Prompt for text extraction
            prompt = raw_data.get(
                'vision_prompt',
                "Extract all visible text from this image. Return only the text content."
            )

            # Call Ollama vision model
            response = ollama.chat(
                model=self.config.vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [img_base64]
                }]
            )

            return response['message']['content'].strip()

        except Exception as e:
            logger.error(f"Ollama vision OCR failed: {e}")
            return ""

    def _extract_image_metadata(self, image) -> Dict[str, Any]:
        """Extract EXIF and other metadata from image."""
        metadata = {
            'width': image.width,
            'height': image.height,
            'format': image.format,
            'mode': image.mode
        }

        # Extract EXIF if available
        if hasattr(image, '_getexif') and image._getexif():
            try:
                exif = image._getexif()
                metadata['exif'] = {k: v for k, v in exif.items() if k < 1000}
            except Exception:
                pass

        return metadata

    def _save_image(self, image, episode: str):
        """Save processed image to storage path."""
        try:
            storage_path = Path(self.config.image_storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)

            image_file = storage_path / f"{episode}.png"
            image.save(image_file)
            logger.info(f"Saved image to {image_file}")
        except Exception as e:
            logger.error(f"Failed to save image: {e}")


@dataclass
class ReceiptSpinnerConfig(ImageSpinnerConfig):
    """Configuration for receipt parsing."""
    extract_items: bool = True  # Parse line items
    extract_totals: bool = True  # Extract total, tax, etc.
    extract_merchant: bool = True  # Extract store/merchant info
    extract_date: bool = True  # Extract transaction date


class GroceryReceiptSpinner(ImageSpinner):
    """
    Specialized spinner for grocery receipts.

    Extracts structured data from receipt images:
    - Merchant/store name
    - Transaction date/time
    - Line items with prices
    - Totals (subtotal, tax, total)
    - Payment method
    """

    def __init__(self, config: ReceiptSpinnerConfig = None):
        if config is None:
            config = ReceiptSpinnerConfig()
        super().__init__(config)

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert receipt image → structured MemoryShards.

        Args:
            raw_data: Image data with optional hints:
                - 'image_path', 'image_data', or 'url': Image source
                - 'expected_merchant': Hint for merchant name
                - 'expected_date': Hint for transaction date
                - 'category': Receipt category (grocery, hardware, etc.)

        Returns:
            MemoryShards with structured receipt data
        """
        # Get base OCR text
        shards = await super().spin(raw_data)
        if not shards:
            return shards

        # Parse receipt structure
        shard = shards[0]
        receipt_data = self._parse_receipt(shard.text, raw_data)

        # Update shard with parsed data
        shard.metadata['receipt'] = receipt_data
        shard.entities = receipt_data.get('items', [])
        shard.motifs = ['receipt', 'purchase', receipt_data.get('merchant', 'unknown')]

        # Create structured summary
        shard.text = self._format_receipt_summary(receipt_data, shard.text)

        return shards

    def _parse_receipt(self, ocr_text: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse receipt structure from OCR text."""
        import re
        from datetime import datetime

        receipt_data = {
            'merchant': None,
            'date': None,
            'items': [],
            'subtotal': None,
            'tax': None,
            'total': None,
            'payment_method': None
        }

        lines = ocr_text.split('\n')

        # Extract merchant (usually first non-empty line)
        if self.config.extract_merchant:
            for line in lines[:5]:
                if line.strip():
                    receipt_data['merchant'] = line.strip()
                    break

        # Extract date
        if self.config.extract_date:
            date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            for line in lines[:10]:
                match = re.search(date_pattern, line)
                if match:
                    receipt_data['date'] = match.group()
                    break

        # Extract line items (simple pattern: text followed by price)
        if self.config.extract_items:
            item_pattern = r'(.+?)\s+\$?(\d+\.\d{2})'
            for line in lines:
                match = re.search(item_pattern, line)
                if match and not any(x in line.lower() for x in ['total', 'tax', 'subtotal']):
                    item_name = match.group(1).strip()
                    price = float(match.group(2))
                    receipt_data['items'].append({
                        'name': item_name,
                        'price': price
                    })

        # Extract totals
        if self.config.extract_totals:
            for line in lines:
                lower_line = line.lower()
                price_match = re.search(r'\$?(\d+\.\d{2})', line)

                if price_match:
                    amount = float(price_match.group(1))
                    if 'subtotal' in lower_line:
                        receipt_data['subtotal'] = amount
                    elif 'tax' in lower_line:
                        receipt_data['tax'] = amount
                    elif 'total' in lower_line and 'subtotal' not in lower_line:
                        receipt_data['total'] = amount

        return receipt_data

    def _format_receipt_summary(self, receipt_data: Dict[str, Any], ocr_text: str) -> str:
        """Format structured receipt data as readable summary."""
        lines = []

        if receipt_data['merchant']:
            lines.append(f"Merchant: {receipt_data['merchant']}")
        if receipt_data['date']:
            lines.append(f"Date: {receipt_data['date']}")

        if receipt_data['items']:
            lines.append(f"\nItems ({len(receipt_data['items'])}):")
            for item in receipt_data['items'][:10]:  # Limit to first 10
                lines.append(f"  - {item['name']}: ${item['price']:.2f}")

        if receipt_data['total']:
            lines.append(f"\nTotal: ${receipt_data['total']:.2f}")
        if receipt_data['tax']:
            lines.append(f"Tax: ${receipt_data['tax']:.2f}")

        lines.append("\n--- Original OCR Text ---")
        lines.append(ocr_text)

        return '\n'.join(lines)


# Convenience factory functions
async def process_image(image_path: str, **kwargs) -> List[MemoryShard]:
    """Quick helper to process an image file."""
    config = ImageSpinnerConfig(**kwargs)
    spinner = ImageSpinner(config)
    return await spinner.spin({'image_path': image_path})


async def process_receipt(image_path: str, **kwargs) -> List[MemoryShard]:
    """Quick helper to process a receipt image."""
    config = ReceiptSpinnerConfig(**kwargs)
    spinner = GroceryReceiptSpinner(config)
    return await spinner.spin({'image_path': image_path})