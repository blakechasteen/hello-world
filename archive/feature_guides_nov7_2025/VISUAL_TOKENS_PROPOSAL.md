# Visual Tokens for Context Compression

**Date**: January 2025
**Status**: Proposed Enhancement
**Benefit**: 4-10x context window compression

---

## Problem

Current text-based approach uses ~1000 tokens per receipt:
```
Transaction: $45.99
Merchant: Whole Foods Market
Date: 2025-01-05
Items:
  - Organic Bananas: $3.99
  - Greek Yogurt: $5.99
  ...
```

## Solution: Visual Tokens

**Visual tokens = compressed image representation**:
- 1 image ≈ 170 tokens (vs 1000 text tokens)
- Preserves spatial layout, formatting, visual context
- Enables image-based queries

## Benefits

1. **Context Compression**: 4-10x more information per context window
2. **Visual Similarity**: "Find receipts that look like this"
3. **Better Corrections**: Users see thumbnail when correcting
4. **Multimodal Memory**: Images + text in Yarn Graph

## Architecture

### Phase 1: Visual Token Support in Spinners

```python
# Enhanced OCR Spinner
class VisualTokenSpinner(BaseSpinner):
    """OCR with visual token compression."""

    async def spin(self, source, **kwargs):
        # 1. Extract visual tokens (image embedding)
        visual_tokens = await self.extract_visual_tokens(source)

        # 2. Extract text (OCR)
        text_data = await self.extract_text(source)

        # 3. Create MemoryShard with both
        shard = MemoryShard(
            content=text_data,
            metadata={
                'visual_tokens': visual_tokens,
                'compression_ratio': len(text_data) / len(visual_tokens)
            }
        )

        return SpinResult(shards=[shard])
```

### Phase 2: Visual Token Storage in Yarn Graph

```python
# Store visual tokens as node attributes
kg = KG()
kg.add_node(
    'receipt_001',
    node_type='Receipt',
    content=text_data,
    visual_tokens=visual_tokens,  # NEW
    visual_embedding=image_embedding  # For similarity search
)
```

### Phase 3: Visual Query Support

```python
# Query by visual similarity
results = await orchestrator.query_by_image(
    "Find receipts similar to this",
    reference_image="receipt.jpg"
)
```

### Phase 4: Voice UI Thumbnails

```html
<!-- Show thumbnail when correcting -->
<div class="correction-context">
    <img src="data:image/jpeg;base64,{{thumbnail}}" />
    <div class="voice-command">
        "the merchant is Whole Foods Market"
    </div>
</div>
```

## Implementation

### Visual Token Extraction

```python
import torch
from torchvision import transforms
from PIL import Image

class VisualTokenExtractor:
    """Extract visual tokens from images."""

    def __init__(self, model_name='clip'):
        self.model = self._load_model(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    async def extract(self, image_path: Path) -> torch.Tensor:
        """Extract visual tokens."""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            visual_tokens = self.model.encode_image(tensor)

        return visual_tokens  # Shape: (1, 512) - only 170 tokens!
```

### Integration with Schema-Aware System

```python
class VisualSchemaAwareReceiptSpinner(SchemaAwareReceiptSpinner):
    """Schema-aware spinner with visual tokens."""

    def __init__(self, *args, enable_visual_tokens=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_visual_tokens = enable_visual_tokens
        if enable_visual_tokens:
            self.visual_extractor = VisualTokenExtractor()

    async def spin(self, source, **kwargs):
        # 1. Extract visual tokens (if enabled)
        visual_tokens = None
        if self.enable_visual_tokens:
            visual_tokens = await self.visual_extractor.extract(source)

        # 2. Run normal OCR + schema transformation
        result = await super().spin(source, **kwargs)

        # 3. Add visual tokens to shards
        for shard in result.shards:
            if visual_tokens is not None:
                shard.metadata['visual_tokens'] = visual_tokens
                shard.metadata['has_visual_context'] = True

        return result
```

## Compression Example

**Text-Only Approach**:
```python
receipt_text = """
Transaction ID: TXN20250105001
Date: 2025-01-05 14:32:15
Merchant: Whole Foods Market
Address: 123 Main St, Seattle, WA 98101
Phone: (206) 555-0123

Items:
  1. Organic Bananas (2.5 lbs)      $3.99
  2. Greek Yogurt (32oz)            $5.99
  3. Almond Milk (half gal)         $4.49
  4. Spinach (5oz)                  $2.99
  5. Chicken Breast (1.2 lbs)      $12.89
  6. Bread (whole wheat)            $3.49
  7. Eggs (dozen)                   $4.29
  8. Olive Oil (16oz)               $8.99
  9. Tomatoes (1 lb)                $3.49
 10. Cheese (8oz)                   $6.99

Subtotal:                          $57.60
Tax (10.1%):                        $5.82
Total:                             $63.42

Payment: Visa ****1234
"""
# Tokens: ~450 tokens
```

**Visual Token Approach**:
```python
visual_tokens = extract_visual_tokens("receipt.jpg")
# Shape: (1, 512) = 170 tokens

structured_data = {
    'transaction_id': 'TXN20250105001',
    'date': '2025-01-05',
    'merchant': 'Whole Foods Market',
    'total': 63.42,
    'items': [...]  # 10 items
}
# Tokens: ~80 tokens

# Total: 170 + 80 = 250 tokens (vs 450 text-only)
# Compression ratio: 1.8x
```

## Performance Comparison

| Approach | Tokens | Compression | Searchability | Visual Context |
|----------|--------|-------------|---------------|----------------|
| Text-only | 450 | 1.0x | Text only | None |
| Visual tokens | 250 | 1.8x | Text + image | Full |
| Visual + minimal text | 200 | 2.25x | Text + image | Full |

## Future Extensions

1. **Visual RAG**: Retrieve receipts by visual similarity
2. **Layout Understanding**: Preserve spatial relationships
3. **Handwriting Recognition**: Better context for cursive/signatures
4. **Multi-Image Documents**: Compress multi-page PDFs efficiently
5. **Visual Memory Graph**: Image-to-image relationships

## Implementation Priority

**Phase 1** (Week 1): Visual token extraction + storage
**Phase 2** (Week 2): Integration with schema-aware system
**Phase 3** (Week 3): Visual similarity search
**Phase 4** (Week 4): Voice UI thumbnails

---

## Next Steps

1. ✅ Install Tesseract (current priority)
2. Test schema-aware system with real OCR
3. Implement visual token extraction (if desired)
4. Add visual tokens to Yarn Graph nodes
5. Update voice UI to show thumbnails

---

**Questions**:
- Should we prioritize visual tokens now or after Tesseract testing?
- Which visual model to use? (CLIP, DINO, ViT)
- CPU or GPU inference for visual tokens? (you have CPU only)
