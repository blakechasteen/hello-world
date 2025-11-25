# Skill: Stable Diffusion

## Metadata

- **Name**: `stable_diffusion`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `creative`
- **Tags**: `ai, image, generation, art, diffusion, ml`

## Description

**Short Description**:
AI-powered image generation with Stable Diffusion models for text-to-image and image manipulation.

**Detailed Description**:
The Stable Diffusion skill provides comprehensive AI image generation capabilities using state-of-the-art diffusion models. Generate images from text prompts (txt2img), transform existing images (img2img), perform intelligent inpainting, apply ControlNet guidance for precise control, and upscale images with AI enhancement. Supports multiple model checkpoints (SD 1.5, SD 2.1, SDXL), negative prompts, CFG scale control, and custom samplers. Ideal for creative workflows, asset generation, prototyping, and visual content creation.

## Required Capabilities

Check all capabilities this skill requires:

- [ ] File system access (read)
- [x] File system access (write)
- [x] Code execution (bash)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `diffusers` (Hugging Face diffusion models library)
- `torch` or `tensorflow` (deep learning framework)
- `transformers` (for CLIP text encoder)
- `accelerate` (for GPU optimization)
- GPU with at least 8GB VRAM (recommended for SDXL)

**HoloLoom Integration**: Integrates with creative workflows, asset generation pipelines, visual prototyping, and content creation systems.

## Input Schema

```json
{
  "operation": "string - txt2img|img2img|inpaint|controlnet|upscale",
  "parameters": {
    "prompt": "string (required for txt2img, img2img, inpaint, controlnet) - Text description",
    "negative_prompt": "string (optional) - What to avoid in generation",
    "image_path": "string (required for img2img, inpaint, controlnet, upscale) - Input image",
    "mask_path": "string (required for inpaint) - Mask image (white = inpaint)",
    "width": "number (optional) - Output width in pixels (default: 512)",
    "height": "number (optional) - Output height in pixels (default: 512)",
    "num_inference_steps": "number (optional) - Denoising steps (default: 50)",
    "guidance_scale": "number (optional) - CFG scale (default: 7.5, range: 1-20)",
    "strength": "number (optional for img2img) - Transformation strength (default: 0.8)",
    "seed": "number (optional) - Random seed for reproducibility",
    "sampler": "string (optional) - Sampler: euler|ddim|pndm|lms (default: euler)",
    "model": "string (optional) - Model checkpoint: sd15|sd21|sdxl (default: sd15)",
    "output_path": "string (optional) - Output file path",
    "controlnet_type": "string (optional for controlnet) - canny|depth|pose|mlsd"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Generated image details",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "prompt": "string - Text prompt used",
    "negative_prompt": "string - Negative prompt used",
    "image_path": "string - Output image file path",
    "width": "number - Image width in pixels",
    "height": "number - Image height in pixels",
    "num_inference_steps": "number - Denoising steps used",
    "guidance_scale": "number - CFG scale used",
    "seed": "number - Random seed used",
    "model": "string - Model checkpoint used",
    "file_size_kb": "number - Output file size"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Text-to-Image Generation

**Input**:
```json
{
  "operation": "txt2img",
  "parameters": {
    "prompt": "A majestic mountain landscape at sunset, highly detailed, 8k, photorealistic",
    "negative_prompt": "blurry, low quality, distorted, ugly",
    "width": 768,
    "height": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42,
    "output_path": "generated/mountain_sunset.png"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "txt2img",
    "prompt": "A majestic mountain landscape at sunset...",
    "image_path": "generated/mountain_sunset.png",
    "width": 768,
    "height": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42,
    "file_size_kb": 345
  },
  "message": "Image generated successfully: 768x512px",
  "execution_time_ms": 8500
}
```

**Explanation**: Generates a photorealistic mountain landscape from text description. High guidance scale ensures prompt adherence.

### Example 2: Image-to-Image Transformation

**Input**:
```json
{
  "operation": "img2img",
  "parameters": {
    "image_path": "input/sketch.png",
    "prompt": "Professional architectural rendering, modern glass building, blue sky",
    "negative_prompt": "sketch, drawing, low quality",
    "strength": 0.75,
    "num_inference_steps": 50,
    "guidance_scale": 9.0,
    "output_path": "output/render.png"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "img2img",
    "prompt": "Professional architectural rendering...",
    "image_path": "output/render.png",
    "input_image": "input/sketch.png",
    "strength": 0.75,
    "width": 512,
    "height": 512,
    "guidance_scale": 9.0,
    "file_size_kb": 280
  },
  "message": "Image transformation complete",
  "execution_time_ms": 7200
}
```

**Explanation**: Transforms a rough sketch into a polished architectural render. Strength=0.75 allows significant transformation while preserving composition.

### Example 3: Intelligent Inpainting

**Input**:
```json
{
  "operation": "inpaint",
  "parameters": {
    "image_path": "photos/portrait.jpg",
    "mask_path": "masks/background_mask.png",
    "prompt": "Beautiful garden with flowers and trees, sunny day",
    "negative_prompt": "indoor, artificial, studio",
    "num_inference_steps": 50,
    "guidance_scale": 8.0,
    "output_path": "output/portrait_garden.jpg"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "inpaint",
    "prompt": "Beautiful garden with flowers and trees...",
    "image_path": "output/portrait_garden.jpg",
    "mask_path": "masks/background_mask.png",
    "inpainted_region": "background",
    "width": 512,
    "height": 768,
    "file_size_kb": 420
  },
  "message": "Inpainting complete: background replaced",
  "execution_time_ms": 9500
}
```

**Explanation**: Replaces portrait background with AI-generated garden scene while preserving the subject. Mask defines inpainting region.

### Example 4: ControlNet-Guided Generation

**Input**:
```json
{
  "operation": "controlnet",
  "parameters": {
    "image_path": "references/pose.jpg",
    "controlnet_type": "pose",
    "prompt": "Professional dancer in elegant dress, stage lighting, dramatic pose",
    "negative_prompt": "amateur, casual, poor lighting",
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "output_path": "output/dancer.png"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "controlnet",
    "controlnet_type": "pose",
    "prompt": "Professional dancer in elegant dress...",
    "image_path": "output/dancer.png",
    "control_image": "references/pose.jpg",
    "width": 512,
    "height": 768,
    "guidance_scale": 7.5,
    "file_size_kb": 385
  },
  "message": "ControlNet generation complete (pose guidance)",
  "execution_time_ms": 12000
}
```

**Explanation**: Generates image following pose from reference while applying new style and context. ControlNet ensures structural consistency.

### Example 5: AI Upscaling

**Input**:
```json
{
  "operation": "upscale",
  "parameters": {
    "image_path": "lowres/icon_64x64.png",
    "scale_factor": 4,
    "prompt": "High resolution icon, sharp details, clean design",
    "num_inference_steps": 30,
    "output_path": "highres/icon_256x256.png"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "upscale",
    "image_path": "highres/icon_256x256.png",
    "input_resolution": "64x64",
    "output_resolution": "256x256",
    "scale_factor": 4,
    "file_size_kb": 85
  },
  "message": "Image upscaled 4x with AI enhancement",
  "execution_time_ms": 5500
}
```

**Explanation**: Upscales low-resolution icon 4x with AI-enhanced details. Better quality than traditional bicubic upscaling.

## Testing Checklist

- [x] **Functionality**: All 5 operations execute correctly
- [x] **Error Handling**: Graceful handling of CUDA OOM, invalid prompts, corrupt images
- [x] **Security**: No arbitrary code execution, safe file handling
- [x] **Performance**: Operations complete within expected time (<60s)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Diffusers ecosystem documented
- [x] **Edge Cases**: Handles GPU memory limits, long prompts, large images
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom creative pipelines

## Security Considerations

**Potential Risks**:
- **NSFW Content**: Models can generate inappropriate images -> Implement content filtering
- **Copyright**: Generated images may resemble copyrighted works -> Add watermarks, disclosure
- **Resource Exhaustion**: GPU memory leaks -> Monitor VRAM usage, implement limits

**Data Privacy**:
- [x] Does not log prompts or generated images
- [x] Does not send images to external servers
- [x] Does not train models on user inputs

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] GPU access controlled and monitored
- [x] File operations restricted to designated output directories

## Performance Characteristics

- **Expected Latency**: 5000-60000ms (5-60 seconds depending on resolution and steps)
- **Token Usage**: 100-1000 tokens per execution
- **Resource Requirements**: GPU with 8-24GB VRAM, CUDA/ROCm drivers
- **Scalability**: Limited by GPU memory and queue management

**Operation-Specific Latencies**:
- `txt2img`: 5000-15000ms (512x512, 50 steps)
- `img2img`: 5000-12000ms (similar to txt2img)
- `inpaint`: 8000-20000ms (additional mask processing)
- `controlnet`: 10000-25000ms (additional control processing)
- `upscale`: 5000-30000ms (depends on scale factor)

## License

MIT License

## Related Documentation

- **Stable Diffusion**: [stability.ai](https://stability.ai)
- **Diffusers Library**: [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- **ControlNet**: [github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
- **HoloLoom Creative Skills**: [../README.md](../README.md)
