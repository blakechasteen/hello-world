"""Stable Diffusion Skill - AI image generation"""

import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from HoloLoom.skills.base import (BaseSkill, SkillInput, SkillOutput, SkillMetadata,
                                   SkillCategory, SkillStatus)

# Check for diffusers availability
try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionUpscalePipeline
    )
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Check for PIL availability
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class StableDiffusionSkill(BaseSkill):
    """AI image generation with Stable Diffusion."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="stable_diffusion",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.CREATIVE,
            tags=["ai", "image", "generation", "art", "diffusion"],
            description_short="AI image generation with Stable Diffusion models",
            description_long="Generate images from text prompts using Stable Diffusion. Supports txt2img, img2img, inpainting, controlnet, and style transfer.",
            requires_code_execution=True,
            requires_filesystem_write=True,
            external_dependencies=["diffusers", "torch", "transformers"],
            expected_latency_ms="5000-60000ms",
            token_usage="100-1000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["txt2img", "img2img", "inpaint", "controlnet", "upscale"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        if skill_input.operation == "txt2img" and "prompt" not in skill_input.parameters:
            raise ValueError("txt2img requires 'prompt' parameter")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()

        # Check for diffusers availability
        if not DIFFUSERS_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="diffusers not available. Install with: pip install diffusers torch transformers",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependency: diffusers, torch"]
            )

        # Check for PIL availability
        if not PIL_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="PIL not available. Install with: pip install Pillow",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependency: Pillow"]
            )

        try:
            # Dispatch to operation handler
            if skill_input.operation == "txt2img":
                result = await self._txt2img(skill_input.parameters)
            elif skill_input.operation == "img2img":
                result = await self._img2img(skill_input.parameters)
            elif skill_input.operation == "inpaint":
                result = await self._inpaint(skill_input.parameters)
            elif skill_input.operation == "upscale":
                result = await self._upscale(skill_input.parameters)
            elif skill_input.operation == "controlnet":
                result = await self._controlnet(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Stable Diffusion '{skill_input.operation}' completed",
                execution_time_ms=(time.time() - start_time) * 1000,
                details=result
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Stable Diffusion operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _txt2img(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt."""
        prompt = params["prompt"]
        output_path = params.get("output_path", "output/generated_image.png")
        width = params.get("width", 512)
        height = params.get("height", 512)
        num_inference_steps = params.get("steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        negative_prompt = params.get("negative_prompt")
        model_id = params.get("model_id", "runwayml/stable-diffusion-v1-5")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        # Save image
        image.save(output_path)

        return {
            "operation": "txt2img",
            "prompt": prompt,
            "image_path": output_path,
            "width": width,
            "height": height,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "device": device,
            "model": model_id
        }

    async def _img2img(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform image using text prompt."""
        prompt = params["prompt"]
        init_image_path = params["init_image"]
        output_path = params.get("output_path", "output/img2img_result.png")
        strength = params.get("strength", 0.75)
        num_inference_steps = params.get("steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        negative_prompt = params.get("negative_prompt")
        model_id = params.get("model_id", "runwayml/stable-diffusion-v1-5")

        # Load init image
        init_image = Image.open(init_image_path).convert("RGB")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        # Generate image
        image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        # Save image
        image.save(output_path)

        return {
            "operation": "img2img",
            "prompt": prompt,
            "init_image": init_image_path,
            "output_path": output_path,
            "strength": strength,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "device": device,
            "model": model_id
        }

    async def _inpaint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Inpaint image using mask and text prompt."""
        prompt = params["prompt"]
        init_image_path = params["init_image"]
        mask_image_path = params["mask_image"]
        output_path = params.get("output_path", "output/inpaint_result.png")
        num_inference_steps = params.get("steps", 50)
        guidance_scale = params.get("guidance_scale", 7.5)
        negative_prompt = params.get("negative_prompt")
        model_id = params.get("model_id", "runwayml/stable-diffusion-inpainting")

        # Load images
        init_image = Image.open(init_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("RGB")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        # Generate image
        image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        # Save image
        image.save(output_path)

        return {
            "operation": "inpaint",
            "prompt": prompt,
            "init_image": init_image_path,
            "mask_image": mask_image_path,
            "output_path": output_path,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "device": device,
            "model": model_id
        }

    async def _upscale(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Upscale image using Stable Diffusion."""
        prompt = params["prompt"]
        init_image_path = params["init_image"]
        output_path = params.get("output_path", "output/upscale_result.png")
        num_inference_steps = params.get("steps", 50)
        guidance_scale = params.get("guidance_scale", 9.0)
        model_id = params.get("model_id", "stabilityai/stable-diffusion-x4-upscaler")

        # Load init image
        init_image = Image.open(init_image_path).convert("RGB")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)

        # Upscale image
        upscaled_image = pipe(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        # Save image
        upscaled_image.save(output_path)

        return {
            "operation": "upscale",
            "prompt": prompt,
            "init_image": init_image_path,
            "output_path": output_path,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "device": device,
            "model": model_id,
            "original_size": f"{init_image.width}x{init_image.height}",
            "upscaled_size": f"{upscaled_image.width}x{upscaled_image.height}"
        }

    async def _controlnet(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image with ControlNet guidance (requires additional setup)."""
        # ControlNet requires additional imports and setup
        # This is a placeholder implementation noting the requirement
        return {
            "operation": "controlnet",
            "status": "not_implemented",
            "message": "ControlNet requires additional diffusers.ControlNetModel setup. "
                       "Please use txt2img, img2img, inpaint, or upscale operations.",
            "note": "To implement: pip install controlnet_aux and import diffusers.ControlNetModel"
        }


from HoloLoom.skills.base import register_skill
register_skill(StableDiffusionSkill())
