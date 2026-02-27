"""Image Processing Skill - Resize, crop, convert images using PIL"""

import time
import os
from typing import Dict, Any, List, Optional
from hololoom.skills.base import (BaseSkill, SkillInput, SkillOutput, SkillMetadata,
                                   SkillCategory, SkillStatus)


# Check for PIL/Pillow availability
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    pass


class ImageProcessingSkill(BaseSkill):
    """Image processing operations using PIL/Pillow."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="image_processing",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.CREATIVE,
            tags=["image", "pil", "pillow", "resize", "crop", "convert"],
            description_short="Image processing with PIL/Pillow",
            description_long=(
                "Process images using PIL/Pillow library. Supports resize, crop, "
                "rotate, flip, format conversion, filters (blur, sharpen, edge enhance), "
                "color adjustments (brightness, contrast, saturation), and thumbnails. "
                "Works with common image formats: PNG, JPEG, GIF, BMP, TIFF, WebP."
            ),
            requires_code_execution=False,
            requires_filesystem_read=True,
            requires_filesystem_write=True,
            external_dependencies=["Pillow"],
            expected_latency_ms="50-1000ms",
            token_usage="50-200 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = [
            "resize", "crop", "rotate", "flip", "convert_format",
            "apply_filter", "adjust_brightness", "adjust_contrast",
            "thumbnail", "grayscale", "get_info"
        ]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")

        # Validate required parameters
        if "input" not in skill_input.parameters:
            raise ValueError("'input' parameter (input file path) is required")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        if not PIL_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="PIL/Pillow is not installed. Install with: pip install Pillow",
                execution_time_ms=0,
                errors=["PIL/Pillow not available"]
            )

        start_time = time.time()
        try:
            result = self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Image processing '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Image processing failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    def _dispatch(self, operation: str, parameters: Dict[str, Any]):
        if operation == "resize":
            return self._resize(parameters)
        elif operation == "crop":
            return self._crop(parameters)
        elif operation == "rotate":
            return self._rotate(parameters)
        elif operation == "flip":
            return self._flip(parameters)
        elif operation == "convert_format":
            return self._convert_format(parameters)
        elif operation == "apply_filter":
            return self._apply_filter(parameters)
        elif operation == "adjust_brightness":
            return self._adjust_brightness(parameters)
        elif operation == "adjust_contrast":
            return self._adjust_contrast(parameters)
        elif operation == "thumbnail":
            return self._thumbnail(parameters)
        elif operation == "grayscale":
            return self._grayscale(parameters)
        elif operation == "get_info":
            return self._get_info(parameters)

    def _resize(self, params):
        """Resize image to specified dimensions."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        width = params.get("width")
        height = params.get("height")
        maintain_aspect = params.get("maintain_aspect", True)

        img = Image.open(input_path)
        original_size = img.size

        if maintain_aspect:
            # Calculate aspect-preserving dimensions
            if width and not height:
                ratio = width / img.width
                height = int(img.height * ratio)
            elif height and not width:
                ratio = height / img.height
                width = int(img.width * ratio)
            elif width and height:
                # Use thumbnail for aspect-preserving fit
                img.thumbnail((width, height), Image.LANCZOS)
                img.save(output_path)
                return {
                    "operation": "resize",
                    "input": input_path,
                    "output": output_path,
                    "original_size": original_size,
                    "new_size": img.size,
                    "maintain_aspect": True,
                    "success": True
                }

        new_size = (width or img.width, height or img.height)
        resized = img.resize(new_size, Image.LANCZOS)
        resized.save(output_path)

        return {
            "operation": "resize",
            "input": input_path,
            "output": output_path,
            "original_size": original_size,
            "new_size": new_size,
            "maintain_aspect": maintain_aspect,
            "success": True
        }

    def _crop(self, params):
        """Crop image to specified box."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        left = params.get("left", 0)
        top = params.get("top", 0)
        right = params.get("right")
        bottom = params.get("bottom")

        img = Image.open(input_path)

        # If right/bottom not specified, crop from left/top to end
        if right is None:
            right = img.width
        if bottom is None:
            bottom = img.height

        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path)

        return {
            "operation": "crop",
            "input": input_path,
            "output": output_path,
            "original_size": img.size,
            "crop_box": (left, top, right, bottom),
            "new_size": cropped.size,
            "success": True
        }

    def _rotate(self, params):
        """Rotate image by specified degrees."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        degrees = params.get("degrees", 0)
        expand = params.get("expand", True)

        img = Image.open(input_path)
        rotated = img.rotate(degrees, expand=expand)
        rotated.save(output_path)

        return {
            "operation": "rotate",
            "input": input_path,
            "output": output_path,
            "degrees": degrees,
            "original_size": img.size,
            "new_size": rotated.size,
            "success": True
        }

    def _flip(self, params):
        """Flip image horizontally or vertically."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        direction = params.get("direction", "horizontal")  # "horizontal" or "vertical"

        img = Image.open(input_path)

        if direction == "horizontal":
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("direction must be 'horizontal' or 'vertical'")

        flipped.save(output_path)

        return {
            "operation": "flip",
            "input": input_path,
            "output": output_path,
            "direction": direction,
            "success": True
        }

    def _convert_format(self, params):
        """Convert image to different format."""
        input_path = params["input"]
        output_path = params["output"]  # Must specify output for format conversion
        format = params.get("format")  # If not specified, infer from output extension

        img = Image.open(input_path)
        original_format = img.format

        # Infer format from output extension if not specified
        if format is None:
            ext = os.path.splitext(output_path)[1].upper().lstrip('.')
            format = ext

        # Convert RGBA to RGB for JPEG (JPEG doesn't support transparency)
        if format.upper() == 'JPEG' and img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = rgb_img

        img.save(output_path, format=format)

        return {
            "operation": "convert_format",
            "input": input_path,
            "output": output_path,
            "original_format": original_format,
            "new_format": format,
            "success": True
        }

    def _apply_filter(self, params):
        """Apply image filter."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        filter_type = params.get("filter", "blur")  # blur, sharpen, edge_enhance, smooth

        img = Image.open(input_path)

        filter_map = {
            "blur": ImageFilter.BLUR,
            "sharpen": ImageFilter.SHARPEN,
            "edge_enhance": ImageFilter.EDGE_ENHANCE,
            "smooth": ImageFilter.SMOOTH,
            "detail": ImageFilter.DETAIL,
            "contour": ImageFilter.CONTOUR
        }

        if filter_type not in filter_map:
            raise ValueError(f"Invalid filter. Must be one of: {', '.join(filter_map.keys())}")

        filtered = img.filter(filter_map[filter_type])
        filtered.save(output_path)

        return {
            "operation": "apply_filter",
            "input": input_path,
            "output": output_path,
            "filter": filter_type,
            "success": True
        }

    def _adjust_brightness(self, params):
        """Adjust image brightness."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        factor = params.get("factor", 1.0)  # 0.0 = black, 1.0 = original, 2.0 = twice as bright

        img = Image.open(input_path)
        enhancer = ImageEnhance.Brightness(img)
        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path)

        return {
            "operation": "adjust_brightness",
            "input": input_path,
            "output": output_path,
            "factor": factor,
            "success": True
        }

    def _adjust_contrast(self, params):
        """Adjust image contrast."""
        input_path = params["input"]
        output_path = params.get("output", input_path)
        factor = params.get("factor", 1.0)  # 0.0 = gray, 1.0 = original, 2.0 = high contrast

        img = Image.open(input_path)
        enhancer = ImageEnhance.Contrast(img)
        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path)

        return {
            "operation": "adjust_contrast",
            "input": input_path,
            "output": output_path,
            "factor": factor,
            "success": True
        }

    def _thumbnail(self, params):
        """Create thumbnail maintaining aspect ratio."""
        input_path = params["input"]
        output_path = params.get("output", input_path.replace('.', '_thumb.'))
        max_size = params.get("max_size", 128)  # Max width or height

        img = Image.open(input_path)
        original_size = img.size

        img.thumbnail((max_size, max_size), Image.LANCZOS)
        img.save(output_path)

        return {
            "operation": "thumbnail",
            "input": input_path,
            "output": output_path,
            "original_size": original_size,
            "thumbnail_size": img.size,
            "max_size": max_size,
            "success": True
        }

    def _grayscale(self, params):
        """Convert image to grayscale."""
        input_path = params["input"]
        output_path = params.get("output", input_path)

        img = Image.open(input_path)
        gray = ImageOps.grayscale(img)
        gray.save(output_path)

        return {
            "operation": "grayscale",
            "input": input_path,
            "output": output_path,
            "original_mode": img.mode,
            "new_mode": gray.mode,
            "success": True
        }

    def _get_info(self, params):
        """Get image information."""
        input_path = params["input"]

        img = Image.open(input_path)
        file_size = os.path.getsize(input_path)

        return {
            "operation": "get_info",
            "input": input_path,
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2),
            "success": True
        }


from hololoom.skills.base import register_skill
register_skill(ImageProcessingSkill())