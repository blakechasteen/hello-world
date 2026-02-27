"""
Stable Diffusion Backend
========================

Integration with open-source image generation backends:
- ComfyUI (recommended - clean API, flexible workflows)
- AUTOMATIC1111 (popular, extensive features)
- Direct diffusers (local Python execution)

Philosophy: "Graceful degradation - works without dependencies."

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, Any, Optional, List, Protocol, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import base64
import asyncio
from pathlib import Path


@dataclass
class GenerationParams:
    """Parameters for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    sampler: str = "euler"
    seed: int = -1  # -1 = random
    batch_size: int = 1
    model: str = ""  # Empty = use default


@dataclass
class GenerationResult:
    """Result from image generation."""
    success: bool
    images: List[bytes] = field(default_factory=list)
    seeds: List[int] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageBackend(ABC):
    """Abstract base for image generation backends."""

    @abstractmethod
    async def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate images from parameters."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is available."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """List available models."""
        pass


class ComfyUIBackend(ImageBackend):
    """
    ComfyUI backend - uses workflow-based generation.

    Requirements:
    - ComfyUI running locally (default: http://127.0.0.1:8188)
    - API enabled (--listen flag)

    ComfyUI is recommended because:
    - Clean, well-documented API
    - Flexible node-based workflows
    - Good memory management
    - Active development
    """

    DEFAULT_WORKFLOW = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 7.0,
                "denoise": 1.0,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "euler",
                "scheduler": "normal",
                "seed": 0,
                "steps": 20
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": 512,
                "width": 512
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": ""
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": ""
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            }
        }
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        timeout: float = 120.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self._http_client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import aiohttp
                self._http_client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
            except ImportError:
                return None
        return self._http_client

    async def health_check(self) -> bool:
        """Check if ComfyUI is running."""
        client = await self._get_client()
        if not client:
            return False

        try:
            async with client.get(f"{self.base_url}/system_stats") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate images using ComfyUI API."""
        client = await self._get_client()
        if not client:
            return GenerationResult(
                success=False,
                error="aiohttp not installed. Run: pip install aiohttp"
            )

        # Build workflow from template
        workflow = self._build_workflow(params)

        try:
            # Queue the prompt
            async with client.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow}
            ) as resp:
                if resp.status != 200:
                    return GenerationResult(
                        success=False,
                        error=f"ComfyUI error: {await resp.text()}"
                    )
                data = await resp.json()
                prompt_id = data.get("prompt_id")

            # Wait for completion
            images, seeds = await self._wait_for_completion(prompt_id)

            return GenerationResult(
                success=True,
                images=images,
                seeds=seeds,
                metadata={"prompt_id": prompt_id, "backend": "comfyui"}
            )

        except Exception as e:
            return GenerationResult(success=False, error=str(e))

    def _build_workflow(self, params: GenerationParams) -> Dict:
        """Build ComfyUI workflow from parameters."""
        import copy
        workflow = copy.deepcopy(self.DEFAULT_WORKFLOW)

        # Update prompts
        workflow["6"]["inputs"]["text"] = params.prompt
        workflow["7"]["inputs"]["text"] = params.negative_prompt

        # Update sampler settings
        workflow["3"]["inputs"]["steps"] = params.steps
        workflow["3"]["inputs"]["cfg"] = params.cfg_scale
        workflow["3"]["inputs"]["sampler_name"] = params.sampler
        workflow["3"]["inputs"]["seed"] = params.seed if params.seed >= 0 else -1

        # Update image size
        workflow["5"]["inputs"]["width"] = params.width
        workflow["5"]["inputs"]["height"] = params.height
        workflow["5"]["inputs"]["batch_size"] = params.batch_size

        # Update model if specified
        if params.model:
            workflow["4"]["inputs"]["ckpt_name"] = params.model

        return workflow

    async def _wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 0.5
    ) -> Tuple[List[bytes], List[int]]:
        """Wait for generation to complete and fetch images."""
        client = await self._get_client()
        images = []
        seeds = []

        elapsed = 0.0
        while elapsed < self.timeout:
            async with client.get(f"{self.base_url}/history/{prompt_id}") as resp:
                if resp.status == 200:
                    history = await resp.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        for node_id, output in outputs.items():
                            if "images" in output:
                                for img_info in output["images"]:
                                    # Fetch the image
                                    img_data = await self._fetch_image(
                                        img_info["filename"],
                                        img_info.get("subfolder", ""),
                                        img_info.get("type", "output")
                                    )
                                    if img_data:
                                        images.append(img_data)
                        return images, seeds

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return images, seeds

    async def _fetch_image(
        self,
        filename: str,
        subfolder: str,
        folder_type: str
    ) -> Optional[bytes]:
        """Fetch generated image from ComfyUI."""
        client = await self._get_client()
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }

        try:
            async with client.get(
                f"{self.base_url}/view",
                params=params
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception:
            pass
        return None

    def get_available_models(self) -> List[str]:
        """List available checkpoint models."""
        # This would need to query ComfyUI's object_info endpoint
        return []

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None


class A1111Backend(ImageBackend):
    """
    AUTOMATIC1111 backend - uses the txt2img API.

    Requirements:
    - AUTOMATIC1111 WebUI running with --api flag
    - Default: http://127.0.0.1:7860

    A1111 is popular because:
    - Extensive feature set
    - Large community and extension ecosystem
    - Good ControlNet support
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        timeout: float = 120.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self._http_client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import aiohttp
                self._http_client = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
            except ImportError:
                return None
        return self._http_client

    async def health_check(self) -> bool:
        """Check if A1111 is running."""
        client = await self._get_client()
        if not client:
            return False

        try:
            async with client.get(f"{self.base_url}/sdapi/v1/sd-models") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate images using A1111 API."""
        client = await self._get_client()
        if not client:
            return GenerationResult(
                success=False,
                error="aiohttp not installed. Run: pip install aiohttp"
            )

        payload = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "cfg_scale": params.cfg_scale,
            "sampler_name": self._map_sampler(params.sampler),
            "seed": params.seed,
            "batch_size": params.batch_size,
        }

        try:
            async with client.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload
            ) as resp:
                if resp.status != 200:
                    return GenerationResult(
                        success=False,
                        error=f"A1111 error: {await resp.text()}"
                    )

                data = await resp.json()
                images = []
                for img_b64 in data.get("images", []):
                    images.append(base64.b64decode(img_b64))

                # Extract seeds from info
                info = json.loads(data.get("info", "{}"))
                seeds = info.get("all_seeds", [])

                return GenerationResult(
                    success=True,
                    images=images,
                    seeds=seeds,
                    metadata={"backend": "a1111", "info": info}
                )

        except Exception as e:
            return GenerationResult(success=False, error=str(e))

    def _map_sampler(self, sampler: str) -> str:
        """Map generic sampler name to A1111 sampler."""
        mapping = {
            "euler": "Euler",
            "euler_a": "Euler a",
            "dpm": "DPM++ 2M",
            "dpm_sde": "DPM++ SDE",
            "ddim": "DDIM",
            "lms": "LMS",
        }
        return mapping.get(sampler.lower(), sampler)

    def get_available_models(self) -> List[str]:
        """List available models - requires sync request."""
        return []

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None


class StableDiffusionBackend(ImageBackend):
    """
    Unified Stable Diffusion backend that auto-detects available backends.

    Tries in order:
    1. ComfyUI (if running)
    2. AUTOMATIC1111 (if running)
    3. Falls back to placeholder
    """

    def __init__(
        self,
        comfyui_port: int = 8188,
        a1111_port: int = 7860,
        host: str = "127.0.0.1",
    ):
        self.comfyui = ComfyUIBackend(host=host, port=comfyui_port)
        self.a1111 = A1111Backend(host=host, port=a1111_port)
        self._active_backend: Optional[ImageBackend] = None

    async def health_check(self) -> bool:
        """Check if any backend is available."""
        # Try ComfyUI first
        if await self.comfyui.health_check():
            self._active_backend = self.comfyui
            return True

        # Try A1111
        if await self.a1111.health_check():
            self._active_backend = self.a1111
            return True

        self._active_backend = None
        return False

    async def generate(self, params: GenerationParams) -> GenerationResult:
        """Generate using the active backend."""
        if self._active_backend is None:
            # Try to find a backend
            await self.health_check()

        if self._active_backend is None:
            return GenerationResult(
                success=False,
                error=(
                    "No image generation backend available. "
                    "Please start ComfyUI (port 8188) or AUTOMATIC1111 (port 7860)."
                )
            )

        return await self._active_backend.generate(params)

    def get_available_models(self) -> List[str]:
        """Get models from active backend."""
        if self._active_backend:
            return self._active_backend.get_available_models()
        return []

    async def close(self):
        """Close all backends."""
        await self.comfyui.close()
        await self.a1111.close()


__all__ = [
    'GenerationParams',
    'GenerationResult',
    'ImageBackend',
    'ComfyUIBackend',
    'A1111Backend',
    'StableDiffusionBackend',
]
