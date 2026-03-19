"""
Depth Estimator - Monocular depth estimation for AR applications

Provides depth maps from single RGB images using deep learning models.
Enables accurate 3D object placement and spatial understanding.

Supported Models:
    - MiDaS (default): Robust general-purpose depth estimation
    - ZoeDepth: High-quality depth for indoor/outdoor scenes
    - Mock: Testing backend (no dependencies)

Use Cases:
    - Accurate 3D object placement in AR
    - Occlusion handling (render objects behind real surfaces)
    - Distance measurement to detected objects
    - Spatial mesh generation

Created: 2025-11-22 (Phase 4)
"""

import logging
from dataclasses import dataclass

import numpy as np

from hololoom.vision.protocol import DepthEstimatorProtocol, DepthMap

logger = logging.getLogger(__name__)


# ============================================================================
# Depth Estimation Models
# ============================================================================

@dataclass
class DepthEstimatorConfig:
    """Configuration for depth estimator"""
    model: str = "midas_small"  # midas_small, midas_v2_1, zoedepth
    device: str = "auto"  # auto, cpu, cuda
    normalize: bool = True  # Normalize depth to 0-1 range
    colormap: str | None = None  # Visualization colormap (viridis, plasma, etc.)


class DepthEstimator(DepthEstimatorProtocol):
    """
    Monocular depth estimation using MiDaS or ZoeDepth.

    MiDaS models:
        - midas_small: Fast, 10.5MB, ~30ms inference (GPU)
        - midas_v2_1: Accurate, 105MB, ~100ms inference (GPU)

    ZoeDepth:
        - zoedepth: Metric depth (meters), ~100ms inference (GPU)
    """

    def __init__(self, config: DepthEstimatorConfig = DepthEstimatorConfig()):
        self.config = config
        self.model = None
        self.transform = None
        self.device = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize depth estimation model"""
        try:
            if self.config.model.startswith("midas"):
                return await self._initialize_midas()
            elif self.config.model == "zoedepth":
                return await self._initialize_zoedepth()
            else:
                logger.warning(f"Unknown model: {self.config.model}, using MiDaS small")
                self.config.model = "midas_small"
                return await self._initialize_midas()

        except Exception as e:
            logger.error(f"Failed to initialize depth estimator: {e}")
            return False

    async def _initialize_midas(self) -> bool:
        """Initialize MiDaS depth estimation"""
        try:
            import torch

            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            # Load model
            if self.config.model == "midas_small":
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            elif self.config.model == "midas_v2_1":
                self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            else:
                logger.error(f"Unknown MiDaS model: {self.config.model}")
                return False

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if self.config.model == "midas_small":
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform

            # Move to device
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            logger.info(f"✅ MiDaS depth estimator initialized (model={self.config.model}, device={self.device})")
            return True

        except ImportError:
            logger.warning("torch not installed. Install with: pip install torch torchvision")
            return False
        except Exception as e:
            logger.error(f"MiDaS initialization failed: {e}")
            return False

    async def _initialize_zoedepth(self) -> bool:
        """Initialize ZoeDepth model"""
        try:
            import torch

            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            # Load ZoeDepth model
            repo = "isl-org/ZoeDepth"
            self.model = torch.hub.load(repo, "ZoeD_N", pretrained=True)
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            logger.info(f"✅ ZoeDepth initialized (device={self.device})")
            return True

        except ImportError:
            logger.warning("torch not installed for ZoeDepth")
            return False
        except Exception as e:
            logger.error(f"ZoeDepth initialization failed: {e}")
            return False

    async def estimate_depth(self, frame: np.ndarray) -> DepthMap:
        """
        Estimate depth map from RGB frame.

        Args:
            frame: RGB image (HxWx3 numpy array)

        Returns:
            DepthMap with depth values and metadata
        """
        if not self.initialized:
            logger.warning("Depth estimator not initialized, using mock")
            return self._mock_depth_map(frame)

        try:
            import torch

            # Convert to tensor
            input_batch = self.transform(frame).to(self.device)

            # Inference
            with torch.no_grad():
                prediction = self.model(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth = prediction.cpu().numpy()

            # Normalize if requested
            if self.config.normalize:
                depth = (depth - depth.min()) / (depth.max() - depth.min())

            # Create depth map
            return DepthMap(
                depth=depth,
                width=frame.shape[1],
                height=frame.shape[0],
                min_depth=float(depth.min()),
                max_depth=float(depth.max()),
                unit="normalized" if self.config.normalize else "disparity",
            )

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return self._mock_depth_map(frame)

    def _mock_depth_map(self, frame: np.ndarray) -> DepthMap:
        """Generate mock depth map for testing"""
        height, width = frame.shape[:2]

        # Create simple gradient depth (closer at bottom, farther at top)
        y_coords = np.linspace(1.0, 0.0, height)
        depth = np.repeat(y_coords[:, np.newaxis], width, axis=1)

        # Add some noise for realism
        noise = np.random.normal(0, 0.05, depth.shape)
        depth = np.clip(depth + noise, 0.0, 1.0)

        return DepthMap(
            depth=depth,
            width=width,
            height=height,
            min_depth=0.0,
            max_depth=1.0,
            unit="normalized",
        )

    async def process_frame(self, frame: np.ndarray) -> DepthMap:
        """Process frame (delegates to estimate_depth)"""
        return await self.estimate_depth(frame)

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
        self.initialized = False
        logger.info("Depth estimator cleaned up")

    def get_capabilities(self) -> dict:
        """Get estimator capabilities"""
        return {
            "depth_estimation": True,
            "model": self.config.model,
            "device": self.device if self.device else "unknown",
            "metric_depth": self.config.model == "zoedepth",
            "normalized": self.config.normalize,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def get_depth_at_point(depth_map: DepthMap, x: int, y: int) -> float | None:
    """
    Get depth value at specific pixel coordinates.

    Args:
        depth_map: DepthMap object
        x, y: Pixel coordinates

    Returns:
        Depth value or None if out of bounds
    """
    if 0 <= x < depth_map.width and 0 <= y < depth_map.height:
        return float(depth_map.depth[y, x])
    return None


def depth_to_3d_point(
    depth_map: DepthMap,
    x: int,
    y: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float] | None:
    """
    Convert 2D pixel + depth to 3D point using camera intrinsics.

    Args:
        depth_map: DepthMap object
        x, y: Pixel coordinates
        fx, fy: Focal lengths
        cx, cy: Principal point

    Returns:
        (X, Y, Z) in camera coordinates or None
    """
    depth = get_depth_at_point(depth_map, x, y)
    if depth is None:
        return None

    # Back-project to 3D
    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return (X, Y, Z)


def create_point_cloud(
    depth_map: DepthMap,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    subsample: int = 4,
) -> np.ndarray:
    """
    Create 3D point cloud from depth map.

    Args:
        depth_map: DepthMap object
        fx, fy: Focal lengths
        cx, cy: Principal point
        subsample: Subsample factor (e.g., 4 = every 4th pixel)

    Returns:
        Nx3 numpy array of 3D points
    """
    points = []

    for y in range(0, depth_map.height, subsample):
        for x in range(0, depth_map.width, subsample):
            point = depth_to_3d_point(depth_map, x, y, fx, fy, cx, cy)
            if point is not None:
                points.append(point)

    return np.array(points)


def visualize_depth(depth_map: DepthMap, colormap: str = "viridis") -> np.ndarray:
    """
    Visualize depth map as colored image.

    Args:
        depth_map: DepthMap object
        colormap: Matplotlib colormap name

    Returns:
        RGB image (HxWx3 numpy array)
    """
    try:
        import matplotlib.pyplot as plt

        # Normalize depth to 0-1
        depth_norm = (depth_map.depth - depth_map.min_depth) / (
            depth_map.max_depth - depth_map.min_depth
        )

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(depth_norm)[:, :, :3]  # Drop alpha channel

        # Convert to uint8
        colored = (colored * 255).astype(np.uint8)

        return colored

    except ImportError:
        logger.warning("matplotlib not available for depth visualization")
        # Fallback: grayscale
        depth_norm = (depth_map.depth - depth_map.min_depth) / (
            depth_map.max_depth - depth_map.min_depth
        )
        grayscale = (depth_norm * 255).astype(np.uint8)
        return np.stack([grayscale] * 3, axis=-1)


# ============================================================================
# Factory
# ============================================================================

def create_depth_estimator(
    model: str = "midas_small",
    device: str = "auto",
    normalize: bool = True,
) -> DepthEstimator:
    """
    Factory for creating depth estimators.

    Args:
        model: Model name (midas_small, midas_v2_1, zoedepth)
        device: Device (auto, cpu, cuda)
        normalize: Normalize depth to 0-1 range

    Returns:
        DepthEstimator instance
    """
    config = DepthEstimatorConfig(
        model=model,
        device=device,
        normalize=normalize,
    )
    return DepthEstimator(config)
