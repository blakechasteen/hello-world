"""
Marker Detector - ArUco and QR code detection for AR tracking

Detects and decodes fiducial markers for precise spatial anchoring.
Provides 6-DOF pose estimation (position + rotation).

Supported Markers:
    - ArUco: 4x4, 5x5, 6x6, 7x7 dictionaries
    - QR Codes: Standard QR code detection and decoding
    - AprilTag: April Robotics fiducial markers

Use Cases:
    - Precise AR object anchoring
    - World-scale tracking (multiple markers)
    - Object identification (marker IDs)
    - Camera calibration
    - Multi-user AR synchronization

Created: 2025-11-22 (Phase 4)
"""

from typing import List, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass

from HoloLoom.vision.protocol import MarkerDetectorProtocol, Marker

logger = logging.getLogger(__name__)


# ============================================================================
# Marker Types
# ============================================================================

class MarkerType:
    """Supported marker types"""
    ARUCO = "aruco"
    QR_CODE = "qr_code"
    APRILTAG = "apriltag"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MarkerDetectorConfig:
    """Configuration for marker detection"""
    marker_type: str = MarkerType.ARUCO
    aruco_dict: str = "DICT_4X4_50"  # ArUco dictionary
    marker_size_mm: float = 100.0  # Physical marker size in mm
    enable_pose: bool = True  # Estimate 6-DOF pose
    min_marker_perimeter: int = 80  # Minimum marker perimeter in pixels


# ============================================================================
# Marker Detector
# ============================================================================

class MarkerDetector(MarkerDetectorProtocol):
    """
    Fiducial marker detection and pose estimation.

    Supports ArUco, QR codes, and AprilTags.
    """

    def __init__(self, config: MarkerDetectorConfig = MarkerDetectorConfig()):
        self.config = config
        self.aruco_dict = None
        self.aruco_params = None
        self.qr_detector = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize marker detector"""
        try:
            if self.config.marker_type == MarkerType.ARUCO:
                return await self._initialize_aruco()
            elif self.config.marker_type == MarkerType.QR_CODE:
                return await self._initialize_qr()
            elif self.config.marker_type == MarkerType.APRILTAG:
                return await self._initialize_apriltag()
            else:
                logger.error(f"Unknown marker type: {self.config.marker_type}")
                return False

        except Exception as e:
            logger.error(f"Marker detector initialization failed: {e}")
            return False

    async def _initialize_aruco(self) -> bool:
        """Initialize ArUco marker detection"""
        try:
            import cv2

            # Get ArUco dictionary
            dict_name = self.config.aruco_dict
            if hasattr(cv2.aruco, dict_name):
                aruco_dict_id = getattr(cv2.aruco, dict_name)
                self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
            else:
                logger.error(f"Unknown ArUco dictionary: {dict_name}")
                return False

            # Create detector parameters
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_params.minMarkerPerimeterRate = self.config.min_marker_perimeter / 1000.0

            self.initialized = True
            logger.info(f"✅ ArUco marker detector initialized (dict={dict_name})")
            return True

        except ImportError:
            logger.warning("opencv-python not installed. Install with: pip install opencv-python opencv-contrib-python")
            return False
        except Exception as e:
            logger.error(f"ArUco initialization failed: {e}")
            return False

    async def _initialize_qr(self) -> bool:
        """Initialize QR code detector"""
        try:
            import cv2

            # Create QR code detector
            self.qr_detector = cv2.QRCodeDetector()

            self.initialized = True
            logger.info("✅ QR code detector initialized")
            return True

        except ImportError:
            logger.warning("opencv-python not installed")
            return False
        except Exception as e:
            logger.error(f"QR detector initialization failed: {e}")
            return False

    async def _initialize_apriltag(self) -> bool:
        """Initialize AprilTag detector"""
        try:
            from pupil_apriltags import Detector

            self.apriltag_detector = Detector()

            self.initialized = True
            logger.info("✅ AprilTag detector initialized")
            return True

        except ImportError:
            logger.warning("pupil-apriltags not installed. Install with: pip install pupil-apriltags")
            return False
        except Exception as e:
            logger.error(f"AprilTag initialization failed: {e}")
            return False

    def set_camera_calibration(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """
        Set camera calibration for pose estimation.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        logger.info("Camera calibration set for pose estimation")

    async def detect_markers(self, frame: np.ndarray) -> List[Marker]:
        """
        Detect markers in frame.

        Args:
            frame: RGB or grayscale image

        Returns:
            List of detected markers with poses
        """
        if not self.initialized:
            logger.warning("Marker detector not initialized")
            return []

        if self.config.marker_type == MarkerType.ARUCO:
            return await self._detect_aruco(frame)
        elif self.config.marker_type == MarkerType.QR_CODE:
            return await self._detect_qr(frame)
        elif self.config.marker_type == MarkerType.APRILTAG:
            return await self._detect_apriltag(frame)
        else:
            return []

    async def _detect_aruco(self, frame: np.ndarray) -> List[Marker]:
        """Detect ArUco markers"""
        try:
            import cv2

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )

            markers = []

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    corner = corners[i][0]  # 4x2 array of corner points

                    # Calculate center
                    center_x = np.mean(corner[:, 0])
                    center_y = np.mean(corner[:, 1])

                    # Estimate pose if camera calibration available
                    position = None
                    rotation = None

                    if self.config.enable_pose and self.camera_matrix is not None:
                        marker_size = self.config.marker_size_mm / 1000.0  # Convert to meters

                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[i],
                            marker_size,
                            self.camera_matrix,
                            self.dist_coeffs
                        )

                        position = tvec[0].flatten().tolist()
                        rotation = rvec[0].flatten().tolist()

                    # Create marker
                    marker = Marker(
                        id=str(marker_id),
                        marker_type=MarkerType.ARUCO,
                        corners=corner.tolist(),
                        center=(float(center_x), float(center_y)),
                        position=position,
                        rotation=rotation,
                        confidence=1.0,  # ArUco doesn't provide confidence
                    )

                    markers.append(marker)

            return markers

        except Exception as e:
            logger.error(f"ArUco detection failed: {e}")
            return []

    async def _detect_qr(self, frame: np.ndarray) -> List[Marker]:
        """Detect QR codes"""
        try:
            import cv2

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Detect QR codes
            retval, decoded_info, points, straight_qrcode = self.qr_detector.detectAndDecodeMulti(gray)

            markers = []

            if retval and points is not None:
                for i, pts in enumerate(points):
                    if pts is not None and len(pts) == 4:
                        corner = pts.reshape(4, 2)

                        # Calculate center
                        center_x = np.mean(corner[:, 0])
                        center_y = np.mean(corner[:, 1])

                        # Get decoded data
                        data = decoded_info[i] if i < len(decoded_info) else ""

                        marker = Marker(
                            id=data,  # Use decoded data as ID
                            marker_type=MarkerType.QR_CODE,
                            corners=corner.tolist(),
                            center=(float(center_x), float(center_y)),
                            data=data,
                            confidence=1.0,
                        )

                        markers.append(marker)

            return markers

        except Exception as e:
            logger.error(f"QR detection failed: {e}")
            return []

    async def _detect_apriltag(self, frame: np.ndarray) -> List[Marker]:
        """Detect AprilTags"""
        try:
            import cv2

            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Detect tags
            results = self.apriltag_detector.detect(
                gray,
                estimate_tag_pose=self.config.enable_pose,
                camera_params=(
                    [
                        self.camera_matrix[0, 0],  # fx
                        self.camera_matrix[1, 1],  # fy
                        self.camera_matrix[0, 2],  # cx
                        self.camera_matrix[1, 2],  # cy
                    ]
                    if self.camera_matrix is not None
                    else None
                ),
                tag_size=self.config.marker_size_mm / 1000.0,
            )

            markers = []

            for tag in results:
                # Extract corners
                corners = tag.corners.tolist()
                center = tag.center.tolist()

                # Extract pose if available
                position = None
                rotation = None

                if tag.pose_t is not None and tag.pose_R is not None:
                    position = tag.pose_t.flatten().tolist()
                    # Convert rotation matrix to Rodrigues vector
                    rvec, _ = cv2.Rodrigues(tag.pose_R)
                    rotation = rvec.flatten().tolist()

                marker = Marker(
                    id=str(tag.tag_id),
                    marker_type=MarkerType.APRILTAG,
                    corners=corners,
                    center=(float(center[0]), float(center[1])),
                    position=position,
                    rotation=rotation,
                    confidence=float(tag.decision_margin),
                )

                markers.append(marker)

            return markers

        except Exception as e:
            logger.error(f"AprilTag detection failed: {e}")
            return []

    async def process_frame(self, frame: np.ndarray) -> List[Marker]:
        """Process frame (delegates to detect_markers)"""
        return await self.detect_markers(frame)

    async def cleanup(self) -> None:
        """Clean up resources"""
        self.aruco_dict = None
        self.aruco_params = None
        self.qr_detector = None
        self.initialized = False
        logger.info("Marker detector cleaned up")

    def get_capabilities(self) -> dict:
        """Get detector capabilities"""
        return {
            "marker_detection": True,
            "marker_type": self.config.marker_type,
            "pose_estimation": self.config.enable_pose,
            "camera_calibrated": self.camera_matrix is not None,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_camera_matrix(width: int, height: int) -> np.ndarray:
    """
    Create default camera matrix (simplified pinhole model).

    Args:
        width: Image width
        height: Image height

    Returns:
        3x3 camera matrix
    """
    # Assume focal length = width (typical for many cameras)
    fx = fy = width
    cx = width / 2
    cy = height / 2

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def marker_to_transform_matrix(marker: Marker) -> Optional[np.ndarray]:
    """
    Convert marker pose to 4x4 transformation matrix.

    Args:
        marker: Marker with position and rotation

    Returns:
        4x4 transformation matrix or None
    """
    if marker.position is None or marker.rotation is None:
        return None

    try:
        import cv2

        # Convert Rodrigues vector to rotation matrix
        rvec = np.array(marker.rotation, dtype=np.float32)
        R, _ = cv2.Rodrigues(rvec)

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = marker.position

        return T

    except Exception as e:
        logger.error(f"Transform matrix conversion failed: {e}")
        return None


# ============================================================================
# Factory
# ============================================================================

def create_marker_detector(
    marker_type: str = MarkerType.ARUCO,
    aruco_dict: str = "DICT_4X4_50",
    marker_size_mm: float = 100.0,
) -> MarkerDetector:
    """
    Factory for creating marker detectors.

    Args:
        marker_type: Marker type (aruco, qr_code, apriltag)
        aruco_dict: ArUco dictionary name
        marker_size_mm: Physical marker size in millimeters

    Returns:
        MarkerDetector instance
    """
    config = MarkerDetectorConfig(
        marker_type=marker_type,
        aruco_dict=aruco_dict,
        marker_size_mm=marker_size_mm,
    )
    return MarkerDetector(config)
