"""
Shared Math Types for HoloLoom Spatial Computing.

Core mathematical primitives used across all spatial modules:
- Vector3: 3D position/direction
- Quaternion: Rotation
- Color: RGBA color
- Transform: Position + Rotation + Scale

Created: November 2025
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Vector3:
    """3D vector for position, direction, and scale."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def magnitude(self) -> float:
        """Length of the vector."""
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def magnitude_squared(self) -> float:
        """Squared length (faster, avoids sqrt)."""
        return self.x*self.x + self.y*self.y + self.z*self.z

    def normalized(self) -> "Vector3":
        """Unit vector in same direction."""
        mag = self.magnitude()
        if mag > 1e-10:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3()

    def dot(self, other: "Vector3") -> float:
        """Dot product."""
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Cross product."""
        return Vector3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

    def distance_to(self, other: "Vector3") -> float:
        """Distance to another point."""
        return (self - other).magnitude()

    def lerp(self, other: "Vector3", t: float) -> "Vector3":
        """Linear interpolation."""
        return self + (other - self) * t

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float]) -> "Vector3":
        return cls(t[0], t[1], t[2])

    @classmethod
    def zero(cls) -> "Vector3":
        return cls(0, 0, 0)

    @classmethod
    def one(cls) -> "Vector3":
        return cls(1, 1, 1)

    @classmethod
    def up(cls) -> "Vector3":
        return cls(0, 1, 0)

    @classmethod
    def down(cls) -> "Vector3":
        return cls(0, -1, 0)

    @classmethod
    def forward(cls) -> "Vector3":
        return cls(0, 0, 1)

    @classmethod
    def back(cls) -> "Vector3":
        return cls(0, 0, -1)

    @classmethod
    def right(cls) -> "Vector3":
        return cls(1, 0, 0)

    @classmethod
    def left(cls) -> "Vector3":
        return cls(-1, 0, 0)


@dataclass
class Quaternion:
    """Quaternion for 3D rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __repr__(self) -> str:
        return f"Quaternion({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.w:.3f})"

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Quaternion multiplication (rotation composition)."""
        return Quaternion(
            self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w,
            self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        )

    def magnitude(self) -> float:
        """Length of quaternion."""
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w)

    def normalized(self) -> "Quaternion":
        """Unit quaternion."""
        mag = self.magnitude()
        if mag > 1e-10:
            return Quaternion(self.x/mag, self.y/mag, self.z/mag, self.w/mag)
        return Quaternion.identity()

    def conjugate(self) -> "Quaternion":
        """Conjugate (inverse for unit quaternions)."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> "Quaternion":
        """Inverse rotation."""
        mag_sq = self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w
        if mag_sq > 1e-10:
            return Quaternion(-self.x/mag_sq, -self.y/mag_sq, -self.z/mag_sq, self.w/mag_sq)
        return Quaternion.identity()

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate a vector by this quaternion."""
        # q * v * q^-1
        qv = Quaternion(v.x, v.y, v.z, 0)
        result = self * qv * self.conjugate()
        return Vector3(result.x, result.y, result.z)

    def slerp(self, other: "Quaternion", t: float) -> "Quaternion":
        """Spherical linear interpolation."""
        dot = self.x*other.x + self.y*other.y + self.z*other.z + self.w*other.w

        # If dot is negative, negate one quaternion to take shorter path
        if dot < 0:
            other = Quaternion(-other.x, -other.y, -other.z, -other.w)
            dot = -dot

        # If very close, use linear interpolation
        if dot > 0.9995:
            result = Quaternion(
                self.x + t*(other.x - self.x),
                self.y + t*(other.y - self.y),
                self.z + t*(other.z - self.z),
                self.w + t*(other.w - self.w)
            )
            return result.normalized()

        theta = math.acos(dot)
        sin_theta = math.sin(theta)

        s0 = math.sin((1 - t) * theta) / sin_theta
        s1 = math.sin(t * theta) / sin_theta

        return Quaternion(
            s0*self.x + s1*other.x,
            s0*self.y + s1*other.y,
            s0*self.z + s1*other.z,
            s0*self.w + s1*other.w
        )

    def to_euler(self) -> tuple[float, float, float]:
        """Convert to Euler angles (pitch, yaw, roll) in radians."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (pitch, yaw, roll)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(0, 0, 0, 1)

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> "Quaternion":
        """Create from Euler angles (radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return cls(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        )

    @classmethod
    def from_axis_angle(cls, axis: Vector3, angle: float) -> "Quaternion":
        """Create from axis and angle (radians)."""
        axis = axis.normalized()
        half_angle = angle * 0.5
        sin_half = math.sin(half_angle)
        return cls(
            axis.x * sin_half,
            axis.y * sin_half,
            axis.z * sin_half,
            math.cos(half_angle)
        )

    @classmethod
    def look_rotation(cls, forward: Vector3, up: Vector3 = None) -> "Quaternion":
        """Create rotation looking in direction."""
        if up is None:
            up = Vector3.up()

        forward = forward.normalized()
        right = up.cross(forward).normalized()
        up = forward.cross(right)

        # Build rotation matrix, extract quaternion
        m00, m01, m02 = right.x, up.x, forward.x
        m10, m11, m12 = right.y, up.y, forward.y
        m20, m21, m22 = right.z, up.z, forward.z

        trace = m00 + m11 + m22

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            return cls(
                (m21 - m12) * s,
                (m02 - m20) * s,
                (m10 - m01) * s,
                0.25 / s
            )
        elif m00 > m11 and m00 > m22:
            s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
            return cls(
                0.25 * s,
                (m01 + m10) / s,
                (m02 + m20) / s,
                (m21 - m12) / s
            )
        elif m11 > m22:
            s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
            return cls(
                (m01 + m10) / s,
                0.25 * s,
                (m12 + m21) / s,
                (m02 - m20) / s
            )
        else:
            s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
            return cls(
                (m02 + m20) / s,
                (m12 + m21) / s,
                0.25 * s,
                (m10 - m01) / s
            )


@dataclass
class Color:
    """RGBA color (0.0-1.0 range)."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def __repr__(self) -> str:
        return f"Color({self.r:.2f}, {self.g:.2f}, {self.b:.2f}, {self.a:.2f})"

    def to_hex(self) -> str:
        """Convert to hex string (#RRGGBB)."""
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}"

    def to_hex_alpha(self) -> str:
        """Convert to hex string with alpha (#RRGGBBAA)."""
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}{int(self.a*255):02x}"

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    def to_dict(self) -> dict[str, float]:
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}

    def lerp(self, other: "Color", t: float) -> "Color":
        """Linear interpolation between colors."""
        return Color(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t
        )

    def with_alpha(self, alpha: float) -> "Color":
        """Return same color with different alpha."""
        return Color(self.r, self.g, self.b, alpha)

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Create from hex string (#RGB, #RRGGBB, or #RRGGBBAA)."""
        hex_str = hex_str.lstrip('#')

        if len(hex_str) == 3:
            r = int(hex_str[0]*2, 16) / 255
            g = int(hex_str[1]*2, 16) / 255
            b = int(hex_str[2]*2, 16) / 255
            return cls(r, g, b, 1.0)
        elif len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255
            g = int(hex_str[2:4], 16) / 255
            b = int(hex_str[4:6], 16) / 255
            return cls(r, g, b, 1.0)
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16) / 255
            g = int(hex_str[2:4], 16) / 255
            b = int(hex_str[4:6], 16) / 255
            a = int(hex_str[6:8], 16) / 255
            return cls(r, g, b, a)
        else:
            raise ValueError(f"Invalid hex color: {hex_str}")

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0) -> "Color":
        """Create from HSV (h: 0-360, s: 0-1, v: 0-1)."""
        h = h % 360
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return cls(r + m, g + m, b + m, a)

    # Common color presets
    @classmethod
    def white(cls) -> "Color": return cls(1.0, 1.0, 1.0, 1.0)
    @classmethod
    def black(cls) -> "Color": return cls(0.0, 0.0, 0.0, 1.0)
    @classmethod
    def red(cls) -> "Color": return cls(1.0, 0.0, 0.0, 1.0)
    @classmethod
    def green(cls) -> "Color": return cls(0.0, 1.0, 0.0, 1.0)
    @classmethod
    def blue(cls) -> "Color": return cls(0.0, 0.0, 1.0, 1.0)
    @classmethod
    def yellow(cls) -> "Color": return cls(1.0, 1.0, 0.0, 1.0)
    @classmethod
    def cyan(cls) -> "Color": return cls(0.0, 1.0, 1.0, 1.0)
    @classmethod
    def magenta(cls) -> "Color": return cls(1.0, 0.0, 1.0, 1.0)
    @classmethod
    def orange(cls) -> "Color": return cls(1.0, 0.5, 0.0, 1.0)
    @classmethod
    def purple(cls) -> "Color": return cls(0.5, 0.0, 1.0, 1.0)
    @classmethod
    def gray(cls) -> "Color": return cls(0.5, 0.5, 0.5, 1.0)
    @classmethod
    def clear(cls) -> "Color": return cls(0.0, 0.0, 0.0, 0.0)


@dataclass
class Transform:
    """3D transform: position, rotation, and scale."""
    position: Vector3 = None
    rotation: Quaternion = None
    scale: Vector3 = None

    def __post_init__(self):
        if self.position is None:
            self.position = Vector3.zero()
        if self.rotation is None:
            self.rotation = Quaternion.identity()
        if self.scale is None:
            self.scale = Vector3.one()

    def __repr__(self) -> str:
        return f"Transform(pos={self.position}, rot={self.rotation}, scale={self.scale})"

    def transform_point(self, point: Vector3) -> Vector3:
        """Transform a point from local to world space."""
        scaled = Vector3(point.x * self.scale.x, point.y * self.scale.y, point.z * self.scale.z)
        rotated = self.rotation.rotate_vector(scaled)
        return rotated + self.position

    def transform_direction(self, direction: Vector3) -> Vector3:
        """Transform a direction (ignores position and scale)."""
        return self.rotation.rotate_vector(direction)

    def inverse_transform_point(self, point: Vector3) -> Vector3:
        """Transform a point from world to local space."""
        translated = point - self.position
        rotated = self.rotation.inverse().rotate_vector(translated)
        return Vector3(
            rotated.x / self.scale.x if self.scale.x != 0 else 0,
            rotated.y / self.scale.y if self.scale.y != 0 else 0,
            rotated.z / self.scale.z if self.scale.z != 0 else 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "scale": self.scale.to_dict()
        }

    @classmethod
    def identity(cls) -> "Transform":
        return cls(Vector3.zero(), Quaternion.identity(), Vector3.one())


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    center: Vector3 = None
    size: Vector3 = None

    def __post_init__(self):
        if self.center is None:
            self.center = Vector3.zero()
        if self.size is None:
            self.size = Vector3.one()

    @property
    def min(self) -> Vector3:
        """Minimum corner."""
        half = Vector3(self.size.x/2, self.size.y/2, self.size.z/2)
        return self.center - half

    @property
    def max(self) -> Vector3:
        """Maximum corner."""
        half = Vector3(self.size.x/2, self.size.y/2, self.size.z/2)
        return self.center + half

    def contains(self, point: Vector3) -> bool:
        """Check if point is inside box."""
        min_p, max_p = self.min, self.max
        return (min_p.x <= point.x <= max_p.x and
                min_p.y <= point.y <= max_p.y and
                min_p.z <= point.z <= max_p.z)

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if boxes overlap."""
        return (abs(self.center.x - other.center.x) <= (self.size.x + other.size.x) / 2 and
                abs(self.center.y - other.center.y) <= (self.size.y + other.size.y) / 2 and
                abs(self.center.z - other.center.z) <= (self.size.z + other.size.z) / 2)

    def expand(self, amount: float) -> "BoundingBox":
        """Return expanded box."""
        return BoundingBox(
            self.center,
            Vector3(self.size.x + amount*2, self.size.y + amount*2, self.size.z + amount*2)
        )

    def get_corners(self) -> list:
        """Get 8 corners of the box."""
        half = Vector3(self.size.x/2, self.size.y/2, self.size.z/2)
        corners = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corners.append(Vector3(
                        self.center.x + half.x * sx,
                        self.center.y + half.y * sy,
                        self.center.z + half.z * sz
                    ))
        return corners

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.to_dict(),
            "size": self.size.to_dict()
        }

    @classmethod
    def from_min_max(cls, min_point: Vector3, max_point: Vector3) -> "BoundingBox":
        """Create from min and max corners."""
        center = Vector3(
            (min_point.x + max_point.x) / 2,
            (min_point.y + max_point.y) / 2,
            (min_point.z + max_point.z) / 2
        )
        size = Vector3(
            max_point.x - min_point.x,
            max_point.y - min_point.y,
            max_point.z - min_point.z
        )
        return cls(center, size)
