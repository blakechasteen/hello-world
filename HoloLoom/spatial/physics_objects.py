"""
Physics-Based Object Manipulation for HoloLoom Spatial Computing.

Provides physics simulation for XR object interaction including:
- Rigid body dynamics
- Grabbing and throwing
- Collision detection and response
- Gravity and forces
- Constraints (hinges, springs, sliders)
- Material properties

Created: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import math


class PhysicsShape(Enum):
    """Collision shape types."""
    SPHERE = "sphere"
    BOX = "box"
    CAPSULE = "capsule"
    CYLINDER = "cylinder"
    MESH = "mesh"
    CONVEX_HULL = "convex_hull"
    COMPOUND = "compound"


class BodyType(Enum):
    """Physics body types."""
    STATIC = "static"       # Immovable
    DYNAMIC = "dynamic"     # Full physics
    KINEMATIC = "kinematic" # Animated, affects others


class CollisionLayer(Enum):
    """Collision layer groups."""
    DEFAULT = 0
    STATIC = 1
    DYNAMIC = 2
    HANDS = 3
    UI = 4
    TRIGGER = 5
    GRABBABLE = 6
    KNOWLEDGE_NODE = 7


@dataclass
class Vector3:
    """Simple 3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag > 0:
            return Vector3(self.x/mag, self.y/mag, self.z/mag)
        return Vector3()

    def dot(self, other: "Vector3") -> float:
        return self.x*other.x + self.y*other.y + self.z*other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vector3":
        return cls(t[0], t[1], t[2])


@dataclass
class Quaternion:
    """Quaternion for rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_euler(self) -> Vector3:
        """Convert to Euler angles (radians)."""
        # Simplified conversion
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return Vector3(roll, pitch, yaw)


@dataclass
class PhysicsMaterial:
    """Material properties for physics interactions."""
    material_id: str
    name: str
    friction: float = 0.5           # 0-1
    restitution: float = 0.3        # Bounciness 0-1
    density: float = 1.0            # kg/m^3
    linear_damping: float = 0.1     # Air resistance
    angular_damping: float = 0.1    # Rotational resistance

    # Advanced properties
    static_friction: Optional[float] = None  # Override for static
    combine_mode: str = "average"  # average, min, max, multiply


@dataclass
class CollisionShape:
    """Collision shape definition."""
    shape_type: PhysicsShape
    size: Vector3 = field(default_factory=Vector3)  # Extents for box, radius for sphere
    offset: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    # For mesh shapes
    vertices: Optional[List[Vector3]] = None
    indices: Optional[List[int]] = None

    # For compound shapes
    children: Optional[List["CollisionShape"]] = None

    def get_volume(self) -> float:
        """Estimate volume of shape."""
        if self.shape_type == PhysicsShape.SPHERE:
            r = self.size.x
            return (4/3) * math.pi * r**3
        elif self.shape_type == PhysicsShape.BOX:
            return self.size.x * self.size.y * self.size.z * 8
        elif self.shape_type == PhysicsShape.CAPSULE:
            r = self.size.x
            h = self.size.y
            return math.pi * r**2 * (h + (4/3) * r)
        elif self.shape_type == PhysicsShape.CYLINDER:
            r = self.size.x
            h = self.size.y
            return math.pi * r**2 * h
        return 1.0


@dataclass
class PhysicsBody:
    """A physics-enabled object."""
    body_id: str
    node_id: Optional[str] = None  # Associated XRNode

    # Transform
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    # Physics properties
    body_type: BodyType = BodyType.DYNAMIC
    mass: float = 1.0
    shape: CollisionShape = field(default_factory=lambda: CollisionShape(PhysicsShape.BOX, Vector3(0.5, 0.5, 0.5)))
    material: Optional[PhysicsMaterial] = None

    # Velocity state
    linear_velocity: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)

    # Collision
    collision_layer: CollisionLayer = CollisionLayer.DYNAMIC
    collision_mask: int = 0xFFFF  # Which layers to collide with

    # Interaction state
    is_grabbable: bool = True
    is_grabbed: bool = False
    grabbed_by: Optional[str] = None  # Hand/controller ID
    grab_offset: Vector3 = field(default_factory=Vector3)

    # Constraints
    is_constrained: bool = False
    freeze_position: Tuple[bool, bool, bool] = (False, False, False)
    freeze_rotation: Tuple[bool, bool, bool] = (False, False, False)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def get_inertia_tensor(self) -> Vector3:
        """Calculate moment of inertia (simplified)."""
        if self.shape.shape_type == PhysicsShape.SPHERE:
            r = self.shape.size.x
            i = (2/5) * self.mass * r**2
            return Vector3(i, i, i)
        elif self.shape.shape_type == PhysicsShape.BOX:
            sx, sy, sz = self.shape.size.x*2, self.shape.size.y*2, self.shape.size.z*2
            return Vector3(
                (1/12) * self.mass * (sy**2 + sz**2),
                (1/12) * self.mass * (sx**2 + sz**2),
                (1/12) * self.mass * (sx**2 + sy**2)
            )
        return Vector3(self.mass, self.mass, self.mass)

    def apply_force(self, force: Vector3, point: Optional[Vector3] = None):
        """Apply force at point (None = center of mass)."""
        if self.body_type != BodyType.DYNAMIC or self.mass <= 0:
            return

        # Linear acceleration: F = ma -> a = F/m
        acceleration = force * (1.0 / self.mass)
        self.linear_velocity = self.linear_velocity + acceleration

        # Torque if force applied off-center
        if point:
            r = point - self.position
            torque = r.cross(force)
            inertia = self.get_inertia_tensor()
            if inertia.x > 0:
                self.angular_velocity.x += torque.x / inertia.x
            if inertia.y > 0:
                self.angular_velocity.y += torque.y / inertia.y
            if inertia.z > 0:
                self.angular_velocity.z += torque.z / inertia.z

    def apply_impulse(self, impulse: Vector3, point: Optional[Vector3] = None):
        """Apply instantaneous impulse."""
        if self.body_type != BodyType.DYNAMIC or self.mass <= 0:
            return

        velocity_change = impulse * (1.0 / self.mass)
        self.linear_velocity = self.linear_velocity + velocity_change

        if point:
            r = point - self.position
            angular_impulse = r.cross(impulse)
            inertia = self.get_inertia_tensor()
            if inertia.x > 0:
                self.angular_velocity.x += angular_impulse.x / inertia.x
            if inertia.y > 0:
                self.angular_velocity.y += angular_impulse.y / inertia.y
            if inertia.z > 0:
                self.angular_velocity.z += angular_impulse.z / inertia.z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.body_id,
            "nodeId": self.node_id,
            "position": self.position.to_tuple(),
            "rotation": (self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w),
            "bodyType": self.body_type.value,
            "mass": self.mass,
            "linearVelocity": self.linear_velocity.to_tuple(),
            "angularVelocity": self.angular_velocity.to_tuple(),
            "isGrabbable": self.is_grabbable,
            "isGrabbed": self.is_grabbed
        }


class ConstraintType(Enum):
    """Physics constraint types."""
    FIXED = "fixed"             # Rigid connection
    HINGE = "hinge"             # Rotation around axis
    SLIDER = "slider"           # Translation along axis
    SPRING = "spring"           # Spring force
    BALL_SOCKET = "ball_socket" # Free rotation
    DISTANCE = "distance"       # Fixed distance
    CONE_TWIST = "cone_twist"   # Limited rotation


@dataclass
class PhysicsConstraint:
    """Constraint between two bodies."""
    constraint_id: str
    constraint_type: ConstraintType
    body_a: str  # Body ID
    body_b: Optional[str] = None  # None = world anchor

    # Anchor points (local to each body)
    anchor_a: Vector3 = field(default_factory=Vector3)
    anchor_b: Vector3 = field(default_factory=Vector3)

    # Axis for hinges/sliders
    axis: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))

    # Limits
    lower_limit: float = 0.0
    upper_limit: float = 0.0
    limits_enabled: bool = False

    # Spring properties
    stiffness: float = 100.0
    damping: float = 10.0

    # State
    enabled: bool = True
    breaking_force: float = 0.0  # 0 = unbreakable
    is_broken: bool = False


@dataclass
class CollisionEvent:
    """Collision event data."""
    body_a: str
    body_b: str
    contact_point: Vector3
    normal: Vector3
    impulse: float
    velocity: float
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class GrabState:
    """State for grabbed object."""
    body_id: str
    grabber_id: str  # Hand/controller ID
    grab_point: Vector3  # World position
    local_offset: Vector3  # Offset in body local space
    start_time: float
    velocity_samples: List[Vector3] = field(default_factory=list)  # For throw velocity
    max_samples: int = 10


class PhysicsWorld:
    """
    Physics simulation world.

    Manages physics bodies, constraints, and simulation.
    """

    def __init__(
        self,
        gravity: Vector3 = None,
        time_step: float = 1/60,  # 60 Hz physics
        max_substeps: int = 4
    ):
        self.gravity = gravity or Vector3(0, -9.81, 0)
        self.time_step = time_step
        self.max_substeps = max_substeps

        # Bodies and constraints
        self.bodies: Dict[str, PhysicsBody] = {}
        self.constraints: Dict[str, PhysicsConstraint] = {}

        # Materials library
        self.materials: Dict[str, PhysicsMaterial] = {}
        self._load_default_materials()

        # Collision tracking
        self.collision_events: List[CollisionEvent] = []
        self.max_collision_events = 100

        # Grab states
        self.grab_states: Dict[str, GrabState] = {}

        # Callbacks
        self.on_collision: List[Callable[[CollisionEvent], None]] = []
        self.on_grab: List[Callable[[str, str], None]] = []
        self.on_release: List[Callable[[str, str, Vector3], None]] = []

        # Simulation state
        self.is_paused: bool = False
        self.accumulated_time: float = 0.0

    def _load_default_materials(self):
        """Load preset physics materials."""
        presets = [
            PhysicsMaterial("default", "Default", friction=0.5, restitution=0.3),
            PhysicsMaterial("rubber", "Rubber", friction=0.9, restitution=0.8),
            PhysicsMaterial("metal", "Metal", friction=0.3, restitution=0.1, density=7.8),
            PhysicsMaterial("wood", "Wood", friction=0.6, restitution=0.2, density=0.6),
            PhysicsMaterial("ice", "Ice", friction=0.05, restitution=0.1),
            PhysicsMaterial("bouncy", "Bouncy Ball", friction=0.7, restitution=0.95),
            PhysicsMaterial("glass", "Glass", friction=0.4, restitution=0.1, density=2.5),
            PhysicsMaterial("concrete", "Concrete", friction=0.7, restitution=0.05, density=2.4),
            PhysicsMaterial("knowledge_node", "Knowledge Node", friction=0.5, restitution=0.4, density=0.1),
        ]

        for material in presets:
            self.materials[material.material_id] = material

    def create_body(
        self,
        body_id: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        body_type: BodyType = BodyType.DYNAMIC,
        shape: PhysicsShape = PhysicsShape.BOX,
        size: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        mass: float = 1.0,
        material_id: str = "default",
        node_id: Optional[str] = None,
        is_grabbable: bool = True
    ) -> PhysicsBody:
        """Create a new physics body."""
        body = PhysicsBody(
            body_id=body_id,
            node_id=node_id,
            position=Vector3.from_tuple(position),
            body_type=body_type,
            mass=mass if body_type == BodyType.DYNAMIC else 0,
            shape=CollisionShape(shape, Vector3.from_tuple(size)),
            material=self.materials.get(material_id),
            is_grabbable=is_grabbable
        )

        # Set collision layer based on type
        if body_type == BodyType.STATIC:
            body.collision_layer = CollisionLayer.STATIC
        elif node_id:
            body.collision_layer = CollisionLayer.KNOWLEDGE_NODE

        self.bodies[body_id] = body
        return body

    def remove_body(self, body_id: str):
        """Remove a physics body."""
        # Release if grabbed
        if body_id in self.grab_states:
            self.release(body_id)

        # Remove constraints involving this body
        constraints_to_remove = [
            c_id for c_id, c in self.constraints.items()
            if c.body_a == body_id or c.body_b == body_id
        ]
        for c_id in constraints_to_remove:
            del self.constraints[c_id]

        self.bodies.pop(body_id, None)

    def create_constraint(
        self,
        constraint_id: str,
        constraint_type: ConstraintType,
        body_a: str,
        body_b: Optional[str] = None,
        anchor_a: Tuple[float, float, float] = (0, 0, 0),
        anchor_b: Tuple[float, float, float] = (0, 0, 0),
        axis: Tuple[float, float, float] = (0, 1, 0)
    ) -> Optional[PhysicsConstraint]:
        """Create a constraint between bodies."""
        if body_a not in self.bodies:
            return None
        if body_b and body_b not in self.bodies:
            return None

        constraint = PhysicsConstraint(
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            body_a=body_a,
            body_b=body_b,
            anchor_a=Vector3.from_tuple(anchor_a),
            anchor_b=Vector3.from_tuple(anchor_b),
            axis=Vector3.from_tuple(axis)
        )

        self.constraints[constraint_id] = constraint
        return constraint

    def set_constraint_limits(
        self,
        constraint_id: str,
        lower: float,
        upper: float
    ) -> bool:
        """Set limits on a constraint."""
        if constraint_id not in self.constraints:
            return False

        constraint = self.constraints[constraint_id]
        constraint.lower_limit = lower
        constraint.upper_limit = upper
        constraint.limits_enabled = True
        return True

    def grab(
        self,
        body_id: str,
        grabber_id: str,
        grab_point: Tuple[float, float, float]
    ) -> bool:
        """Grab a physics body."""
        if body_id not in self.bodies:
            return False

        body = self.bodies[body_id]
        if not body.is_grabbable or body.is_grabbed:
            return False

        # Calculate local offset
        world_point = Vector3.from_tuple(grab_point)
        local_offset = world_point - body.position

        # Create grab state
        grab_state = GrabState(
            body_id=body_id,
            grabber_id=grabber_id,
            grab_point=world_point,
            local_offset=local_offset,
            start_time=datetime.now().timestamp()
        )
        self.grab_states[body_id] = grab_state

        # Update body
        body.is_grabbed = True
        body.grabbed_by = grabber_id
        body.grab_offset = local_offset

        # Stop velocity while grabbed
        body.linear_velocity = Vector3()
        body.angular_velocity = Vector3()

        # Fire callback
        for callback in self.on_grab:
            callback(body_id, grabber_id)

        return True

    def update_grab(
        self,
        body_id: str,
        hand_position: Tuple[float, float, float],
        hand_rotation: Optional[Tuple[float, float, float, float]] = None
    ) -> bool:
        """Update grabbed object position."""
        if body_id not in self.grab_states:
            return False

        grab_state = self.grab_states[body_id]
        body = self.bodies.get(body_id)
        if not body:
            return False

        # Calculate new position
        hand_pos = Vector3.from_tuple(hand_position)
        new_position = hand_pos - grab_state.local_offset

        # Track velocity for throwing
        grab_state.velocity_samples.append(new_position - body.position)
        if len(grab_state.velocity_samples) > grab_state.max_samples:
            grab_state.velocity_samples.pop(0)

        # Update position
        body.position = new_position

        # Update rotation if provided
        if hand_rotation:
            body.rotation = Quaternion(*hand_rotation)

        return True

    def release(
        self,
        body_id: str,
        throw_velocity_scale: float = 1.0
    ) -> Optional[Vector3]:
        """Release a grabbed object."""
        if body_id not in self.grab_states:
            return None

        grab_state = self.grab_states[body_id]
        body = self.bodies.get(body_id)
        if not body:
            return None

        # Calculate throw velocity from samples
        throw_velocity = Vector3()
        if grab_state.velocity_samples:
            for v in grab_state.velocity_samples:
                throw_velocity = throw_velocity + v
            throw_velocity = throw_velocity * (1.0 / len(grab_state.velocity_samples))
            throw_velocity = throw_velocity * (throw_velocity_scale / self.time_step)

        # Apply to body
        body.linear_velocity = throw_velocity
        body.is_grabbed = False
        body.grabbed_by = None

        # Clean up
        del self.grab_states[body_id]

        # Fire callback
        for callback in self.on_release:
            callback(body_id, grab_state.grabber_id, throw_velocity)

        return throw_velocity

    def step(self, delta_time: float):
        """Step physics simulation."""
        if self.is_paused:
            return

        self.accumulated_time += delta_time

        # Fixed timestep with substeps
        substeps = 0
        while self.accumulated_time >= self.time_step and substeps < self.max_substeps:
            self._physics_step()
            self.accumulated_time -= self.time_step
            substeps += 1

    def _physics_step(self):
        """Single physics timestep."""
        dt = self.time_step

        for body in self.bodies.values():
            if body.body_type != BodyType.DYNAMIC:
                continue
            if body.is_grabbed:
                continue

            # Apply gravity
            if body.mass > 0:
                body.linear_velocity = body.linear_velocity + self.gravity * dt

            # Apply damping
            material = body.material or self.materials["default"]
            body.linear_velocity = body.linear_velocity * (1.0 - material.linear_damping * dt)
            body.angular_velocity = body.angular_velocity * (1.0 - material.angular_damping * dt)

            # Apply position freeze
            if body.freeze_position[0]:
                body.linear_velocity.x = 0
            if body.freeze_position[1]:
                body.linear_velocity.y = 0
            if body.freeze_position[2]:
                body.linear_velocity.z = 0

            # Apply rotation freeze
            if body.freeze_rotation[0]:
                body.angular_velocity.x = 0
            if body.freeze_rotation[1]:
                body.angular_velocity.y = 0
            if body.freeze_rotation[2]:
                body.angular_velocity.z = 0

            # Integrate position
            body.position = body.position + body.linear_velocity * dt

            # Integrate rotation (simplified)
            omega = body.angular_velocity
            body.rotation.x += omega.x * dt * 0.5
            body.rotation.y += omega.y * dt * 0.5
            body.rotation.z += omega.z * dt * 0.5

        # Process constraints
        self._solve_constraints()

        # Collision detection (simplified)
        self._detect_collisions()

    def _solve_constraints(self):
        """Solve physics constraints."""
        for constraint in self.constraints.values():
            if not constraint.enabled or constraint.is_broken:
                continue

            body_a = self.bodies.get(constraint.body_a)
            body_b = self.bodies.get(constraint.body_b) if constraint.body_b else None

            if not body_a:
                continue

            if constraint.constraint_type == ConstraintType.SPRING:
                self._solve_spring_constraint(constraint, body_a, body_b)
            elif constraint.constraint_type == ConstraintType.DISTANCE:
                self._solve_distance_constraint(constraint, body_a, body_b)
            # Other constraint types would be implemented here

    def _solve_spring_constraint(
        self,
        constraint: PhysicsConstraint,
        body_a: PhysicsBody,
        body_b: Optional[PhysicsBody]
    ):
        """Solve spring constraint."""
        # Get world positions of anchors
        anchor_a_world = body_a.position + constraint.anchor_a
        if body_b:
            anchor_b_world = body_b.position + constraint.anchor_b
        else:
            anchor_b_world = constraint.anchor_b

        # Calculate displacement
        delta = anchor_b_world - anchor_a_world
        distance = delta.magnitude()

        if distance < 0.0001:
            return

        direction = delta.normalized()

        # Spring force: F = -kx - cv
        rest_length = (constraint.lower_limit + constraint.upper_limit) / 2
        displacement = distance - rest_length

        # Relative velocity
        rel_velocity = Vector3()
        if body_a.body_type == BodyType.DYNAMIC:
            rel_velocity = rel_velocity - body_a.linear_velocity
        if body_b and body_b.body_type == BodyType.DYNAMIC:
            rel_velocity = rel_velocity + body_b.linear_velocity

        velocity_along_spring = rel_velocity.dot(direction)

        # Force magnitude
        force = constraint.stiffness * displacement + constraint.damping * velocity_along_spring

        # Apply forces
        force_vector = direction * force
        if body_a.body_type == BodyType.DYNAMIC:
            body_a.apply_force(force_vector)
        if body_b and body_b.body_type == BodyType.DYNAMIC:
            body_b.apply_force(force_vector * -1)

    def _solve_distance_constraint(
        self,
        constraint: PhysicsConstraint,
        body_a: PhysicsBody,
        body_b: Optional[PhysicsBody]
    ):
        """Solve distance constraint (keep fixed distance)."""
        anchor_a_world = body_a.position + constraint.anchor_a
        if body_b:
            anchor_b_world = body_b.position + constraint.anchor_b
        else:
            anchor_b_world = constraint.anchor_b

        delta = anchor_b_world - anchor_a_world
        distance = delta.magnitude()
        target_distance = constraint.lower_limit

        if distance < 0.0001:
            return

        correction = (distance - target_distance) * 0.5
        direction = delta.normalized()

        if body_a.body_type == BodyType.DYNAMIC:
            body_a.position = body_a.position + direction * correction
        if body_b and body_b.body_type == BodyType.DYNAMIC:
            body_b.position = body_b.position - direction * correction

    def _detect_collisions(self):
        """Simple collision detection between bodies."""
        body_list = list(self.bodies.values())

        for i, body_a in enumerate(body_list):
            for body_b in body_list[i+1:]:
                # Skip if both static
                if body_a.body_type == BodyType.STATIC and body_b.body_type == BodyType.STATIC:
                    continue

                # Check layer collision
                if not self._layers_collide(body_a.collision_layer, body_b.collision_layer):
                    continue

                # Simple sphere-sphere collision for now
                if self._check_collision(body_a, body_b):
                    self._resolve_collision(body_a, body_b)

    def _layers_collide(self, layer_a: CollisionLayer, layer_b: CollisionLayer) -> bool:
        """Check if two collision layers interact."""
        # Simplified - all layers collide by default
        return True

    def _check_collision(self, body_a: PhysicsBody, body_b: PhysicsBody) -> bool:
        """Check collision between two bodies (simplified sphere check)."""
        # Use bounding sphere approximation
        radius_a = body_a.shape.size.magnitude()
        radius_b = body_b.shape.size.magnitude()

        delta = body_b.position - body_a.position
        distance = delta.magnitude()

        return distance < (radius_a + radius_b)

    def _resolve_collision(self, body_a: PhysicsBody, body_b: PhysicsBody):
        """Resolve collision between two bodies."""
        # Get materials
        mat_a = body_a.material or self.materials["default"]
        mat_b = body_b.material or self.materials["default"]

        # Average restitution
        restitution = (mat_a.restitution + mat_b.restitution) / 2

        # Calculate collision normal and relative velocity
        normal = (body_b.position - body_a.position).normalized()
        rel_velocity = body_b.linear_velocity - body_a.linear_velocity
        velocity_along_normal = rel_velocity.dot(normal)

        # Don't resolve if separating
        if velocity_along_normal > 0:
            return

        # Calculate impulse
        inv_mass_a = 1.0 / body_a.mass if body_a.mass > 0 and body_a.body_type == BodyType.DYNAMIC else 0
        inv_mass_b = 1.0 / body_b.mass if body_b.mass > 0 and body_b.body_type == BodyType.DYNAMIC else 0

        j = -(1 + restitution) * velocity_along_normal
        j /= (inv_mass_a + inv_mass_b) if (inv_mass_a + inv_mass_b) > 0 else 1

        # Apply impulse
        impulse = normal * j
        if body_a.body_type == BodyType.DYNAMIC:
            body_a.linear_velocity = body_a.linear_velocity - impulse * inv_mass_a
        if body_b.body_type == BodyType.DYNAMIC:
            body_b.linear_velocity = body_b.linear_velocity + impulse * inv_mass_b

        # Create collision event
        contact_point = body_a.position + normal * (body_a.shape.size.magnitude())
        event = CollisionEvent(
            body_a=body_a.body_id,
            body_b=body_b.body_id,
            contact_point=contact_point,
            normal=normal,
            impulse=j,
            velocity=abs(velocity_along_normal)
        )

        self.collision_events.append(event)
        if len(self.collision_events) > self.max_collision_events:
            self.collision_events.pop(0)

        # Fire callbacks
        for callback in self.on_collision:
            callback(event)

    def raycast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_distance: float = 100.0
    ) -> Optional[Dict[str, Any]]:
        """Cast ray and return first hit."""
        ray_origin = Vector3.from_tuple(origin)
        ray_dir = Vector3.from_tuple(direction).normalized()

        closest_hit = None
        closest_distance = max_distance

        for body in self.bodies.values():
            # Simplified sphere intersection
            radius = body.shape.size.magnitude()
            to_center = body.position - ray_origin

            # Project center onto ray
            proj_distance = to_center.dot(ray_dir)
            if proj_distance < 0 or proj_distance > closest_distance:
                continue

            # Distance from center to ray
            closest_point = ray_origin + ray_dir * proj_distance
            dist_to_ray = (body.position - closest_point).magnitude()

            if dist_to_ray < radius:
                # Hit!
                hit_distance = proj_distance - math.sqrt(radius**2 - dist_to_ray**2)
                if hit_distance < closest_distance:
                    closest_distance = hit_distance
                    hit_point = ray_origin + ray_dir * hit_distance
                    hit_normal = (hit_point - body.position).normalized()

                    closest_hit = {
                        "body_id": body.body_id,
                        "node_id": body.node_id,
                        "point": hit_point.to_tuple(),
                        "normal": hit_normal.to_tuple(),
                        "distance": hit_distance
                    }

        return closest_hit

    def get_bodies_in_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float
    ) -> List[str]:
        """Get all bodies within sphere."""
        center_vec = Vector3.from_tuple(center)
        results = []

        for body in self.bodies.values():
            distance = (body.position - center_vec).magnitude()
            body_radius = body.shape.size.magnitude()

            if distance - body_radius < radius:
                results.append(body.body_id)

        return results

    def set_gravity(self, gravity: Tuple[float, float, float]):
        """Set world gravity."""
        self.gravity = Vector3.from_tuple(gravity)

    def get_statistics(self) -> Dict[str, Any]:
        """Get physics world statistics."""
        dynamic_count = sum(1 for b in self.bodies.values() if b.body_type == BodyType.DYNAMIC)
        grabbed_count = sum(1 for b in self.bodies.values() if b.is_grabbed)

        return {
            "total_bodies": len(self.bodies),
            "dynamic_bodies": dynamic_count,
            "static_bodies": len(self.bodies) - dynamic_count,
            "constraints": len(self.constraints),
            "grabbed_objects": grabbed_count,
            "collision_events": len(self.collision_events),
            "is_paused": self.is_paused,
            "gravity": self.gravity.to_tuple(),
            "time_step": self.time_step
        }

    def to_state(self) -> Dict[str, Any]:
        """Export full state for WebXR client."""
        return {
            "type": "physics_state",
            "bodies": {bid: b.to_dict() for bid, b in self.bodies.items()},
            "gravity": self.gravity.to_tuple(),
            "isPaused": self.is_paused,
            "statistics": self.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }
