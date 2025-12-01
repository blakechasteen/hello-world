"""
Architecture Visualizer
=======================

Renders system architectures, code structures, and component diagrams
as interactive 3D visualizations.

When discussing technical systems, this creates visual representations
of components, data flows, layers, and connections.

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import re
from enum import Enum


class ComponentType(Enum):
    SERVICE = "service"           # Microservice, API
    DATABASE = "database"         # Database, storage
    QUEUE = "queue"              # Message queue
    CACHE = "cache"              # Cache layer
    GATEWAY = "gateway"          # API gateway, load balancer
    CLIENT = "client"            # Frontend, mobile app
    FUNCTION = "function"        # Function, method
    CLASS = "class"              # Class, module
    LAYER = "layer"              # Architectural layer
    CONTAINER = "container"      # Docker, K8s
    EXTERNAL = "external"        # External service


class ConnectionType(Enum):
    HTTP = "http"                # REST API call
    GRPC = "grpc"               # gRPC call
    WEBSOCKET = "websocket"     # WebSocket connection
    MESSAGE = "message"          # Message queue
    DATA = "data"               # Data flow
    DEPENDS = "depends"          # Dependency
    INHERITS = "inherits"        # Inheritance
    IMPLEMENTS = "implements"    # Interface implementation


@dataclass
class ArchComponent:
    """A component in the architecture."""
    id: str
    name: str
    component_type: ComponentType
    description: str = ""
    layer: int = 0  # Vertical layer (0=top, increases downward)
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    children: List[str] = field(default_factory=list)  # Child component IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchConnection:
    """A connection between components."""
    id: str
    source: str  # Component ID
    target: str  # Component ID
    connection_type: ConnectionType
    label: str = ""
    bidirectional: bool = False
    animated: bool = True


@dataclass
class Architecture:
    """Complete architecture diagram."""
    name: str
    components: List[ArchComponent]
    connections: List[ArchConnection]
    layers: List[str] = field(default_factory=list)  # Layer names top to bottom


# Component type styling
COMPONENT_STYLES = {
    ComponentType.SERVICE: {
        'color': '#3b82f6',      # Blue
        'shape': 'rounded_box',
        'icon': '⚙️',
        'height': 1.0
    },
    ComponentType.DATABASE: {
        'color': '#10b981',      # Green
        'shape': 'cylinder',
        'icon': '🗄️',
        'height': 1.2
    },
    ComponentType.QUEUE: {
        'color': '#f59e0b',      # Amber
        'shape': 'box',
        'icon': '📨',
        'height': 0.6
    },
    ComponentType.CACHE: {
        'color': '#ef4444',      # Red
        'shape': 'box',
        'icon': '⚡',
        'height': 0.6
    },
    ComponentType.GATEWAY: {
        'color': '#8b5cf6',      # Purple
        'shape': 'hexagon',
        'icon': '🚪',
        'height': 0.8
    },
    ComponentType.CLIENT: {
        'color': '#06b6d4',      # Cyan
        'shape': 'rounded_box',
        'icon': '📱',
        'height': 1.0
    },
    ComponentType.FUNCTION: {
        'color': '#ec4899',      # Pink
        'shape': 'box',
        'icon': 'λ',
        'height': 0.5
    },
    ComponentType.CLASS: {
        'color': '#6366f1',      # Indigo
        'shape': 'box',
        'icon': '📦',
        'height': 0.8
    },
    ComponentType.LAYER: {
        'color': '#64748b',      # Slate
        'shape': 'plane',
        'icon': '═',
        'height': 0.2
    },
    ComponentType.CONTAINER: {
        'color': '#0ea5e9',      # Sky
        'shape': 'box_outline',
        'icon': '🐳',
        'height': 1.5
    },
    ComponentType.EXTERNAL: {
        'color': '#9ca3af',      # Gray
        'shape': 'cloud',
        'icon': '☁️',
        'height': 1.0
    }
}

# Connection type styling
CONNECTION_STYLES = {
    ConnectionType.HTTP: {'color': '#3b82f6', 'dash': False, 'width': 2},
    ConnectionType.GRPC: {'color': '#10b981', 'dash': False, 'width': 2},
    ConnectionType.WEBSOCKET: {'color': '#8b5cf6', 'dash': True, 'width': 2},
    ConnectionType.MESSAGE: {'color': '#f59e0b', 'dash': True, 'width': 2},
    ConnectionType.DATA: {'color': '#64748b', 'dash': False, 'width': 3},
    ConnectionType.DEPENDS: {'color': '#9ca3af', 'dash': True, 'width': 1},
    ConnectionType.INHERITS: {'color': '#6366f1', 'dash': False, 'width': 2},
    ConnectionType.IMPLEMENTS: {'color': '#ec4899', 'dash': True, 'width': 2}
}


class ArchitectureExtractor:
    """Extract architecture from text description."""

    # Component detection patterns
    COMPONENT_PATTERNS = {
        ComponentType.SERVICE: [
            r'(\w+)\s+(?:service|api|microservice)',
            r'(\w+)Service',
            r'(\w+)\s+server'
        ],
        ComponentType.DATABASE: [
            r'(\w+)\s+(?:database|db|storage)',
            r'(?:PostgreSQL|MySQL|MongoDB|Redis|Cassandra|Neo4j|Qdrant)',
            r'(\w+)DB'
        ],
        ComponentType.QUEUE: [
            r'(\w+)\s+(?:queue|broker|kafka|rabbitmq)',
            r'message\s+(?:queue|broker)'
        ],
        ComponentType.CACHE: [
            r'(\w+)\s+cache',
            r'(?:redis|memcached)\s+cache'
        ],
        ComponentType.GATEWAY: [
            r'(?:api\s+)?gateway',
            r'load\s+balancer',
            r'(\w+)\s+proxy'
        ],
        ComponentType.CLIENT: [
            r'(?:web|mobile)\s+(?:client|app)',
            r'frontend',
            r'(\w+)\s+client'
        ],
        ComponentType.FUNCTION: [
            r'(?:function|method)\s+(\w+)',
            r'(\w+)\(\)',
            r'lambda\s+(\w+)'
        ],
        ComponentType.CLASS: [
            r'class\s+(\w+)',
            r'(\w+)Controller',
            r'(\w+)Manager',
            r'(\w+)Handler'
        ],
        ComponentType.LAYER: [
            r'(\w+)\s+layer',
            r'(\w+)\s+tier'
        ],
        ComponentType.CONTAINER: [
            r'(?:docker|kubernetes|k8s)\s+(\w+)',
            r'(\w+)\s+container',
            r'(\w+)\s+pod'
        ],
        ComponentType.EXTERNAL: [
            r'(?:external|third[\s-]?party)\s+(\w+)',
            r'(\w+)\s+(?:api|integration)'
        ]
    }

    # Connection patterns
    CONNECTION_PATTERNS = [
        (r'(\w+)\s+(?:calls?|requests?|sends?\s+to)\s+(\w+)', ConnectionType.HTTP),
        (r'(\w+)\s+(?:->|→|connects?\s+to)\s+(\w+)', ConnectionType.DATA),
        (r'(\w+)\s+(?:publishes?|emits?)\s+to\s+(\w+)', ConnectionType.MESSAGE),
        (r'(\w+)\s+(?:reads?|queries?|fetches?)\s+(?:from\s+)?(\w+)', ConnectionType.DATA),
        (r'(\w+)\s+(?:writes?|stores?|saves?)\s+(?:to\s+)?(\w+)', ConnectionType.DATA),
        (r'(\w+)\s+(?:extends?|inherits?)\s+(\w+)', ConnectionType.INHERITS),
        (r'(\w+)\s+(?:implements?)\s+(\w+)', ConnectionType.IMPLEMENTS),
        (r'(\w+)\s+(?:depends?\s+on|uses?)\s+(\w+)', ConnectionType.DEPENDS)
    ]

    def extract_architecture(self, text: str) -> Architecture:
        """Extract architecture from text description."""
        components = []
        connections = []
        component_map = {}

        # Extract components
        for comp_type, patterns in self.COMPONENT_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match if isinstance(match, str) else match[0] if match else None
                    if name and len(name) > 1 and name.lower() not in component_map:
                        comp_id = f"comp_{len(components)}"
                        comp = ArchComponent(
                            id=comp_id,
                            name=name.replace('_', ' ').title(),
                            component_type=comp_type
                        )
                        components.append(comp)
                        component_map[name.lower()] = comp_id

        # Extract connections
        for pattern, conn_type in self.CONNECTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    source_name, target_name = match[0].lower(), match[1].lower()
                    source_id = component_map.get(source_name)
                    target_id = component_map.get(target_name)

                    if source_id and target_id:
                        conn = ArchConnection(
                            id=f"conn_{len(connections)}",
                            source=source_id,
                            target=target_id,
                            connection_type=conn_type
                        )
                        connections.append(conn)

        # Infer layers and position components
        self._infer_layers(components, connections)
        self._position_components(components)

        return Architecture(
            name="System Architecture",
            components=components,
            connections=connections
        )

    def _infer_layers(self, components: List[ArchComponent],
                      connections: List[ArchConnection]) -> None:
        """Infer architectural layers from component types."""
        # Default layer ordering
        layer_order = {
            ComponentType.CLIENT: 0,
            ComponentType.GATEWAY: 1,
            ComponentType.SERVICE: 2,
            ComponentType.FUNCTION: 2,
            ComponentType.CLASS: 2,
            ComponentType.CACHE: 3,
            ComponentType.QUEUE: 3,
            ComponentType.DATABASE: 4,
            ComponentType.EXTERNAL: 5,
            ComponentType.LAYER: 3,
            ComponentType.CONTAINER: 2
        }

        for comp in components:
            comp.layer = layer_order.get(comp.component_type, 2)

    def _position_components(self, components: List[ArchComponent]) -> None:
        """Position components in 3D space based on layers."""
        # Group by layer
        layers = {}
        for comp in components:
            if comp.layer not in layers:
                layers[comp.layer] = []
            layers[comp.layer].append(comp)

        # Position each layer
        layer_y_start = 4
        layer_spacing = 3.0

        for layer_num in sorted(layers.keys()):
            layer_comps = layers[layer_num]
            y = layer_y_start - layer_num * layer_spacing

            # Spread components horizontally
            n = len(layer_comps)
            spacing = 5.0
            start_x = -spacing * (n - 1) / 2

            for i, comp in enumerate(layer_comps):
                comp.position = {
                    'x': start_x + i * spacing,
                    'y': y,
                    'z': 0
                }


class ArchitectureSceneBuilder:
    """Build 3D scene data for architecture visualization."""

    def build_scene(self, arch: Architecture) -> Dict[str, Any]:
        """Convert architecture to 3D scene representation."""
        elements = []

        # Add components
        for comp in arch.components:
            elements.append(self._build_component(comp))

        # Add connections
        for conn in arch.connections:
            source = next((c for c in arch.components if c.id == conn.source), None)
            target = next((c for c in arch.components if c.id == conn.target), None)
            if source and target:
                elements.append(self._build_connection(conn, source, target))

        # Add layer labels if multiple layers
        layers_used = set(c.layer for c in arch.components)
        if len(layers_used) > 1:
            for layer in layers_used:
                elements.append(self._build_layer_label(layer, arch.components))

        return {
            'mode': 'architecture',
            'title': arch.name,
            'elements': elements,
            'camera': self._get_camera_settings(arch),
            'lighting': 'neutral',
            'background': {
                'type': 'grid',
                'colors': ['#f8fafc', '#e2e8f0']
            },
            'animation': {
                'connections': 'pulse',  # Animate data flow
                'speed': 0.5
            }
        }

    def _build_component(self, comp: ArchComponent) -> Dict[str, Any]:
        """Build component element."""
        style = COMPONENT_STYLES.get(comp.component_type, COMPONENT_STYLES[ComponentType.SERVICE])

        return {
            'id': comp.id,
            'type': 'arch_component',
            'component_type': comp.component_type.value,
            'name': comp.name,
            'description': comp.description,
            'position': comp.position,
            'size': {
                'w': 3.0,
                'h': style['height'],
                'd': 2.0
            },
            'style': {
                'color': style['color'],
                'shape': style['shape'],
                'icon': style['icon']
            },
            'layer': comp.layer
        }

    def _build_connection(self, conn: ArchConnection,
                         source: ArchComponent, target: ArchComponent) -> Dict[str, Any]:
        """Build connection element."""
        style = CONNECTION_STYLES.get(conn.connection_type, CONNECTION_STYLES[ConnectionType.DATA])

        return {
            'id': conn.id,
            'type': 'arch_connection',
            'connection_type': conn.connection_type.value,
            'source_id': conn.source,
            'target_id': conn.target,
            'source_pos': source.position,
            'target_pos': target.position,
            'label': conn.label,
            'bidirectional': conn.bidirectional,
            'style': {
                'color': style['color'],
                'dashed': style['dash'],
                'width': style['width'],
                'animated': conn.animated
            }
        }

    def _build_layer_label(self, layer: int, components: List[ArchComponent]) -> Dict[str, Any]:
        """Build layer label element."""
        layer_names = {
            0: 'Presentation',
            1: 'Gateway',
            2: 'Business Logic',
            3: 'Infrastructure',
            4: 'Data',
            5: 'External'
        }

        # Get leftmost position for this layer
        layer_comps = [c for c in components if c.layer == layer]
        if not layer_comps:
            return None

        min_x = min(c.position['x'] for c in layer_comps)
        y = layer_comps[0].position['y']

        return {
            'id': f'layer_label_{layer}',
            'type': 'text_label',
            'text': layer_names.get(layer, f'Layer {layer}'),
            'position': {'x': min_x - 4, 'y': y, 'z': 0},
            'style': {
                'color': '#64748b',
                'font_size': 14,
                'rotation': 0
            }
        }

    def _get_camera_settings(self, arch: Architecture) -> Dict[str, Any]:
        """Get appropriate camera settings for the architecture."""
        if not arch.components:
            return {
                'position': {'x': 0, 'y': 2, 'z': 15},
                'target': {'x': 0, 'y': 0, 'z': 0}
            }

        # Calculate bounding box
        min_x = min(c.position['x'] for c in arch.components)
        max_x = max(c.position['x'] for c in arch.components)
        min_y = min(c.position['y'] for c in arch.components)
        max_y = max(c.position['y'] for c in arch.components)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Distance based on spread
        spread_x = max_x - min_x
        spread_y = max_y - min_y
        distance = max(spread_x, spread_y) * 1.5 + 10

        return {
            'position': {'x': center_x, 'y': center_y + 5, 'z': distance},
            'target': {'x': center_x, 'y': center_y, 'z': 0},
            'fov': 50
        }


def extract_architecture_from_text(text: str) -> Dict[str, Any]:
    """Main entry point: extract architecture and build scene from text."""
    extractor = ArchitectureExtractor()
    builder = ArchitectureSceneBuilder()

    arch = extractor.extract_architecture(text)

    if not arch.components:
        return None

    return builder.build_scene(arch)


__all__ = [
    'ArchComponent',
    'ArchConnection',
    'Architecture',
    'ComponentType',
    'ConnectionType',
    'ArchitectureExtractor',
    'ArchitectureSceneBuilder',
    'COMPONENT_STYLES',
    'CONNECTION_STYLES',
    'extract_architecture_from_text'
]
