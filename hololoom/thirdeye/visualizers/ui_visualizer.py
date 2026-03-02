"""
UI Design Visualizer
====================

Renders actual UI mockups, wireframes, and component hierarchies
as 3D floating panels that you can inspect and navigate.

When discussing UI design, this creates visual representations
of buttons, forms, layouts, navigation, etc.

Created: 2025-12-01
Author: HoloLoom Team
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import re
import json


@dataclass
class UIComponent:
    """A UI component extracted from conversation."""
    id: str
    component_type: str  # 'button', 'input', 'card', 'navbar', etc.
    label: str
    props: Dict[str, Any] = field(default_factory=dict)
    children: List['UIComponent'] = field(default_factory=list)
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    size: Dict[str, float] = field(default_factory=lambda: {"w": 100, "h": 40})


@dataclass
class UIScreen:
    """A complete UI screen/page."""
    id: str
    name: str
    components: List[UIComponent]
    layout: str = "vertical"  # 'vertical', 'horizontal', 'grid', 'absolute'
    background: str = "#ffffff"
    width: int = 375  # Mobile-first default
    height: int = 812


# Component templates with CSS
COMPONENT_TEMPLATES = {
    'button': {
        'html': '<button class="ui-button {variant}">{label}</button>',
        'css': '''
            .ui-button {
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            .ui-button.primary {
                background: #3b82f6;
                color: white;
                border: none;
            }
            .ui-button.secondary {
                background: white;
                color: #1e293b;
                border: 2px solid #e2e8f0;
            }
        ''',
        'default_size': {'w': 120, 'h': 44}
    },
    'input': {
        'html': '<div class="ui-input-group"><label>{label}</label><input type="{type}" placeholder="{placeholder}"/></div>',
        'css': '''
            .ui-input-group {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }
            .ui-input-group label {
                font-size: 14px;
                font-weight: 500;
                color: #374151;
            }
            .ui-input-group input {
                padding: 12px 16px;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                font-size: 16px;
            }
        ''',
        'default_size': {'w': 280, 'h': 70}
    },
    'card': {
        'html': '''
            <div class="ui-card">
                <div class="card-header">{title}</div>
                <div class="card-body">{content}</div>
            </div>
        ''',
        'css': '''
            .ui-card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .card-header {
                padding: 16px 20px;
                font-weight: 600;
                border-bottom: 1px solid #e5e7eb;
            }
            .card-body {
                padding: 20px;
            }
        ''',
        'default_size': {'w': 320, 'h': 200}
    },
    'navbar': {
        'html': '''
            <nav class="ui-navbar">
                <div class="nav-brand">{brand}</div>
                <div class="nav-links">{links}</div>
            </nav>
        ''',
        'css': '''
            .ui-navbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 24px;
                background: white;
                border-bottom: 1px solid #e5e7eb;
            }
            .nav-brand {
                font-weight: 700;
                font-size: 20px;
            }
            .nav-links {
                display: flex;
                gap: 24px;
            }
        ''',
        'default_size': {'w': 375, 'h': 64}
    },
    'sidebar': {
        'html': '''
            <aside class="ui-sidebar">
                <div class="sidebar-header">{header}</div>
                <ul class="sidebar-menu">{items}</ul>
            </aside>
        ''',
        'css': '''
            .ui-sidebar {
                width: 260px;
                background: #1e293b;
                color: white;
                height: 100%;
            }
            .sidebar-header {
                padding: 20px;
                font-weight: 600;
            }
            .sidebar-menu {
                list-style: none;
                padding: 0;
            }
            .sidebar-menu li {
                padding: 12px 20px;
                cursor: pointer;
            }
            .sidebar-menu li:hover {
                background: rgba(255,255,255,0.1);
            }
        ''',
        'default_size': {'w': 260, 'h': 600}
    },
    'modal': {
        'html': '''
            <div class="ui-modal-backdrop">
                <div class="ui-modal">
                    <div class="modal-header">
                        <h2>{title}</h2>
                        <button class="close">×</button>
                    </div>
                    <div class="modal-body">{content}</div>
                    <div class="modal-footer">{actions}</div>
                </div>
            </div>
        ''',
        'css': '''
            .ui-modal-backdrop {
                position: absolute;
                inset: 0;
                background: rgba(0,0,0,0.5);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .ui-modal {
                background: white;
                border-radius: 16px;
                width: 90%;
                max-width: 480px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            }
            .modal-header {
                display: flex;
                justify-content: space-between;
                padding: 20px;
                border-bottom: 1px solid #e5e7eb;
            }
            .modal-body {
                padding: 24px 20px;
            }
            .modal-footer {
                padding: 16px 20px;
                background: #f9fafb;
                display: flex;
                justify-content: flex-end;
                gap: 12px;
            }
        ''',
        'default_size': {'w': 480, 'h': 320}
    },
    'form': {
        'html': '''
            <form class="ui-form">
                <h3>{title}</h3>
                <div class="form-fields">{fields}</div>
                <div class="form-actions">{actions}</div>
            </form>
        ''',
        'css': '''
            .ui-form {
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }
            .ui-form h3 {
                margin-bottom: 20px;
                color: #1e293b;
            }
            .form-fields {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .form-actions {
                margin-top: 24px;
                display: flex;
                gap: 12px;
            }
        ''',
        'default_size': {'w': 400, 'h': 400}
    },
    'list': {
        'html': '''
            <div class="ui-list">
                <div class="list-header">{title}</div>
                <ul class="list-items">{items}</ul>
            </div>
        ''',
        'css': '''
            .ui-list {
                background: white;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
            .list-header {
                padding: 16px;
                font-weight: 600;
                border-bottom: 1px solid #e5e7eb;
            }
            .list-items {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            .list-items li {
                padding: 16px;
                border-bottom: 1px solid #f3f4f6;
                display: flex;
                align-items: center;
                gap: 12px;
            }
        ''',
        'default_size': {'w': 360, 'h': 300}
    },
    'header': {
        'html': '''
            <header class="ui-header">
                <h1>{title}</h1>
                <p class="subtitle">{subtitle}</p>
            </header>
        ''',
        'css': '''
            .ui-header {
                text-align: center;
                padding: 40px 20px;
            }
            .ui-header h1 {
                font-size: 32px;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 8px;
            }
            .ui-header .subtitle {
                color: #64748b;
                font-size: 18px;
            }
        ''',
        'default_size': {'w': 375, 'h': 140}
    },
    'tab': {
        'html': '''
            <div class="ui-tabs">
                <div class="tab-list">{tabs}</div>
                <div class="tab-panel">{content}</div>
            </div>
        ''',
        'css': '''
            .ui-tabs {
                background: white;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
            .tab-list {
                display: flex;
                border-bottom: 1px solid #e5e7eb;
            }
            .tab-list button {
                flex: 1;
                padding: 16px;
                background: none;
                border: none;
                cursor: pointer;
            }
            .tab-list button.active {
                border-bottom: 2px solid #3b82f6;
                color: #3b82f6;
            }
            .tab-panel {
                padding: 20px;
            }
        ''',
        'default_size': {'w': 360, 'h': 280}
    }
}


class UIExtractor:
    """Extract UI components from natural language."""

    # Patterns to detect UI components
    COMPONENT_PATTERNS = {
        'button': r'button\s*(?:that\s+)?(?:says?\s+)?["\']?([^"\']+)["\']?|([a-z]+)\s+button',
        'input': r'(?:text\s+)?(?:input|field)\s+(?:for\s+)?([a-z\s]+)|([a-z\s]+)\s+input',
        'card': r'card\s+(?:with|showing|for)\s+([a-z\s]+)',
        'navbar': r'nav(?:bar|igation)?\s+(?:with|showing)\s+([a-z\s,]+)',
        'sidebar': r'sidebar\s+(?:with|containing)\s+([a-z\s,]+)',
        'modal': r'modal\s+(?:for|with|saying)\s+([a-z\s]+)',
        'form': r'form\s+(?:for|with|to)\s+([a-z\s]+)',
        'list': r'list\s+(?:of|showing)\s+([a-z\s]+)',
        'header': r'header\s+(?:with|saying)\s+["\']?([^"\']+)["\']?',
    }

    def extract_components(self, text: str) -> List[UIComponent]:
        """Extract UI components from text description."""
        components = []
        text_lower = text.lower()

        # Check for each component type
        for comp_type, pattern in self.COMPONENT_PATTERNS.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                label = match[0] if isinstance(match, tuple) else match
                label = label.strip() if label else comp_type

                comp = UIComponent(
                    id=f"{comp_type}_{len(components)}",
                    component_type=comp_type,
                    label=label.title(),
                    size=COMPONENT_TEMPLATES.get(comp_type, {}).get('default_size', {'w': 100, 'h': 40})
                )
                components.append(comp)

        # If no specific components found, try to infer from context
        if not components:
            components = self._infer_components(text_lower)

        return components

    def _infer_components(self, text: str) -> List[UIComponent]:
        """Infer components when no explicit patterns match."""
        components = []

        # Common UI keywords
        if 'login' in text or 'sign in' in text:
            components.extend([
                UIComponent(id="header_0", component_type="header", label="Login",
                           size={'w': 375, 'h': 80}),
                UIComponent(id="input_0", component_type="input", label="Email",
                           props={'type': 'email', 'placeholder': 'Enter your email'},
                           size={'w': 320, 'h': 70}),
                UIComponent(id="input_1", component_type="input", label="Password",
                           props={'type': 'password', 'placeholder': 'Enter password'},
                           size={'w': 320, 'h': 70}),
                UIComponent(id="button_0", component_type="button", label="Sign In",
                           props={'variant': 'primary'},
                           size={'w': 320, 'h': 48})
            ])
        elif 'dashboard' in text:
            components.extend([
                UIComponent(id="navbar_0", component_type="navbar", label="Dashboard",
                           props={'brand': 'AppName'},
                           size={'w': 375, 'h': 64}),
                UIComponent(id="sidebar_0", component_type="sidebar", label="Navigation",
                           size={'w': 260, 'h': 500}),
                UIComponent(id="card_0", component_type="card", label="Stats Card",
                           props={'title': 'Overview'},
                           size={'w': 300, 'h': 150}),
                UIComponent(id="card_1", component_type="card", label="Activity",
                           props={'title': 'Recent Activity'},
                           size={'w': 300, 'h': 200})
            ])
        elif 'settings' in text or 'preferences' in text:
            components.extend([
                UIComponent(id="header_0", component_type="header", label="Settings",
                           size={'w': 375, 'h': 80}),
                UIComponent(id="list_0", component_type="list", label="Settings Menu",
                           size={'w': 360, 'h': 400})
            ])

        return components


class UISceneBuilder:
    """Build 3D scene data for UI visualization."""

    def build_scene(self, screen: UIScreen) -> Dict[str, Any]:
        """Convert UI screen to 3D scene representation."""
        elements = []
        y_offset = 0

        for comp in screen.components:
            template = COMPONENT_TEMPLATES.get(comp.component_type, {})

            # Generate HTML for the component
            html = self._render_component_html(comp, template)
            css = template.get('css', '')

            # Position in 3D space (panels floating in space)
            element = {
                'id': comp.id,
                'type': 'ui_panel',
                'component_type': comp.component_type,
                'label': comp.label,
                'position': {
                    'x': comp.position.get('x', 0),
                    'y': -y_offset,
                    'z': comp.position.get('z', 0)
                },
                'size': comp.size,
                'content': {
                    'html': html,
                    'css': css
                },
                'props': comp.props
            }
            elements.append(element)

            # Stack vertically with spacing
            y_offset += (comp.size.get('h', 40) / 100) + 0.5

        return {
            'mode': 'ui_design',
            'title': screen.name,
            'screen': {
                'width': screen.width,
                'height': screen.height,
                'background': screen.background
            },
            'elements': elements,
            'camera': {
                'position': {'x': 0, 'y': -y_offset/2, 'z': 8},
                'target': {'x': 0, 'y': -y_offset/2, 'z': 0}
            },
            'lighting': 'soft',
            'background': {
                'type': 'gradient',
                'colors': ['#f0f4f8', '#d9e2ec']
            }
        }

    def _render_component_html(self, comp: UIComponent, template: Dict) -> str:
        """Render component to HTML string."""
        html_template = template.get('html', '<div>{label}</div>')

        # Replace placeholders
        html = html_template.format(
            label=comp.label,
            title=comp.props.get('title', comp.label),
            content=comp.props.get('content', ''),
            brand=comp.props.get('brand', 'Brand'),
            links=comp.props.get('links', '<a>Home</a><a>About</a><a>Contact</a>'),
            items=comp.props.get('items', '<li>Item 1</li><li>Item 2</li><li>Item 3</li>'),
            fields=comp.props.get('fields', ''),
            actions=comp.props.get('actions', '<button>Submit</button>'),
            tabs=comp.props.get('tabs', '<button class="active">Tab 1</button><button>Tab 2</button>'),
            header=comp.props.get('header', comp.label),
            subtitle=comp.props.get('subtitle', ''),
            type=comp.props.get('type', 'text'),
            placeholder=comp.props.get('placeholder', 'Enter text...'),
            variant=comp.props.get('variant', 'primary')
        )

        return html


def extract_ui_from_text(text: str) -> Dict[str, Any]:
    """Main entry point: extract UI and build scene from text."""
    extractor = UIExtractor()
    builder = UISceneBuilder()

    components = extractor.extract_components(text)

    if not components:
        return None

    screen = UIScreen(
        id="screen_0",
        name="UI Design",
        components=components
    )

    return builder.build_scene(screen)


__all__ = [
    'UIComponent',
    'UIScreen',
    'UIExtractor',
    'UISceneBuilder',
    'COMPONENT_TEMPLATES',
    'extract_ui_from_text'
]
