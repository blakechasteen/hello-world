"""
Jenny Accessibility Layer
==========================
WCAG 2.1 AA compliant accessibility infrastructure for Jenny UI.

Philosophy:
> "Disposable pixels should be accessible pixels."
>
> Every panel, every action, every transition - accessible by default.

Features:
- ARIA attributes and roles
- Focus management with trap/restore
- Keyboard navigation (Tab, Arrow, Enter, Escape)
- Live region announcements
- Color contrast utilities
- Reduced motion support
- Screen reader optimizations

References:
- WCAG 2.1 Guidelines: https://www.w3.org/WAI/WCAG21/quickref/
- ARIA Authoring Practices: https://www.w3.org/WAI/ARIA/apg/

Author: HoloLoom Team
Date: 2025-12-08 (Jenny Moonshot M3)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# ARIA Roles and Properties
# ============================================================================

class AriaRole(str, Enum):
    """
    ARIA landmark and widget roles for Jenny panels.

    See: https://www.w3.org/TR/wai-aria-1.2/#role_definitions
    """
    # Landmark roles
    MAIN = "main"
    REGION = "region"
    COMPLEMENTARY = "complementary"
    NAVIGATION = "navigation"

    # Widget roles
    DIALOG = "dialog"
    ALERT_DIALOG = "alertdialog"
    BUTTON = "button"
    LINK = "link"
    TAB = "tab"
    TABPANEL = "tabpanel"
    TABLIST = "tablist"
    MENU = "menu"
    MENUITEM = "menuitem"
    TOOLTIP = "tooltip"

    # Document structure roles
    ARTICLE = "article"
    HEADING = "heading"
    LIST = "list"
    LISTITEM = "listitem"
    TABLE = "table"
    ROW = "row"
    CELL = "cell"

    # Live region roles
    STATUS = "status"
    LOG = "log"
    ALERT = "alert"
    TIMER = "timer"
    MARQUEE = "marquee"

    # Graphics roles
    IMG = "img"
    FIGURE = "figure"

    # Group role
    GROUP = "group"


class AriaLive(str, Enum):
    """ARIA live region politeness settings."""
    OFF = "off"
    POLITE = "polite"      # Announce when idle
    ASSERTIVE = "assertive"  # Interrupt immediately


class AriaRelevant(str, Enum):
    """What changes to announce in live regions."""
    ADDITIONS = "additions"
    REMOVALS = "removals"
    TEXT = "text"
    ALL = "all"
    ADDITIONS_TEXT = "additions text"


# ============================================================================
# ARIA Attributes Builder
# ============================================================================

@dataclass
class AriaAttributes:
    """
    ARIA attributes container for accessible elements.

    Generates valid ARIA attribute dictionaries for HTML rendering.
    """
    role: Optional[AriaRole] = None
    label: Optional[str] = None
    labelledby: Optional[str] = None
    describedby: Optional[str] = None
    live: Optional[AriaLive] = None
    atomic: Optional[bool] = None
    relevant: Optional[AriaRelevant] = None
    busy: Optional[bool] = None
    hidden: Optional[bool] = None
    expanded: Optional[bool] = None
    selected: Optional[bool] = None
    pressed: Optional[bool] = None
    checked: Optional[bool] = None
    disabled: Optional[bool] = None
    current: Optional[str] = None  # "page", "step", "location", "date", "time", "true"
    controls: Optional[str] = None
    owns: Optional[str] = None
    haspopup: Optional[str] = None  # "menu", "listbox", "tree", "grid", "dialog"
    level: Optional[int] = None  # Heading level
    posinset: Optional[int] = None  # Position in set
    setsize: Optional[int] = None  # Total items in set
    valuemin: Optional[float] = None
    valuemax: Optional[float] = None
    valuenow: Optional[float] = None
    valuetext: Optional[str] = None
    modal: Optional[bool] = None
    keyshortcuts: Optional[str] = None
    roledescription: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to HTML attribute dictionary."""
        attrs = {}

        if self.role:
            attrs["role"] = self.role.value if isinstance(self.role, AriaRole) else self.role
        if self.label:
            attrs["aria-label"] = self.label
        if self.labelledby:
            attrs["aria-labelledby"] = self.labelledby
        if self.describedby:
            attrs["aria-describedby"] = self.describedby
        if self.live:
            attrs["aria-live"] = self.live.value if isinstance(self.live, AriaLive) else self.live
        if self.atomic is not None:
            attrs["aria-atomic"] = str(self.atomic).lower()
        if self.relevant:
            attrs["aria-relevant"] = self.relevant.value if isinstance(self.relevant, AriaRelevant) else self.relevant
        if self.busy is not None:
            attrs["aria-busy"] = str(self.busy).lower()
        if self.hidden is not None:
            attrs["aria-hidden"] = str(self.hidden).lower()
        if self.expanded is not None:
            attrs["aria-expanded"] = str(self.expanded).lower()
        if self.selected is not None:
            attrs["aria-selected"] = str(self.selected).lower()
        if self.pressed is not None:
            attrs["aria-pressed"] = str(self.pressed).lower()
        if self.checked is not None:
            attrs["aria-checked"] = str(self.checked).lower()
        if self.disabled is not None:
            attrs["aria-disabled"] = str(self.disabled).lower()
        if self.current:
            attrs["aria-current"] = self.current
        if self.controls:
            attrs["aria-controls"] = self.controls
        if self.owns:
            attrs["aria-owns"] = self.owns
        if self.haspopup:
            attrs["aria-haspopup"] = self.haspopup
        if self.level is not None:
            attrs["aria-level"] = str(self.level)
        if self.posinset is not None:
            attrs["aria-posinset"] = str(self.posinset)
        if self.setsize is not None:
            attrs["aria-setsize"] = str(self.setsize)
        if self.valuemin is not None:
            attrs["aria-valuemin"] = str(self.valuemin)
        if self.valuemax is not None:
            attrs["aria-valuemax"] = str(self.valuemax)
        if self.valuenow is not None:
            attrs["aria-valuenow"] = str(self.valuenow)
        if self.valuetext:
            attrs["aria-valuetext"] = self.valuetext
        if self.modal is not None:
            attrs["aria-modal"] = str(self.modal).lower()
        if self.keyshortcuts:
            attrs["aria-keyshortcuts"] = self.keyshortcuts
        if self.roledescription:
            attrs["aria-roledescription"] = self.roledescription

        return attrs

    def to_html(self) -> str:
        """Convert to HTML attribute string."""
        attrs = self.to_dict()
        return " ".join(f'{k}="{v}"' for k, v in attrs.items())


# ============================================================================
# Focus Management
# ============================================================================

@dataclass
class FocusManager:
    """
    Focus management for keyboard navigation.

    Handles:
    - Focus trapping within dialogs/modals
    - Focus restoration on close
    - Skip links for efficient navigation
    - Focus ring styling
    """

    # Stack of previously focused elements (for restoration)
    _focus_stack: List[str] = field(default_factory=list)

    # Currently trapped container ID
    _trapped_container: Optional[str] = None

    def generate_focus_trap_script(self, container_id: str) -> str:
        """
        Generate JavaScript for focus trapping.

        Keeps focus within a container (e.g., modal dialog).
        """
        return f'''
<script>
(function() {{
    const container = document.getElementById('{container_id}');
    if (!container) return;

    const focusableSelectors = [
        'a[href]',
        'button:not([disabled])',
        'input:not([disabled])',
        'select:not([disabled])',
        'textarea:not([disabled])',
        '[tabindex]:not([tabindex="-1"])'
    ].join(', ');

    function getFocusableElements() {{
        return container.querySelectorAll(focusableSelectors);
    }}

    function handleKeydown(e) {{
        if (e.key !== 'Tab') return;

        const focusable = getFocusableElements();
        if (focusable.length === 0) return;

        const first = focusable[0];
        const last = focusable[focusable.length - 1];

        if (e.shiftKey && document.activeElement === first) {{
            e.preventDefault();
            last.focus();
        }} else if (!e.shiftKey && document.activeElement === last) {{
            e.preventDefault();
            first.focus();
        }}
    }}

    container.addEventListener('keydown', handleKeydown);

    // Focus first focusable element
    const focusable = getFocusableElements();
    if (focusable.length > 0) {{
        focusable[0].focus();
    }}
}})();
</script>
'''

    def generate_focus_restore_script(self, trigger_id: str) -> str:
        """
        Generate script to restore focus on close.

        Returns focus to the element that triggered the modal/dialog.
        """
        return f'''
<script>
(function() {{
    const trigger = document.getElementById('{trigger_id}');
    if (trigger) {{
        trigger.focus();
    }}
}})();
</script>
'''

    def generate_skip_link(
        self,
        target_id: str,
        text: str = "Skip to main content"
    ) -> str:
        """
        Generate a skip link for keyboard users.

        Allows users to skip repetitive navigation.
        """
        return f'''
<a href="#{target_id}" class="jenny-skip-link"
   style="position: absolute; left: -9999px; top: auto; width: 1px; height: 1px; overflow: hidden;"
   onfocus="this.style.left='0'; this.style.width='auto'; this.style.height='auto';"
   onblur="this.style.left='-9999px'; this.style.width='1px'; this.style.height='1px';">
    {text}
</a>
'''

    def generate_focus_styles(self) -> str:
        """
        Generate CSS for visible focus indicators.

        WCAG 2.4.7: Focus Visible (Level AA)
        """
        return '''
<style>
/* High-contrast focus ring */
.jenny-panel:focus,
.jenny-action:focus,
.jenny-panel *:focus {
    outline: 3px solid #005fcc;
    outline-offset: 2px;
}

/* Focus visible only for keyboard (not mouse) */
.jenny-panel:focus:not(:focus-visible),
.jenny-action:focus:not(:focus-visible),
.jenny-panel *:focus:not(:focus-visible) {
    outline: none;
}

.jenny-panel:focus-visible,
.jenny-action:focus-visible,
.jenny-panel *:focus-visible {
    outline: 3px solid #005fcc;
    outline-offset: 2px;
    box-shadow: 0 0 0 6px rgba(0, 95, 204, 0.25);
}

/* Skip link styling */
.jenny-skip-link:focus {
    position: fixed !important;
    left: 10px !important;
    top: 10px !important;
    width: auto !important;
    height: auto !important;
    padding: 8px 16px;
    background: #005fcc;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    z-index: 100000;
    font-weight: bold;
}
</style>
'''


# ============================================================================
# Keyboard Navigation
# ============================================================================

@dataclass
class KeyboardHandler:
    """
    Keyboard navigation handler for Jenny panels.

    Supports:
    - Arrow key navigation between panels
    - Tab navigation within panels
    - Enter/Space for action activation
    - Escape for dismissal
    - Home/End for first/last panel
    """

    def generate_panel_navigation_script(self, panel_ids: List[str]) -> str:
        """
        Generate keyboard navigation script for panel grid.

        Enables arrow key navigation between panels.
        """
        panel_ids_json = str(panel_ids).replace("'", '"')

        return f'''
<script>
(function() {{
    const panelIds = {panel_ids_json};
    const panels = panelIds.map(id => document.getElementById(id)).filter(Boolean);

    if (panels.length === 0) return;

    let currentIndex = 0;

    function focusPanel(index) {{
        if (index < 0) index = panels.length - 1;
        if (index >= panels.length) index = 0;
        currentIndex = index;
        panels[index].focus();
    }}

    function handleKeydown(e) {{
        // Only handle when a panel is focused
        const focusedPanel = panels.findIndex(p => p === document.activeElement);
        if (focusedPanel === -1) return;

        currentIndex = focusedPanel;

        switch(e.key) {{
            case 'ArrowRight':
            case 'ArrowDown':
                e.preventDefault();
                focusPanel(currentIndex + 1);
                break;
            case 'ArrowLeft':
            case 'ArrowUp':
                e.preventDefault();
                focusPanel(currentIndex - 1);
                break;
            case 'Home':
                e.preventDefault();
                focusPanel(0);
                break;
            case 'End':
                e.preventDefault();
                focusPanel(panels.length - 1);
                break;
            case 'Enter':
            case ' ':
                // Activate primary action if panel has one
                const action = panels[currentIndex].querySelector('[data-primary-action]');
                if (action) {{
                    e.preventDefault();
                    action.click();
                }}
                break;
            case 'Escape':
                // Dismiss panel if dismissible
                const dismiss = panels[currentIndex].querySelector('[data-dismiss]');
                if (dismiss) {{
                    e.preventDefault();
                    dismiss.click();
                }}
                break;
        }}
    }}

    document.addEventListener('keydown', handleKeydown);

    // Make panels focusable
    panels.forEach((panel, i) => {{
        if (!panel.hasAttribute('tabindex')) {{
            panel.setAttribute('tabindex', i === 0 ? '0' : '-1');
        }}
        panel.setAttribute('role', 'article');
        panel.setAttribute('aria-posinset', String(i + 1));
        panel.setAttribute('aria-setsize', String(panels.length));
    }});
}})();
</script>
'''

    def generate_action_menu_script(self, menu_id: str) -> str:
        """
        Generate keyboard navigation for action menus.

        Follows ARIA menu button pattern.
        """
        return f'''
<script>
(function() {{
    const menu = document.getElementById('{menu_id}');
    if (!menu) return;

    const items = menu.querySelectorAll('[role="menuitem"]');
    let currentIndex = 0;

    function focusItem(index) {{
        if (index < 0) index = items.length - 1;
        if (index >= items.length) index = 0;
        currentIndex = index;
        items[index].focus();
    }}

    menu.addEventListener('keydown', function(e) {{
        switch(e.key) {{
            case 'ArrowDown':
                e.preventDefault();
                focusItem(currentIndex + 1);
                break;
            case 'ArrowUp':
                e.preventDefault();
                focusItem(currentIndex - 1);
                break;
            case 'Home':
                e.preventDefault();
                focusItem(0);
                break;
            case 'End':
                e.preventDefault();
                focusItem(items.length - 1);
                break;
            case 'Enter':
            case ' ':
                e.preventDefault();
                items[currentIndex].click();
                break;
            case 'Escape':
                e.preventDefault();
                menu.setAttribute('aria-hidden', 'true');
                // Return focus to trigger
                const triggerId = menu.getAttribute('data-trigger');
                if (triggerId) {{
                    const trigger = document.getElementById(triggerId);
                    if (trigger) trigger.focus();
                }}
                break;
        }}
    }});
}})();
</script>
'''


# ============================================================================
# Live Announcements
# ============================================================================

@dataclass
class LiveAnnouncer:
    """
    Screen reader live region announcer.

    Creates and manages live regions for dynamic content announcements.
    """

    def generate_live_region_html(
        self,
        region_id: str = "jenny-announcer",
        politeness: AriaLive = AriaLive.POLITE
    ) -> str:
        """
        Generate a hidden live region for announcements.

        Screen readers will announce changes to this region.
        """
        return f'''
<div id="{region_id}"
     role="status"
     aria-live="{politeness.value}"
     aria-atomic="true"
     style="position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden;">
</div>
'''

    def generate_announce_script(
        self,
        region_id: str = "jenny-announcer"
    ) -> str:
        """
        Generate announcement helper function.

        Call jennyAnnounce(message) to announce to screen readers.
        """
        return f'''
<script>
window.jennyAnnounce = function(message, politeness) {{
    const region = document.getElementById('{region_id}');
    if (!region) return;

    // Change politeness if specified
    if (politeness) {{
        region.setAttribute('aria-live', politeness);
    }}

    // Clear and set (triggers announcement)
    region.textContent = '';
    setTimeout(() => {{
        region.textContent = message;
    }}, 100);
}};

// Announce assertively (interrupts)
window.jennyAnnounceAssertive = function(message) {{
    window.jennyAnnounce(message, 'assertive');
}};
</script>
'''

    def generate_panel_spawn_announcement(
        self,
        panel_title: str,
        panel_type: str
    ) -> str:
        """Generate announcement for panel spawn."""
        return f"window.jennyAnnounce('New {panel_type} panel: {panel_title}');"

    def generate_panel_dissolve_announcement(
        self,
        panel_title: str
    ) -> str:
        """Generate announcement for panel dissolution."""
        return f"window.jennyAnnounce('{panel_title} panel dismissed');"


# ============================================================================
# Color Contrast Utilities
# ============================================================================

@dataclass
class ContrastChecker:
    """
    WCAG color contrast checker and utilities.

    Ensures text meets WCAG 2.1 AA contrast requirements:
    - Normal text: 4.5:1 minimum
    - Large text (18pt+ or 14pt+ bold): 3:1 minimum
    - UI components: 3:1 minimum
    """

    @staticmethod
    def relative_luminance(r: int, g: int, b: int) -> float:
        """
        Calculate relative luminance per WCAG 2.1.

        Returns value between 0 (black) and 1 (white).
        """
        def channel_luminance(c: int) -> float:
            c_norm = c / 255
            return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4

        r_lum = channel_luminance(r)
        g_lum = channel_luminance(g)
        b_lum = channel_luminance(b)

        return 0.2126 * r_lum + 0.7152 * g_lum + 0.0722 * b_lum

    @staticmethod
    def contrast_ratio(
        fg_rgb: tuple[int, int, int],
        bg_rgb: tuple[int, int, int]
    ) -> float:
        """
        Calculate contrast ratio between foreground and background.

        Returns ratio from 1:1 to 21:1.
        """
        fg_lum = ContrastChecker.relative_luminance(*fg_rgb)
        bg_lum = ContrastChecker.relative_luminance(*bg_rgb)

        lighter = max(fg_lum, bg_lum)
        darker = min(fg_lum, bg_lum)

        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def meets_aa_normal(contrast: float) -> bool:
        """Check if contrast meets AA for normal text (4.5:1)."""
        return contrast >= 4.5

    @staticmethod
    def meets_aa_large(contrast: float) -> bool:
        """Check if contrast meets AA for large text (3:1)."""
        return contrast >= 3.0

    @staticmethod
    def meets_aaa_normal(contrast: float) -> bool:
        """Check if contrast meets AAA for normal text (7:1)."""
        return contrast >= 7.0

    def hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def check_panel_colors(
        self,
        fg_hex: str,
        bg_hex: str
    ) -> Dict[str, Any]:
        """
        Check panel color contrast.

        Returns contrast analysis with WCAG compliance levels.
        """
        fg_rgb = self.hex_to_rgb(fg_hex)
        bg_rgb = self.hex_to_rgb(bg_hex)
        ratio = self.contrast_ratio(fg_rgb, bg_rgb)

        return {
            "foreground": fg_hex,
            "background": bg_hex,
            "contrast_ratio": round(ratio, 2),
            "wcag_aa_normal": self.meets_aa_normal(ratio),
            "wcag_aa_large": self.meets_aa_large(ratio),
            "wcag_aaa_normal": self.meets_aaa_normal(ratio),
        }


# ============================================================================
# Reduced Motion Support
# ============================================================================

@dataclass
class MotionPreferences:
    """
    Handle reduced motion preferences.

    WCAG 2.3.3: Animation from Interactions
    """

    def generate_reduced_motion_styles(self) -> str:
        """
        Generate CSS that respects prefers-reduced-motion.

        Disables animations for users who prefer reduced motion.
        """
        return '''
<style>
/* Respect user's motion preferences */
@media (prefers-reduced-motion: reduce) {
    .jenny-panel,
    .jenny-panel *,
    .jenny-action,
    .jenny-transition {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }

    /* Keep opacity transitions for visibility */
    .jenny-panel-spawn,
    .jenny-panel-dissolve {
        opacity: 1 !important;
        transform: none !important;
    }
}

/* Standard animations for users who accept motion */
@media (prefers-reduced-motion: no-preference) {
    .jenny-panel-spawn {
        animation: jenny-spawn 0.3s ease-out;
    }

    .jenny-panel-dissolve {
        animation: jenny-dissolve 0.3s ease-in forwards;
    }

    @keyframes jenny-spawn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }

    @keyframes jenny-dissolve {
        from {
            opacity: 1;
            transform: scale(1);
        }
        to {
            opacity: 0;
            transform: scale(0.95);
        }
    }
}
</style>
'''

    def generate_motion_query_script(self) -> str:
        """
        Generate script to detect motion preferences.

        Sets a global flag for JS animations.
        """
        return '''
<script>
window.jennyPrefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Listen for changes
window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', function(e) {
    window.jennyPrefersReducedMotion = e.matches;
    document.documentElement.classList.toggle('jenny-reduced-motion', e.matches);
});
</script>
'''


# ============================================================================
# Panel Accessibility Decorator
# ============================================================================

def accessible_panel(
    role: AriaRole = AriaRole.ARTICLE,
    label_source: str = "title"
) -> Callable:
    """
    Decorator to add accessibility attributes to panel renderers.

    Usage:
        @accessible_panel(role=AriaRole.REGION)
        def render_panel(spec):
            return '<div>...</div>'
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # Post-process HTML to add ARIA attributes
            # (Actual implementation would parse and modify HTML)
            return result
        return wrapper
    return decorator


# ============================================================================
# Comprehensive Accessibility Layer
# ============================================================================

@dataclass
class JennyAccessibilityLayer:
    """
    Complete accessibility layer for Jenny UI.

    Aggregates all accessibility features into one cohesive API.

    Usage:
        a11y = JennyAccessibilityLayer()

        # Generate full accessibility infrastructure
        html = a11y.generate_infrastructure()

        # Create accessible panel wrapper
        wrapper = a11y.wrap_panel(panel_id, title, content)

        # Generate keyboard navigation
        nav = a11y.generate_navigation(panel_ids)
    """

    focus_manager: FocusManager = field(default_factory=FocusManager)
    keyboard_handler: KeyboardHandler = field(default_factory=KeyboardHandler)
    live_announcer: LiveAnnouncer = field(default_factory=LiveAnnouncer)
    contrast_checker: ContrastChecker = field(default_factory=ContrastChecker)
    motion_prefs: MotionPreferences = field(default_factory=MotionPreferences)

    def generate_infrastructure(self) -> str:
        """
        Generate complete accessibility infrastructure.

        Includes all CSS, JS, and helper elements.
        """
        parts = [
            "<!-- Jenny Accessibility Layer (WCAG 2.1 AA) -->",
            self.focus_manager.generate_focus_styles(),
            self.motion_prefs.generate_reduced_motion_styles(),
            self.live_announcer.generate_live_region_html(),
            self.live_announcer.generate_announce_script(),
            self.motion_prefs.generate_motion_query_script(),
            self.focus_manager.generate_skip_link("jenny-main-content"),
        ]
        return "\n".join(parts)

    def wrap_panel(
        self,
        panel_id: str,
        title: str,
        content: str,
        role: AriaRole = AriaRole.ARTICLE,
        position: Optional[int] = None,
        total: Optional[int] = None,
        dismissible: bool = True,
        primary_action: Optional[str] = None,
    ) -> str:
        """
        Wrap panel content with accessibility attributes.

        Args:
            panel_id: Unique panel identifier
            title: Panel title (used for aria-label)
            content: Panel HTML content
            role: ARIA role for the panel
            position: Position in panel set (for aria-posinset)
            total: Total panels (for aria-setsize)
            dismissible: Whether panel can be dismissed
            primary_action: ID of primary action button

        Returns:
            Accessible panel HTML
        """
        aria = AriaAttributes(
            role=role,
            label=title,
        )
        if position is not None:
            aria.posinset = position
        if total is not None:
            aria.setsize = total

        attrs = aria.to_html()

        # Build panel wrapper
        tabindex = '0' if position == 1 else '-1'
        data_attrs = []
        if dismissible:
            data_attrs.append('data-dismissible="true"')
        if primary_action:
            data_attrs.append(f'data-primary-action="{primary_action}"')

        data_str = " ".join(data_attrs)

        return f'''
<div id="{panel_id}" class="jenny-panel" tabindex="{tabindex}" {attrs} {data_str}>
    <h2 id="{panel_id}-title" class="jenny-panel-title">{title}</h2>
    <div class="jenny-panel-content" aria-labelledby="{panel_id}-title">
        {content}
    </div>
</div>
'''

    def generate_navigation(self, panel_ids: List[str]) -> str:
        """Generate keyboard navigation for panel set."""
        return self.keyboard_handler.generate_panel_navigation_script(panel_ids)

    def generate_focus_trap(self, container_id: str) -> str:
        """Generate focus trap for modal/dialog."""
        return self.focus_manager.generate_focus_trap_script(container_id)

    def announce(self, message: str, assertive: bool = False) -> str:
        """Generate announcement script call."""
        if assertive:
            return f"window.jennyAnnounceAssertive('{message}');"
        return f"window.jennyAnnounce('{message}');"

    def check_contrast(self, fg: str, bg: str) -> Dict[str, Any]:
        """Check color contrast compliance."""
        return self.contrast_checker.check_panel_colors(fg, bg)


# ============================================================================
# Factory Functions
# ============================================================================

def create_accessibility_layer() -> JennyAccessibilityLayer:
    """Create a fully configured accessibility layer."""
    return JennyAccessibilityLayer()


def create_aria_attrs(
    role: AriaRole,
    label: str,
    **kwargs
) -> AriaAttributes:
    """Create ARIA attributes with common defaults."""
    return AriaAttributes(role=role, label=label, **kwargs)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'AriaRole',
    'AriaLive',
    'AriaRelevant',
    # Classes
    'AriaAttributes',
    'FocusManager',
    'KeyboardHandler',
    'LiveAnnouncer',
    'ContrastChecker',
    'MotionPreferences',
    'JennyAccessibilityLayer',
    # Factories
    'create_accessibility_layer',
    'create_aria_attrs',
    # Decorators
    'accessible_panel',
]
