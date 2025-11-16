"""
HoloLoom Internationalization (i18n) System
=============================================

Provides translation support for the HoloLoom dashboard across 6 languages:
- English (en) - Default
- Spanish (es)
- French (fr)
- German (de)
- Chinese Simplified (zh)
- Japanese (ja)

Usage:
    from HoloLoom.i18n import TranslationManager

    i18n = TranslationManager()
    title = i18n.get("app.title", locale="es")  # "Panel de control HoloLoom Promptly"

Created: 2025-11-16
"""

from .translations import TranslationManager

__all__ = ["TranslationManager"]
