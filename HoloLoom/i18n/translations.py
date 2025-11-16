"""
Translation Manager for HoloLoom Dashboard
============================================

Simple JSON-based translation system with support for:
- Multiple locales (6 languages)
- Nested key access (dot notation)
- Parameter formatting
- Fallback to default locale
- Available locale enumeration

Author: HoloLoom Team
Created: 2025-11-16
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class TranslationManager:
    """
    Manages translations for HoloLoom dashboard across multiple locales.

    Supported locales:
    - en: English (default)
    - es: Spanish
    - fr: French
    - de: German
    - zh: Chinese (Simplified)
    - ja: Japanese
    """

    def __init__(self, default_locale: str = "en"):
        """
        Initialize translation manager.

        Args:
            default_locale: Default locale to use as fallback (default: "en")
        """
        self.default_locale = default_locale
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.locales_dir = Path(__file__).parent / "locales"

        # Load all translation files
        self._load_all_translations()

        logger.info(
            f"TranslationManager initialized with {len(self.translations)} locales. "
            f"Default: {default_locale}"
        )

    def _load_all_translations(self):
        """Load all translation files from locales directory."""
        if not self.locales_dir.exists():
            logger.warning(f"Locales directory not found: {self.locales_dir}")
            return

        for locale_file in sorted(self.locales_dir.glob("*.json")):
            locale = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self.translations[locale] = json.load(f)
                logger.debug(f"Loaded translations for locale: {locale}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load translations for {locale}: {e}")

    def get(
        self,
        key: str,
        locale: str = "en",
        default: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get translated string with optional formatting.

        Supports:
        - Nested keys with dot notation: "metrics.total_queries"
        - Parameter formatting: "Welcome, {username}!" → "Welcome, Alice!"
        - Automatic fallback to default locale if key not found

        Args:
            key: Translation key (supports dot notation for nested keys)
            locale: Target locale (falls back to default if not found)
            default: Default value if translation not found
            **kwargs: Named parameters for string formatting

        Returns:
            Translated string, or default/key if translation not found

        Example:
            >>> i18n.get("app.title", locale="es")
            "Panel de control HoloLoom Promptly"

            >>> i18n.get("welcome", locale="en", username="Alice")
            "Welcome, Alice!"
        """
        # Try requested locale, fall back to default
        trans_dict = self.translations.get(locale)

        if not trans_dict and locale != self.default_locale:
            trans_dict = self.translations.get(self.default_locale)
            logger.debug(f"Locale '{locale}' not found, using default '{self.default_locale}'")

        if not trans_dict:
            # No translations available at all
            if default is not None:
                return default
            return key

        # Navigate nested keys (e.g., "metrics.total_queries")
        keys = key.split('.')
        value = trans_dict

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                # Key path broken, return default/key
                if default is not None:
                    return default
                return key

        if value is None:
            # Key not found in translation
            if default is not None:
                return default
            # Fallback to default locale if available
            if locale != self.default_locale:
                return self.get(key, locale=self.default_locale, default=default, **kwargs)
            return key

        if not isinstance(value, str):
            # Value is not a string (probably a dict), return default/key
            if default is not None:
                return default
            return key

        # Format with kwargs if provided
        if kwargs:
            try:
                return value.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing format parameter '{e}' for key '{key}'")
                return value

        return value

    def get_all(self, locale: str = "en") -> Dict[str, Any]:
        """
        Get all translations for a locale.

        Args:
            locale: Target locale

        Returns:
            Dictionary of all translations for the locale

        Example:
            >>> translations = i18n.get_all("es")
            >>> translations["app"]["title"]
            "Panel de control HoloLoom Promptly"
        """
        trans_dict = self.translations.get(locale)

        if not trans_dict and locale != self.default_locale:
            logger.debug(f"Locale '{locale}' not found, using default '{self.default_locale}'")
            trans_dict = self.translations.get(self.default_locale, {})

        return trans_dict or {}

    def available_locales(self) -> list:
        """
        Get list of available locales.

        Returns:
            List of locale codes (e.g., ["en", "es", "fr", "de", "zh", "ja"])

        Example:
            >>> i18n.available_locales()
            ['de', 'en', 'es', 'fr', 'ja', 'zh']
        """
        return sorted(list(self.translations.keys()))

    def locale_exists(self, locale: str) -> bool:
        """
        Check if a locale is available.

        Args:
            locale: Locale code to check

        Returns:
            True if locale is available, False otherwise
        """
        return locale in self.translations

    def get_locale_metadata(self, locale: str) -> Dict[str, str]:
        """
        Get metadata about a locale (name, native name, flag emoji).

        Args:
            locale: Locale code

        Returns:
            Dictionary with locale metadata

        Example:
            >>> i18n.get_locale_metadata("es")
            {
                "code": "es",
                "name": "Spanish",
                "native": "Español",
                "flag": "🇪🇸"
            }
        """
        metadata = {
            "en": {
                "code": "en",
                "name": "English",
                "native": "English",
                "flag": "🇬🇧"
            },
            "es": {
                "code": "es",
                "name": "Spanish",
                "native": "Español",
                "flag": "🇪🇸"
            },
            "fr": {
                "code": "fr",
                "name": "French",
                "native": "Français",
                "flag": "🇫🇷"
            },
            "de": {
                "code": "de",
                "name": "German",
                "native": "Deutsch",
                "flag": "🇩🇪"
            },
            "zh": {
                "code": "zh",
                "name": "Chinese (Simplified)",
                "native": "简体中文",
                "flag": "🇨🇳"
            },
            "ja": {
                "code": "ja",
                "name": "Japanese",
                "native": "日本語",
                "flag": "🇯🇵"
            }
        }
        return metadata.get(locale, {"code": locale, "name": locale, "native": locale, "flag": "🌍"})

    def get_language_selector_html(self, current_locale: str = "en") -> str:
        """
        Generate HTML for language selector dropdown.

        Args:
            current_locale: Currently selected locale

        Returns:
            HTML string for language selector
        """
        options = []
        for locale in self.available_locales():
            meta = self.get_locale_metadata(locale)
            selected = "selected" if locale == current_locale else ""
            options.append(
                f'<option value="{meta["code"]}" {selected}>'
                f'{meta["flag"]} {meta["name"]}'
                f'</option>'
            )

        return f"""
<select id="lang-select" onchange="switchLanguage(this.value)" style="padding: 8px; border-radius: 4px; border: 1px solid #2a3555; background: #1a1f3a; color: #e0e0e0; cursor: pointer;">
    {chr(10).join(options)}
</select>
"""

    def __repr__(self) -> str:
        """String representation of TranslationManager."""
        return (
            f"TranslationManager(default_locale='{self.default_locale}', "
            f"locales={self.available_locales()})"
        )
