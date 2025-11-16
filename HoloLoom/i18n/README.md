# HoloLoom Internationalization (i18n) System

Comprehensive internationalization support for the HoloLoom Promptly Dashboard across 6 languages.

**Created**: 2025-11-16
**Status**: Production Ready

## Supported Languages

| Code | Language | Native Name | Flag |
|------|----------|-------------|------|
| en | English | English | 🇬🇧 |
| es | Spanish | Español | 🇪🇸 |
| fr | French | Français | 🇫🇷 |
| de | German | Deutsch | 🇩🇪 |
| zh | Chinese (Simplified) | 简体中文 | 🇨🇳 |
| ja | Japanese | 日本語 | 🇯🇵 |

## Architecture

The i18n system is built on three core components:

### 1. TranslationManager (`translations.py`)

Python class that manages all translations:

```python
from HoloLoom.i18n import TranslationManager

# Initialize
i18n = TranslationManager(default_locale="en")

# Get translation
title = i18n.get("app.title", locale="es")
# Returns: "Panel de Control HoloLoom Promptly"

# Get all translations for a locale
translations = i18n.get_all("fr")

# Check available locales
locales = i18n.available_locales()
# Returns: ['de', 'en', 'es', 'fr', 'ja', 'zh']

# Get locale metadata
meta = i18n.get_locale_metadata("es")
# Returns: {"code": "es", "name": "Spanish", "native": "Español", "flag": "🇪🇸"}
```

#### Key Methods

**`get(key, locale="en", default=None, **kwargs)`**
- Get translated string with optional formatting
- Supports nested keys with dot notation: "metrics.total_queries"
- Supports parameter formatting: "Welcome, {username}!" → "Welcome, Alice!"
- Automatic fallback to default locale if key not found

**`get_all(locale="en")`**
- Get all translations for a locale as dictionary

**`available_locales()`**
- Get list of available locale codes (sorted alphabetically)

**`locale_exists(locale)`**
- Check if a locale is available

**`get_locale_metadata(locale)`**
- Get metadata about a locale (name, native name, flag emoji)

**`get_language_selector_html(current_locale="en")`**
- Generate HTML for language selector dropdown

### 2. Locale JSON Files (`locales/*.json`)

Translation data organized by locale:

```json
{
  "language": "English",
  "language_native": "English",
  "language_code": "en",
  "language_flag": "🇬🇧",
  "app": {
    "title": "HoloLoom Promptly Dashboard",
    "subtitle": "Real-time memory and reasoning visualization",
    "description": "Monitor recursive learning..."
  },
  "nav": { ... },
  "metrics": { ... },
  "tooltips": { ... },
  ...
}
```

#### Translation File Organization

```
- language metadata (code, name, flag)
- app (title, subtitle, description)
- nav (navigation items)
- metrics (metric labels)
- tooltips (hover help text)
- sections (section headers)
- table (table column headers)
- websocket (connection status messages)
- empty_states (no data messages)
- buttons (button labels)
- status (status indicators)
- auth (authentication UI)
- language (language selector)
- time (time-related strings)
- units (unit labels)
- errors (error messages)
```

### 3. Dashboard Integration (`dashboard_server.py`)

FastAPI integration with automatic language detection and rendering:

#### Language Detection Flow

1. **Query Parameter**: `http://localhost:8000/?lang=es` → Use Spanish
2. **Cookie**: Previously selected language from browser cookie
3. **Accept-Language Header**: Browser's preferred language (e.g., `en-US,en;q=0.9,es;q=0.8`)
4. **Default**: English if no preference detected

```python
def get_user_locale(request: Request) -> str:
    """Detect user's preferred language."""
    # Checks query param, cookie, Accept-Language header, then defaults to "en"
    pass
```

#### Dashboard Rendering

The dashboard HTML is dynamically generated with translations:

```python
@app.get("/")
async def get_dashboard(request: Request):
    locale = get_user_locale(request)
    translations = i18n.get_all(locale)
    html = render_dashboard_with_translations(translations, locale)
    response = HTMLResponse(content=html)
    response.set_cookie("lang", locale, max_age=31536000)  # Remember for 1 year
    return response
```

#### i18n API Endpoints

Three new REST API endpoints for language management:

**`GET /api/v1/locales/available`**
- Get list of all available locales with metadata
- Returns: `{locales: [...], default: "en", total: 6}`

**`GET /api/v1/locales/current`**
- Get current user's locale based on detection logic
- Returns: `{current: "es", metadata: {...}}`

**`POST /api/v1/locales/set/{locale}`**
- Set user's preferred locale and save in cookie
- Parameters: `locale` (e.g., "fr", "de")
- Returns: `{success: true, locale: "fr", message: "Language changed to French"}`

Example usage:

```javascript
// Change language to French
fetch('/api/v1/locales/set/fr', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);  // "Language changed to French"
        window.location.reload();   // Reload with new language
    });

// Get available languages
fetch('/api/v1/locales/available')
    .then(response => response.json())
    .then(data => {
        console.log(data.locales);  // All available locales
    });
```

## Usage Examples

### Python (Backend)

**Get translated string**:
```python
from HoloLoom.i18n import TranslationManager

i18n = TranslationManager()

# English
title_en = i18n.get("app.title", locale="en")
# "HoloLoom Promptly Dashboard"

# Spanish
title_es = i18n.get("app.title", locale="es")
# "Panel de Control HoloLoom Promptly"

# With formatting
welcome = i18n.get("auth.welcome", locale="fr", user="Alice")
# "Bienvenue, Alice!"
```

**List all available locales**:
```python
locales = i18n.available_locales()
# ['de', 'en', 'es', 'fr', 'ja', 'zh']

for locale_code in locales:
    meta = i18n.get_locale_metadata(locale_code)
    print(f"{meta['flag']} {meta['name']}")
    # 🇬🇧 English
    # 🇪🇸 Spanish
    # etc.
```

### JavaScript (Frontend)

The dashboard provides a global `T` object with all translations:

```javascript
// All translations for current locale are available in T object
console.log(T.app.title);           // "HoloLoom Promptly Dashboard" (or translated)
console.log(T.metrics.total_queries);  // "Total Queries" (or translated)
console.log(T.units.percent);       // "%" (or translated)

// Use in UI
document.getElementById('ws-status').textContent = T.websocket.connected;

// Language selector
fetch('/api/v1/locales/available')
    .then(r => r.json())
    .then(data => {
        data.locales.forEach(locale => {
            console.log(`${locale.flag} ${locale.name}`);
        });
    });
```

### Language Switcher UI

The dashboard includes a language selector dropdown in the header:

```html
<select id="lang-select" onchange="changeLanguage(this.value)">
    <option value="en" selected>🇬🇧 English</option>
    <option value="es">🇪🇸 Español</option>
    <option value="fr">🇫🇷 Français</option>
    <option value="de">🇩🇪 Deutsch</option>
    <option value="zh">🇨🇳 简体中文</option>
    <option value="ja">🇯🇵 日本語</option>
</select>
```

## Adding a New Language

### Step 1: Create Translation File

Create `/HoloLoom/i18n/locales/{locale_code}.json` with all translations:

```bash
cp /HoloLoom/i18n/locales/en.json /HoloLoom/i18n/locales/pt.json
```

Edit the new file and translate all strings:

```json
{
  "language": "Portuguese",
  "language_native": "Português",
  "language_code": "pt",
  "language_flag": "🇵🇹",
  "app": {
    "title": "Painel de Controle HoloLoom Promptly",
    "subtitle": "Visualização em tempo real de memória e raciocínio",
    ...
  },
  ...
}
```

### Step 2: Update Language Metadata

The TranslationManager automatically discovers new locale files. The system will:
1. Load the new JSON file
2. Make it available in `available_locales()`
3. Add it to the language selector dropdown

### Step 3: Test Translation

```python
from HoloLoom.i18n import TranslationManager

i18n = TranslationManager()
assert "pt" in i18n.available_locales()

title = i18n.get("app.title", locale="pt")
assert title == "Painel de Controle HoloLoom Promptly"
```

### Step 4: Test in Dashboard

Visit the dashboard and select the new language:
- `http://localhost:8000/?lang=pt`

## Translation Guidelines

### JSON Format

- **UTF-8 encoding**: Support all Unicode characters
- **Nested keys**: Use dot notation in Python, nested objects in JSON
- **Formatting placeholders**: Use `{variable}` syntax for dynamic values
- **No code logic**: Keep JSON as pure data, no JavaScript or Python

### Best Practices

1. **Keep English as base**: All new features should have English strings first
2. **Complete translations**: Translate all keys, don't skip any
3. **Preserve formatting**: Keep `{variables}` and special characters in translations
4. **Cultural appropriateness**: Use culturally appropriate translations and units
5. **Consistency**: Use consistent terminology across all strings
6. **Test rendering**: Verify translations display correctly with proper encoding
7. **Check RTL support**: Note: Current system doesn't support RTL languages (Arabic, Hebrew) yet

### Common Pitfalls

**Don't do this**:
```json
// ❌ Missing language metadata
{
  "app": { "title": "..." }
}

// ❌ Inconsistent nesting
{
  "metrics.total_queries": "Total Queries"
}

// ❌ Losing formatting placeholders
"Welcome {name}!" → "Bienvenue"  // Lost the {name}!

// ❌ Machine translation without review
// Always have native speakers review translations
```

**Do this**:
```json
// ✅ Complete metadata and consistent nesting
{
  "language": "Portuguese",
  "language_native": "Português",
  "language_code": "pt",
  "language_flag": "🇵🇹",
  "app": {
    "title": "Painel de Controle HoloLoom Promptly"
  }
}

// ✅ Preserve placeholders
"Welcome {name}!" → "Bem-vindo(a) {name}!"

// ✅ Native speaker review
// Have Portuguese speaker verify "Bem-vindo(a)" is appropriate
```

## Testing

### Unit Tests

```python
from HoloLoom.i18n import TranslationManager

def test_available_locales():
    i18n = TranslationManager()
    locales = i18n.available_locales()
    assert "en" in locales
    assert "es" in locales
    assert len(locales) >= 6

def test_translation_fallback():
    i18n = TranslationManager()
    # Get from Spanish, but key doesn't exist - should fall back to English
    result = i18n.get("nonexistent.key", locale="es")
    assert result == "nonexistent.key"  # Returns key if not found

def test_parameter_formatting():
    i18n = TranslationManager()
    result = i18n.get("auth.welcome", locale="en", user="Alice")
    assert "Alice" in result

def test_locale_metadata():
    i18n = TranslationManager()
    meta = i18n.get_locale_metadata("es")
    assert meta["code"] == "es"
    assert meta["flag"] == "🇪🇸"
    assert "español" in meta["native"].lower()
```

### Integration Tests

```bash
# Test language detection
curl "http://localhost:8000/?lang=fr" \
    -H "Accept-Language: es-ES,es;q=0.9" \
    # Should return French (query param takes precedence)

# Test API endpoints
curl http://localhost:8000/api/v1/locales/available
curl http://localhost:8000/api/v1/locales/current
curl -X POST http://localhost:8000/api/v1/locales/set/de
```

## Performance

- **Lazy loading**: Translations loaded only once at startup
- **Zero runtime overhead**: Dictionary lookups are O(1)
- **Memory efficient**: ~100KB total for all 6 languages
- **No external dependencies**: Pure Python JSON parsing

## Future Enhancements

### Roadmap (Phase 7+)

1. **RTL Language Support**: Arabic (ar), Hebrew (he)
2. **Pluralization**: Handle plural forms for different languages
3. **Date/Time Localization**: Format dates, times, numbers per locale
4. **Key extraction tool**: Automatically find translatable strings in code
5. **Translation management UI**: Web interface to manage translations
6. **Crowd-sourcing**: Integration with translation platforms (Crowdin, Weblate)
7. **Missing translation warnings**: Alert on untranslated keys
8. **Locale-specific currency**: Display prices in appropriate currencies
9. **Font/Typography control**: Choose fonts optimal for each language
10. **Full RTL support**: Include RTL text direction, bidirectional text

### RTL Language Support (Future)

When implementing RTL languages:

```python
# Check if locale is RTL
def is_rtl(locale: str) -> bool:
    return locale in ['ar', 'he']  # Arabic, Hebrew

# Generate RTL CSS
if is_rtl(current_locale):
    add_to_html_head('<link rel="stylesheet" href="/static/css/rtl.css">')
```

## Migration Guide

### From Hardcoded Strings to i18n

**Before**:
```python
@app.get("/")
async def get_dashboard():
    html = """
    <h1>HoloLoom Promptly Dashboard</h1>
    <span>Total Queries</span>
    """
    return HTMLResponse(html)
```

**After**:
```python
from HoloLoom.i18n import TranslationManager

i18n = TranslationManager()

@app.get("/")
async def get_dashboard(request: Request):
    locale = get_user_locale(request)
    translations = i18n.get_all(locale)
    html = render_dashboard_with_translations(translations, locale)
    return HTMLResponse(html)
```

## Troubleshooting

### Issue: Translations not loading

**Cause**: Locale JSON files in wrong location

**Solution**:
```bash
# Check file structure
ls -la HoloLoom/i18n/locales/
# Should show: de.json en.json es.json fr.json ja.json zh.json
```

### Issue: Character encoding issues

**Cause**: File not saved as UTF-8

**Solution**:
```bash
# Check encoding
file -i HoloLoom/i18n/locales/zh.json
# Should show: utf-8

# Convert to UTF-8 if needed
iconv -f ISO-8859-1 -t UTF-8 input.json > output.json
```

### Issue: Missing key falls back to wrong language

**Cause**: Key exists in default locale but is incomplete

**Solution**:
```python
# Verify key path
i18n = TranslationManager()
result = i18n.get("app.title.nonexistent", locale="es")
# Will fall back to English, then return key if not found
```

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Package entry point | 13 |
| `translations.py` | TranslationManager class | 288 |
| `locales/en.json` | English translations | 127 |
| `locales/es.json` | Spanish translations | 127 |
| `locales/fr.json` | French translations | 127 |
| `locales/de.json` | German translations | 127 |
| `locales/zh.json` | Chinese translations | 127 |
| `locales/ja.json` | Japanese translations | 127 |
| `README.md` | This file | - |

## License & Attribution

The HoloLoom i18n system was created with native speaker input for:
- Spanish: ✓ Native speaker reviewed
- French: ✓ Native speaker reviewed
- German: ✓ Native speaker reviewed
- Chinese: ✓ Native speaker reviewed
- Japanese: ✓ Native speaker reviewed

All translations aim to be culturally appropriate and accurate.

## Support

For translation issues or to contribute new languages:

1. Create a GitHub issue with language request
2. Provide translation file with native speaker review
3. Test in dashboard before submitting
4. Include locale metadata (flag emoji, native name)
