# Skill: Playwright

## Metadata

- **Name**: `playwright`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `web`
- **Tags**: `browser, automation, testing, scraping, e2e, screenshots`

## Description

**Short Description**:
Browser automation and end-to-end testing with Playwright across Chrome, Firefox, and WebKit.

**Detailed Description**:
The Playwright skill provides comprehensive browser automation capabilities for testing, web scraping, screenshots, PDF generation, and end-to-end workflows. Supports all major browsers (Chrome, Firefox, WebKit/Safari) in both headful and headless modes. Features include device emulation (mobile/tablet), network interception, geolocation mocking, and screenshot/PDF generation. Ideal for automated testing, visual regression testing, data extraction, and workflow automation.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [x] File system access (write)
- [x] Code execution (bash)
- [x] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `playwright` Python library (pip install playwright)
- Browser binaries via `playwright install` (chromium, firefox, webkit)
- Optional: `playwright-stealth` for anti-detection

**HoloLoom Integration**: Integrates with testing pipelines, data extraction workflows, visual regression testing, and monitoring systems.

## Input Schema

```json
{
  "operation": "string - navigate|screenshot|pdf|scrape|fill_form|click|wait_for",
  "parameters": {
    "url": "string (required for navigate, screenshot, pdf, scrape) - Target URL",
    "browser": "string (optional) - Browser type: chromium|firefox|webkit (default: chromium)",
    "headless": "boolean (optional) - Headless mode (default: true)",
    "selector": "string (required for click, wait_for) - CSS selector",
    "output_path": "string (optional for screenshot, pdf) - Output file path",
    "format": "string (optional for screenshot) - png|jpeg (default: png)",
    "full_page": "boolean (optional for screenshot) - Capture full page (default: false)",
    "fields": "object (required for fill_form) - Form field values {selector: value}",
    "submit": "boolean (optional for fill_form) - Submit form (default: true)",
    "wait_timeout": "number (optional) - Wait timeout in milliseconds (default: 30000)",
    "device": "string (optional) - Device emulation: iPhone 12|Pixel 5|iPad Pro, etc.",
    "viewport": "object (optional) - Custom viewport {width: number, height: number}"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Operation-specific result",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "url": "string - Target URL (if applicable)",
    "browser": "string - Browser used",
    "headless": "boolean - Headless mode",
    "title": "string - Page title (for navigate)",
    "path": "string - Output file path (for screenshot, pdf)",
    "format": "string - Screenshot format (png, jpeg)",
    "size_kb": "number - File size in KB (for screenshot, pdf)",
    "pages": "number - Number of pages (for pdf)",
    "text": "string - Scraped text content (for scrape)",
    "links": "number - Number of links found (for scrape)",
    "images": "number - Number of images found (for scrape)",
    "fields_filled": "number - Number of form fields filled (for fill_form)",
    "submitted": "boolean - Form submitted (for fill_form)",
    "clicked": "boolean - Element clicked (for click)",
    "found": "boolean - Element found (for wait_for)"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Navigate and Screenshot

**Input**:
```json
{
  "operation": "screenshot",
  "parameters": {
    "url": "https://example.com",
    "browser": "chromium",
    "headless": true,
    "output_path": "screenshots/homepage.png",
    "format": "png",
    "full_page": true
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "screenshot",
    "url": "https://example.com",
    "browser": "chromium",
    "path": "screenshots/homepage.png",
    "format": "png",
    "size_kb": 245,
    "width": 1920,
    "height": 4320,
    "full_page": true
  },
  "message": "Screenshot captured: screenshots/homepage.png",
  "execution_time_ms": 2150
}
```

**Explanation**: Navigates to example.com and captures a full-page screenshot in PNG format. Useful for visual regression testing and documentation.

### Example 2: Generate PDF Report

**Input**:
```json
{
  "operation": "pdf",
  "parameters": {
    "url": "https://docs.example.com/report",
    "browser": "chromium",
    "headless": true,
    "output_path": "reports/quarterly_report.pdf",
    "format": "A4",
    "print_background": true
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "pdf",
    "url": "https://docs.example.com/report",
    "path": "reports/quarterly_report.pdf",
    "pages": 12,
    "size_kb": 890,
    "format": "A4"
  },
  "message": "PDF generated: reports/quarterly_report.pdf (12 pages)",
  "execution_time_ms": 5420
}
```

**Explanation**: Converts a web page to a PDF document with proper formatting. Ideal for generating reports from web dashboards.

### Example 3: Web Scraping

**Input**:
```json
{
  "operation": "scrape",
  "parameters": {
    "url": "https://news.example.com",
    "browser": "firefox",
    "headless": true,
    "selectors": {
      "title": "h1.article-title",
      "content": "div.article-body",
      "author": "span.author-name",
      "date": "time.publish-date"
    }
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "scrape",
    "url": "https://news.example.com",
    "data": {
      "title": "Breaking News Article",
      "content": "Full article text content...",
      "author": "John Doe",
      "date": "2025-11-24"
    },
    "text_length": 2450,
    "links": 18,
    "images": 5
  },
  "message": "Scraped 4 fields from https://news.example.com",
  "execution_time_ms": 1890
}
```

**Explanation**: Extracts structured data from a news article using CSS selectors. Returns cleaned text, metadata, and statistics about page content.

### Example 4: Automated Form Submission

**Input**:
```json
{
  "operation": "fill_form",
  "parameters": {
    "url": "https://app.example.com/signup",
    "browser": "chromium",
    "headless": false,
    "fields": {
      "input[name='email']": "user@example.com",
      "input[name='password']": "SecurePass123!",
      "input[name='name']": "Test User",
      "select[name='country']": "US"
    },
    "submit": true,
    "submit_selector": "button[type='submit']"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "fill_form",
    "url": "https://app.example.com/signup",
    "fields_filled": 4,
    "submitted": true,
    "redirect_url": "https://app.example.com/welcome"
  },
  "message": "Form filled and submitted successfully (4 fields)",
  "execution_time_ms": 3200
}
```

**Explanation**: Automates form filling and submission for signup workflows. Supports text inputs, dropdowns, checkboxes, and custom submit buttons.

### Example 5: Mobile Device Emulation

**Input**:
```json
{
  "operation": "navigate",
  "parameters": {
    "url": "https://mobile.example.com",
    "browser": "webkit",
    "device": "iPhone 12",
    "headless": true,
    "screenshot": "mobile_view.png"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "navigate",
    "url": "https://mobile.example.com",
    "browser": "webkit",
    "device": "iPhone 12",
    "viewport": {"width": 390, "height": 844},
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)...",
    "title": "Mobile Site",
    "screenshot": "mobile_view.png"
  },
  "message": "Navigated with iPhone 12 emulation",
  "execution_time_ms": 1650
}
```

**Explanation**: Emulates mobile devices for responsive testing. Automatically sets viewport, user agent, touch events, and device pixel ratio.

## Testing Checklist

- [x] **Functionality**: All 7 operations execute correctly
- [x] **Error Handling**: Graceful handling of network errors, timeouts, missing elements
- [x] **Security**: No command injection, safe URL handling
- [x] **Performance**: Operations complete within expected time (<30s)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Playwright installation documented
- [x] **Edge Cases**: Handles slow networks, missing selectors, popup blockers
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom testing and scraping pipelines

## Security Considerations

**Potential Risks**:
- **URL Injection**: Malicious URLs could exploit browser vulnerabilities -> Validate and sanitize URLs
- **Cross-Site Scripting (XSS)**: Scraped content may contain malicious scripts -> Sanitize extracted data
- **Resource Exhaustion**: Large page rendering -> Implement timeouts and memory limits

**Data Privacy**:
- [x] Does not log credentials or sensitive form data
- [x] Screenshots/PDFs stored securely with proper permissions
- [x] Does not send browser data to external servers

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Browser runs in isolated context
- [x] Does not access files outside designated output directories

## Performance Characteristics

- **Expected Latency**: 1000-30000ms (1-30 seconds depending on page complexity)
- **Token Usage**: 200-2000 tokens per execution
- **Resource Requirements**: Playwright browser binaries, sufficient memory for rendering
- **Scalability**: Limited by concurrent browser instances (recommend <10 concurrent)

**Operation-Specific Latencies**:
- `navigate`: 1000-5000ms (depends on page load time)
- `screenshot`: 500-3000ms (depends on page size)
- `pdf`: 2000-10000ms (depends on page count)
- `scrape`: 1000-5000ms (depends on content extraction complexity)
- `fill_form`: 2000-8000ms (depends on form validation)
- `click`: 500-2000ms (depends on element interaction)
- `wait_for`: 100-30000ms (depends on timeout setting)

## License

MIT License

## Related Documentation

- **Playwright Docs**: [playwright.dev](https://playwright.dev)
- **Browser Compatibility**: [caniuse.com](https://caniuse.com)
- **Device Emulation**: [playwright.dev/docs/emulation](https://playwright.dev/docs/emulation)
- **HoloLoom Web Skills**: [../README.md](../README.md)
