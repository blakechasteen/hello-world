# HoloLoom UX Audit

**Date**: 2026-03-18
**Scope**: React frontend (hololoom-ui/), Python API surface, CLI, documentation

---

## Executive Summary

HoloLoom's UX is strong in its design system fundamentals (3 themes, 12 composable components, semantic color tokens) and API ergonomics (5-method Lite API, graceful degradation). The main gaps are **accessibility on the canvas-based MemoryGraph**, **keyboard-only navigation holes**, and **silent fallback behavior** that leaves users unaware of degraded state.

---

## Findings

### 1. CRITICAL: Canvas MemoryGraph Not Accessible

**File**: `hololoom-ui/apps/web/src/components/memory/MemoryGraph.tsx`

The knowledge graph is rendered entirely on a `<canvas>` element with no accessibility support:
- No `aria-label` on the canvas
- No keyboard navigation (can't tab to nodes, use arrow keys)
- No screen reader description of graph structure
- Hover tooltips only visible with mouse

**Fix applied**: Added `role="img"`, `aria-label`, `tabIndex`, and keyboard event handlers for arrow key navigation and Enter to select.

**Remaining work** (future): Implement a visually-hidden node list (`role="listbox"`) as a full screen reader alternative to the canvas.

---

### 2. HIGH: Tooltip Only Shows on Hover

**File**: `hololoom-ui/packages/design-system/src/components/Tooltip.tsx`

Tooltips use `group-hover:` CSS only. Keyboard-only users never see tooltip content.

**Fix applied**: Added `group-focus-within:opacity-100 group-focus-within:visible` so tooltips appear when the trigger element receives keyboard focus.

---

### 3. HIGH: Icon-Only Buttons Missing Accessible Labels

**File**: `hololoom-ui/apps/web/src/components/memory/MemoryGraph.tsx:291-296`

Zoom in/out buttons have no text or `aria-label`. Screen readers announce them as empty buttons.

**Fix applied**: Added `aria-label` to zoom in, zoom out, and reset view buttons.

---

### 4. HIGH: Chat Input Missing Form Labels

**File**: `hololoom-ui/apps/web/src/components/ChatInterface.tsx:258-271`

The main chat input has `placeholder` text but no associated `<label>` element. Placeholder text disappears on focus and isn't announced reliably by screen readers.

**Fix applied**: Added `aria-label` to the input. Added `role="log"` and `aria-live="polite"` to the messages area. Added `aria-pressed` to reasoning mode selector buttons.

---

### 5. HIGH: No React Error Boundary

No error boundary exists. A rendering crash in any component takes down the entire page with a white screen.

**Fix applied**: Added `ErrorBoundary` component to the design system with a user-friendly fallback UI and retry button.

---

### 6. MEDIUM: Python __init__.py Error Messages Lack Guidance

**File**: `hololoom/__init__.py:235`

When users try to import a non-existent attribute (e.g., typo), they get a generic `AttributeError` with no suggestion of correct names.

**Fix applied**: Added fuzzy matching with `difflib.get_close_matches()` to suggest similar valid attribute names.

---

### 7. MEDIUM: Silent Fallback to In-Memory Backend

**File**: `hololoom/lite/core.py:178-182`

When Docker/persistent backends are unavailable, the system silently falls back to in-memory storage. Users may not realize their data isn't being persisted.

**Recommendation**: Surface a non-blocking banner or status indicator in the UI when running in degraded mode.

---

### 8. MEDIUM: Error Display Uses "Dismiss" That Clears Chat

**File**: `hololoom-ui/apps/web/src/components/ChatInterface.tsx:198`

The error banner's "Dismiss" button calls `clearConversation()`, which destroys the entire chat history. This is destructive and unexpected—users likely want to dismiss just the error.

**Recommendation**: Add a separate `clearError()` method that only clears the error state without destroying messages.

---

### 9. LOW: Reasoning Mode Selector Missing Keyboard Support

**File**: `hololoom-ui/apps/web/src/components/ChatInterface.tsx:222-242`

The reasoning mode dropdown has no `aria-expanded`, no `Escape` to close, and no `aria-pressed` state on selected mode.

**Fix applied**: Added `aria-expanded` to the toggle button and `aria-pressed` to mode options.

---

### 10. LOW: Loading States Use Generic "Loading" Labels

**File**: `hololoom-ui/packages/design-system/src/components/Loading.tsx`

`LoadingSpinner` and `LoadingDots` already have `role="status"` and `aria-label="Loading"`. The `WeavingIndicator` correctly uses `role="progressbar"` with `aria-valuenow`. No changes needed — this was well implemented.

---

## What's Working Well

| Area | Details |
|------|---------|
| **Design System** | 12 composable components with consistent API (size, variant, className) |
| **3 Themes** | Modern, Tufte, Terminal — persisted in localStorage with system preference detection |
| **Color Tokens** | Semantic tokens (confidence-high, safety-danger) decouple from raw hex values |
| **API Client** | Retry with exponential backoff, rate limit awareness, WebSocket auto-reconnect |
| **Loading States** | WeavingIndicator shows 9-stage progress, not just a spinner |
| **Focus Rings** | `focus-visible:ring-2` on all interactive elements |
| **ConfidenceIndicator** | Uses `role="meter"` with proper ARIA attributes |
| **Lite API** | 5 methods, clear fallback behavior, factory presets |
| **CLI** | Color-coded output, helpful error messages with install instructions |
| **Documentation** | Progressive depth (quickstart → guides → expert), vocabulary mapping table |
| **Graceful Degradation** | Optional deps warn but never crash |
| **Reduced Motion** | `prefers-reduced-motion` respected in globals.css |

---

## Changes Made in This Audit

| File | Change |
|------|--------|
| `hololoom-ui/apps/web/src/components/memory/MemoryGraph.tsx` | Canvas a11y: `aria-label`, `tabIndex`, keyboard nav, icon button labels |
| `hololoom-ui/packages/design-system/src/components/Tooltip.tsx` | Focus-visible tooltip display |
| `hololoom-ui/apps/web/src/components/ChatInterface.tsx` | Form labels, `aria-live` messages area, `aria-pressed`/`aria-expanded` |
| `hololoom-ui/packages/design-system/src/components/ErrorBoundary.tsx` | New error boundary component |
| `hololoom-ui/packages/design-system/src/components/index.ts` | Export ErrorBoundary |
| `hololoom/__init__.py` | "Did you mean?" suggestions on AttributeError |

---

## Recommended Future Work

1. **Visually-hidden node list** for MemoryGraph as full screen reader alternative
2. **Separate `clearError()` from `clearConversation()`** in ChatInterface
3. **Degraded-mode banner** when running without persistent backends
4. **Skeleton loading placeholders** instead of spinners for initial page loads
5. **Offline detection** with fallback UI when API is unreachable
6. **WCAG audit** with automated tooling (axe-core) in CI pipeline
7. **Mobile graph interaction** — pinch-to-zoom, touch drag on MemoryGraph canvas
