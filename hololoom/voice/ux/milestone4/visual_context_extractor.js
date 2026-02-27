/**
 * Visual Context Extractor - Screen Content Analysis
 * Milestone 4: Visual + voice fusion
 * Date: November 2025
 *
 * Features:
 * - Screen content extraction (UI elements, text, images)
 * - Element location and bounding box detection
 * - Visual attention tracking (gaze, mouse position)
 * - Semantic element classification
 * - Spatial relationship analysis
 * - Deictic reference target identification
 * - Screen region management (active window, viewport)
 */

/**
 * UI Element Types
 */
const ElementType = {
    BUTTON: 'button',
    INPUT: 'input',
    TEXT: 'text',
    IMAGE: 'image',
    LINK: 'link',
    HEADING: 'heading',
    LIST: 'list',
    CONTAINER: 'container',
    ICON: 'icon',
    MENU: 'menu',
    MODAL: 'modal',
    THREAD: 'thread',      // Loom-specific
    MESSAGE: 'message',    // Loom-specific
    ATTACHMENT: 'attachment'
};

/**
 * Spatial Relationships
 */
const SpatialRelation = {
    ABOVE: 'above',
    BELOW: 'below',
    LEFT: 'left',
    RIGHT: 'right',
    INSIDE: 'inside',
    NEAR: 'near',
    FAR: 'far'
};

/**
 * Visual Element
 */
class VisualElement {
    constructor(data) {
        this.id = data.id || `element_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.type = data.type;
        this.label = data.label || '';
        this.text = data.text || '';

        // Location (bounding box)
        this.x = data.x || 0;
        this.y = data.y || 0;
        this.width = data.width || 0;
        this.height = data.height || 0;

        // Visual properties
        this.visible = data.visible !== false;
        this.interactive = data.interactive !== false;
        this.focused = data.focused || false;

        // Semantic properties
        this.importance = data.importance || 0.5; // 0.0-1.0
        this.keywords = data.keywords || [];

        // Metadata
        this.metadata = data.metadata || {};
        this.timestamp = data.timestamp || new Date();
    }

    /**
     * Get center point
     */
    getCenter() {
        return {
            x: this.x + this.width / 2,
            y: this.y + this.height / 2
        };
    }

    /**
     * Check if point is inside element
     */
    containsPoint(x, y) {
        return x >= this.x &&
               x <= this.x + this.width &&
               y >= this.y &&
               y <= this.y + this.height;
    }

    /**
     * Calculate distance to point
     */
    distanceToPoint(x, y) {
        const center = this.getCenter();
        const dx = center.x - x;
        const dy = center.y - y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Calculate spatial relationship to another element
     */
    relationTo(other) {
        const thisCenter = this.getCenter();
        const otherCenter = other.getCenter();

        // Check containment
        if (this.x >= other.x &&
            this.x + this.width <= other.x + other.width &&
            this.y >= other.y &&
            this.y + this.height <= other.y + other.height) {
            return SpatialRelation.INSIDE;
        }

        // Check directional relationships
        const dx = thisCenter.x - otherCenter.x;
        const dy = thisCenter.y - otherCenter.y;

        if (Math.abs(dx) > Math.abs(dy)) {
            return dx > 0 ? SpatialRelation.RIGHT : SpatialRelation.LEFT;
        } else {
            return dy > 0 ? SpatialRelation.BELOW : SpatialRelation.ABOVE;
        }
    }
}

/**
 * Visual Context (screen state snapshot)
 */
class VisualContext {
    constructor(data) {
        this.id = data.id || `context_${Date.now()}`;
        this.elements = data.elements || [];
        this.timestamp = data.timestamp || new Date();

        // Active regions
        this.activeWindow = data.activeWindow || null;
        this.viewport = data.viewport || { x: 0, y: 0, width: 1920, height: 1080 };

        // Attention tracking
        this.gazePoint = data.gazePoint || null; // {x, y}
        this.mousePosition = data.mousePosition || null; // {x, y}
        this.focusedElement = data.focusedElement || null;

        // Metadata
        this.appContext = data.appContext || 'unknown'; // 'loom', 'browser', 'editor'
    }

    /**
     * Find elements by type
     */
    getElementsByType(type) {
        return this.elements.filter(el => el.type === type);
    }

    /**
     * Find elements containing text
     */
    getElementsByText(text, caseSensitive = false) {
        const query = caseSensitive ? text : text.toLowerCase();
        return this.elements.filter(el => {
            const content = caseSensitive ? el.text : el.text.toLowerCase();
            return content.includes(query);
        });
    }

    /**
     * Find element at point (e.g., gaze position)
     */
    getElementAtPoint(x, y) {
        // Return topmost element at point
        const candidates = this.elements.filter(el => el.containsPoint(x, y));
        if (candidates.length === 0) return null;

        // Sort by importance and return best match
        candidates.sort((a, b) => b.importance - a.importance);
        return candidates[0];
    }

    /**
     * Find elements near point
     */
    getElementsNearPoint(x, y, maxDistance = 100) {
        return this.elements
            .map(el => ({ element: el, distance: el.distanceToPoint(x, y) }))
            .filter(item => item.distance <= maxDistance)
            .sort((a, b) => a.distance - b.distance)
            .map(item => item.element);
    }

    /**
     * Get currently focused element (attention-weighted)
     */
    getMostLikelyTarget() {
        // Priority: focused > gaze > mouse > recent interaction
        if (this.focusedElement) {
            return this.focusedElement;
        }

        if (this.gazePoint) {
            const gazeElement = this.getElementAtPoint(this.gazePoint.x, this.gazePoint.y);
            if (gazeElement) return gazeElement;
        }

        if (this.mousePosition) {
            const mouseElement = this.getElementAtPoint(this.mousePosition.x, this.mousePosition.y);
            if (mouseElement) return mouseElement;
        }

        return null;
    }
}

/**
 * Visual Context Extractor
 * Extracts and analyzes screen content
 */
class VisualContextExtractor {
    constructor(config = {}) {
        this.config = {
            enabled: config.enabled !== false,

            // Extraction settings
            updateIntervalMs: config.updateIntervalMs || 500, // 2 FPS default
            enableGazeTracking: config.enableGazeTracking || false,
            enableMouseTracking: config.enableMouseTracking !== false,

            // Element detection
            minElementSize: config.minElementSize || 10, // pixels
            importanceThreshold: config.importanceThreshold || 0.3,

            // Spatial analysis
            nearDistanceThreshold: config.nearDistanceThreshold || 100, // pixels

            // Performance
            maxElementsPerFrame: config.maxElementsPerFrame || 100,

            ...config
        };

        // Current context
        this.currentContext = null;
        this.contextHistory = [];
        this.maxHistorySize = 10;

        // Extraction state
        this.isExtracting = false;
        this.extractionTimer = null;

        // Attention tracking
        this.gazePosition = null;
        this.mousePosition = null;

        // Callbacks
        this.onContextUpdateCallback = null;
        this.onElementFocusCallback = null;

        // Metrics
        this.metrics = {
            totalExtractions: 0,
            avgElementsPerFrame: 0,
            avgExtractionTimeMs: 0,
            lastExtractionTime: null
        };

        console.log('[VisualContextExtractor] Initialized');
    }

    /**
     * Start continuous extraction
     */
    async startExtraction() {
        if (!this.config.enabled || this.isExtracting) {
            return false;
        }

        this.isExtracting = true;

        // Setup mouse tracking
        if (this.config.enableMouseTracking) {
            this._startMouseTracking();
        }

        // Setup gaze tracking (requires eye-tracking hardware)
        if (this.config.enableGazeTracking) {
            await this._startGazeTracking();
        }

        // Start extraction loop
        this._startExtractionLoop();

        console.log('[VisualContextExtractor] Extraction started');
        return true;
    }

    /**
     * Stop extraction
     */
    stopExtraction() {
        if (!this.isExtracting) {
            return false;
        }

        this.isExtracting = false;

        if (this.extractionTimer) {
            clearInterval(this.extractionTimer);
            this.extractionTimer = null;
        }

        this._stopMouseTracking();
        this._stopGazeTracking();

        console.log('[VisualContextExtractor] Extraction stopped');
        return true;
    }

    /**
     * Extract current visual context (one-time)
     */
    async extractContext() {
        const startTime = Date.now();

        // Extract elements from DOM
        const elements = await this._extractElementsFromDOM();

        // Create context
        const context = new VisualContext({
            elements: elements,
            activeWindow: this._getActiveWindow(),
            viewport: this._getViewport(),
            gazePoint: this.gazePosition,
            mousePosition: this.mousePosition,
            focusedElement: this._getFocusedElement(elements),
            appContext: this._detectAppContext()
        });

        // Update current context
        this.currentContext = context;

        // Add to history
        this.contextHistory.push(context);
        if (this.contextHistory.length > this.maxHistorySize) {
            this.contextHistory.shift();
        }

        // Update metrics
        const extractionTime = Date.now() - startTime;
        this.metrics.totalExtractions++;
        this.metrics.avgElementsPerFrame =
            (this.metrics.avgElementsPerFrame * (this.metrics.totalExtractions - 1) + elements.length) /
            this.metrics.totalExtractions;
        this.metrics.avgExtractionTimeMs =
            (this.metrics.avgExtractionTimeMs * (this.metrics.totalExtractions - 1) + extractionTime) /
            this.metrics.totalExtractions;
        this.metrics.lastExtractionTime = new Date();

        // Emit event
        if (this.onContextUpdateCallback) {
            this.onContextUpdateCallback({
                type: 'context_update',
                context: context,
                timestamp: new Date()
            });
        }

        console.log('[VisualContextExtractor] Context extracted:', elements.length, 'elements');

        return context;
    }

    /**
     * Start extraction loop
     */
    _startExtractionLoop() {
        if (!this.isExtracting) {
            return;
        }

        const extract = async () => {
            if (!this.isExtracting) {
                return; // Stop loop
            }

            await this.extractContext();
        };

        // Initial extraction
        extract();

        // Periodic extraction
        this.extractionTimer = setInterval(extract, this.config.updateIntervalMs);
    }

    /**
     * Extract elements from DOM
     */
    async _extractElementsFromDOM() {
        const elements = [];

        // Extract interactive elements
        const selectors = [
            'button',
            'a',
            'input',
            'textarea',
            'select',
            '[role="button"]',
            '[role="link"]',
            '[onclick]',
            '.thread-item',    // Loom-specific
            '.message-item',   // Loom-specific
            'h1, h2, h3, h4, h5, h6',
            'img',
            'ul, ol'
        ];

        const domElements = document.querySelectorAll(selectors.join(', '));

        for (const el of domElements) {
            // Skip hidden or tiny elements
            const rect = el.getBoundingClientRect();
            if (rect.width < this.config.minElementSize ||
                rect.height < this.config.minElementSize) {
                continue;
            }

            const visualElement = this._createVisualElement(el, rect);

            // Filter by importance
            if (visualElement.importance >= this.config.importanceThreshold) {
                elements.push(visualElement);
            }

            // Limit elements per frame
            if (elements.length >= this.config.maxElementsPerFrame) {
                break;
            }
        }

        return elements;
    }

    /**
     * Create VisualElement from DOM element
     */
    _createVisualElement(domElement, rect) {
        const type = this._classifyElement(domElement);
        const label = this._extractLabel(domElement);
        const text = domElement.textContent?.trim() || '';
        const keywords = this._extractKeywords(text, label);
        const importance = this._calculateImportance(domElement, type);

        return new VisualElement({
            type: type,
            label: label,
            text: text.substring(0, 200), // Limit text length
            x: rect.left,
            y: rect.top,
            width: rect.width,
            height: rect.height,
            visible: this._isVisible(domElement),
            interactive: this._isInteractive(domElement),
            focused: document.activeElement === domElement,
            importance: importance,
            keywords: keywords,
            metadata: {
                tagName: domElement.tagName,
                className: domElement.className,
                id: domElement.id
            }
        });
    }

    /**
     * Classify element type
     */
    _classifyElement(el) {
        const tag = el.tagName.toLowerCase();

        if (tag === 'button' || el.getAttribute('role') === 'button') return ElementType.BUTTON;
        if (tag === 'a') return ElementType.LINK;
        if (tag === 'input' || tag === 'textarea' || tag === 'select') return ElementType.INPUT;
        if (tag === 'img') return ElementType.IMAGE;
        if (tag.match(/^h[1-6]$/)) return ElementType.HEADING;
        if (tag === 'ul' || tag === 'ol') return ElementType.LIST;

        // Loom-specific
        if (el.className?.includes('thread')) return ElementType.THREAD;
        if (el.className?.includes('message')) return ElementType.MESSAGE;

        return ElementType.CONTAINER;
    }

    /**
     * Extract human-readable label
     */
    _extractLabel(el) {
        // Priority: aria-label > title > alt > text content
        return el.getAttribute('aria-label') ||
               el.getAttribute('title') ||
               el.getAttribute('alt') ||
               el.textContent?.trim().substring(0, 50) ||
               '';
    }

    /**
     * Extract keywords from text
     */
    _extractKeywords(text, label) {
        const combined = `${text} ${label}`.toLowerCase();
        const words = combined.split(/\s+/).filter(w => w.length > 3);
        return [...new Set(words)].slice(0, 5); // Top 5 unique keywords
    }

    /**
     * Calculate element importance (0.0-1.0)
     */
    _calculateImportance(el, type) {
        let importance = 0.5;

        // Type-based importance
        const typeImportance = {
            [ElementType.BUTTON]: 0.8,
            [ElementType.LINK]: 0.7,
            [ElementType.INPUT]: 0.9,
            [ElementType.HEADING]: 0.7,
            [ElementType.THREAD]: 0.8,
            [ElementType.MESSAGE]: 0.6
        };
        importance = typeImportance[type] || 0.5;

        // Focused element gets boost
        if (document.activeElement === el) {
            importance = Math.min(1.0, importance + 0.2);
        }

        // Prominent positioning (center of screen)
        const rect = el.getBoundingClientRect();
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        const distanceFromCenter = Math.sqrt(
            Math.pow(rect.left + rect.width / 2 - centerX, 2) +
            Math.pow(rect.top + rect.height / 2 - centerY, 2)
        );
        const maxDistance = Math.sqrt(centerX * centerX + centerY * centerY);
        const centralityBoost = 0.1 * (1 - distanceFromCenter / maxDistance);
        importance = Math.min(1.0, importance + centralityBoost);

        return importance;
    }

    /**
     * Check if element is visible
     */
    _isVisible(el) {
        const style = window.getComputedStyle(el);
        return style.display !== 'none' &&
               style.visibility !== 'hidden' &&
               style.opacity !== '0';
    }

    /**
     * Check if element is interactive
     */
    _isInteractive(el) {
        const tag = el.tagName.toLowerCase();
        const interactiveTags = ['button', 'a', 'input', 'textarea', 'select'];
        return interactiveTags.includes(tag) ||
               el.hasAttribute('onclick') ||
               el.getAttribute('role') === 'button';
    }

    /**
     * Get active window
     */
    _getActiveWindow() {
        return {
            title: document.title,
            url: window.location.href
        };
    }

    /**
     * Get viewport
     */
    _getViewport() {
        return {
            x: window.scrollX,
            y: window.scrollY,
            width: window.innerWidth,
            height: window.innerHeight
        };
    }

    /**
     * Get focused element from list
     */
    _getFocusedElement(elements) {
        return elements.find(el => el.focused) || null;
    }

    /**
     * Detect application context
     */
    _detectAppContext() {
        if (window.location.href.includes('loom.com')) return 'loom';
        if (window.location.protocol === 'file:') return 'local';
        return 'browser';
    }

    /**
     * Start mouse tracking
     */
    _startMouseTracking() {
        this._mouseMoveHandler = (event) => {
            this.mousePosition = { x: event.clientX, y: event.clientY };
        };
        document.addEventListener('mousemove', this._mouseMoveHandler);
    }

    /**
     * Stop mouse tracking
     */
    _stopMouseTracking() {
        if (this._mouseMoveHandler) {
            document.removeEventListener('mousemove', this._mouseMoveHandler);
            this._mouseMoveHandler = null;
        }
    }

    /**
     * Start gaze tracking (requires eye-tracking API)
     */
    async _startGazeTracking() {
        // Placeholder - requires WebGazer.js or similar
        console.log('[VisualContextExtractor] Gaze tracking not yet implemented');
        return false;
    }

    /**
     * Stop gaze tracking
     */
    _stopGazeTracking() {
        // Placeholder
    }

    /**
     * Resolve deictic reference ("that", "this") to element
     */
    resolveDeictic(deicticWord, voiceContext = null) {
        if (!this.currentContext) {
            return null;
        }

        // Use attention to find most likely target
        const target = this.currentContext.getMostLikelyTarget();

        if (!target) {
            return null;
        }

        return {
            resolved: true,
            element: target,
            deicticWord: deicticWord,
            confidence: target.importance,
            resolutionMethod: this._getResolutionMethod()
        };
    }

    /**
     * Get resolution method (how target was identified)
     */
    _getResolutionMethod() {
        if (this.currentContext?.focusedElement) return 'focus';
        if (this.currentContext?.gazePoint) return 'gaze';
        if (this.currentContext?.mousePosition) return 'mouse';
        return 'heuristic';
    }

    /**
     * Get current context
     */
    getCurrentContext() {
        return this.currentContext;
    }

    /**
     * Get context history
     */
    getContextHistory(limit = 10) {
        return this.contextHistory.slice(-limit);
    }

    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isExtracting: this.isExtracting,
            currentElementCount: this.currentContext?.elements.length || 0
        };
    }

    /**
     * Cleanup
     */
    async destroy() {
        this.stopExtraction();
        this.currentContext = null;
        this.contextHistory = [];
        console.log('[VisualContextExtractor] Destroyed');
    }

    // Public API for callbacks

    onContextUpdate(callback) {
        this.onContextUpdateCallback = callback;
    }

    onElementFocus(callback) {
        this.onElementFocusCallback = callback;
    }
}


/**
 * Integration Example
 * Shows how to use VisualContextExtractor with MultimodalRouter
 */
class MultimodalVoiceSystem {
    constructor(multimodalRouter, visualExtractor) {
        this.multimodalRouter = multimodalRouter;
        this.visualExtractor = visualExtractor;

        this._setupIntegration();
    }

    _setupIntegration() {
        // Listen for visual context updates
        this.visualExtractor.onContextUpdate((event) => {
            console.log('[MultimodalVoiceSystem] Visual context updated:',
                event.context.elements.length, 'elements');
        });

        // Register visual modality handler
        this.multimodalRouter.registerHandler(
            'visual',
            {
                interpret: async (content) => {
                    // Visual content is element ID or description
                    const element = this.visualExtractor.getCurrentContext()
                        ?.elements.find(el => el.id === content || el.label === content);

                    return {
                        action: 'select',
                        target: element,
                        confidence: element ? element.importance : 0.0
                    };
                }
            }
        );
    }

    async processMultimodalInput(voiceCommand) {
        // Extract visual context
        const context = await this.visualExtractor.extractContext();

        // Resolve deictic references
        const deicticMatch = voiceCommand.match(/\b(that|this|it)\b/i);
        if (deicticMatch) {
            const resolution = this.visualExtractor.resolveDeictic(deicticMatch[1]);

            if (resolution && resolution.resolved) {
                // Process as multimodal input (voice + visual)
                const fusedInput = await this.multimodalRouter.processInput(
                    'voice',
                    voiceCommand,
                    { confidence: 0.9 }
                );

                await this.multimodalRouter.processInput(
                    'visual',
                    resolution.element.id,
                    { confidence: resolution.confidence }
                );

                return fusedInput;
            }
        }

        // Voice-only input
        return await this.multimodalRouter.processInput(
            'voice',
            voiceCommand,
            { confidence: 0.9 }
        );
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        VisualContextExtractor,
        VisualElement,
        VisualContext,
        ElementType,
        SpatialRelation,
        MultimodalVoiceSystem
    };
}
