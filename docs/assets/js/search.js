/**
 * HoloLoom Documentation Search System
 * Client-side full-text search with autocomplete, highlighting, and history
 *
 * Features:
 * - Full-text search with keyword matching
 * - Autocomplete suggestions (show as you type)
 * - Result highlighting
 * - Search ranking by relevance (title > headers > body)
 * - Keyboard navigation (up/down arrows, Enter to select)
 * - Search history in localStorage (last 10 searches)
 * - Debounced search (300ms delay)
 * - Max 10 results displayed
 * - Results grouped by page
 */

class DocumentSearch {
  constructor(indexPath = '/data/search-index.json') {
    this.indexPath = indexPath;
    this.index = null;
    this.searchIndex = null; // Tokenized index for faster searching
    this.results = [];
    this.selectedResultIndex = -1;
    this.searchHistory = this.loadSearchHistory();
    this.debounceTimer = null;
    this.isLoading = false;

    // Configuration
    this.config = {
      maxResults: 10,
      debounceDelay: 300,
      minQueryLength: 1,
      highlightClass: 'search-highlight',
      maxHistorySize: 10
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.init());
    } else {
      this.init();
    }
  }

  /**
   * Initialize search UI and load index
   */
  async init() {
    this.setupUI();
    await this.loadIndex();
    this.setupEventListeners();
  }

  /**
   * Setup search UI elements
   */
  setupUI() {
    const searchInput = document.getElementById('search-input');
    const resultsContainer = document.getElementById('search-results');

    if (!searchInput) {
      console.warn('Search input element not found (id="search-input")');
      return;
    }

    // Create results dropdown if not present
    if (!resultsContainer) {
      const dropdown = document.createElement('div');
      dropdown.id = 'search-results';
      dropdown.className = 'search-results';
      dropdown.setAttribute('role', 'region');
      dropdown.setAttribute('aria-label', 'Search results');
      dropdown.setAttribute('aria-live', 'polite');
      searchInput.parentElement.appendChild(dropdown);
    }
  }

  /**
   * Load and parse search index
   */
  async loadIndex() {
    try {
      const response = await fetch(this.indexPath);

      if (!response.ok) {
        console.warn(`Failed to load search index: ${response.status}`);
        return;
      }

      this.index = await response.json();
      this.buildSearchIndex();
    } catch (error) {
      console.warn('Error loading search index:', error);
      this.index = { pages: [] };
    }
  }

  /**
   * Build tokenized search index for fast lookups
   * Creates a map: token -> [{ pageIndex, sectionIndex, frequency, type }]
   */
  buildSearchIndex() {
    if (!this.index || !this.index.pages) return;

    this.searchIndex = {};

    this.index.pages.forEach((page, pageIdx) => {
      // Index title
      this.indexText(page.title, pageIdx, -1, 'title', 3.0);

      // Index sections
      if (page.sections && Array.isArray(page.sections)) {
        page.sections.forEach((section, secIdx) => {
          if (section.heading) {
            this.indexText(section.heading, pageIdx, secIdx, 'heading', 2.0);
          }
          if (section.content) {
            this.indexText(section.content, pageIdx, secIdx, 'body', 1.0);
          }
        });
      }
    });
  }

  /**
   * Add text to search index with tokenization
   */
  indexText(text, pageIdx, sectionIdx, type, weight) {
    if (!text) return;

    // Tokenize: lowercase, split on whitespace/punctuation, remove empty
    const tokens = text
      .toLowerCase()
      .split(/[\s\p{P}]+/u)
      .filter(t => t.length > 0);

    tokens.forEach(token => {
      if (!this.searchIndex[token]) {
        this.searchIndex[token] = [];
      }

      // Check if this token is already indexed for this location
      const existing = this.searchIndex[token].find(
        e => e.pageIdx === pageIdx && e.sectionIdx === sectionIdx && e.type === type
      );

      if (existing) {
        existing.frequency += weight;
      } else {
        this.searchIndex[token].push({
          pageIdx,
          sectionIdx,
          type,
          weight
        });
      }
    });
  }

  /**
   * Perform search on query string
   */
  search(query) {
    if (!this.index || !this.searchIndex) {
      return [];
    }

    query = query.trim().toLowerCase();

    if (query.length < this.config.minQueryLength) {
      return [];
    }

    // Tokenize query
    const queryTokens = query.split(/[\s\p{P}]+/u).filter(t => t.length > 0);

    if (queryTokens.length === 0) {
      return [];
    }

    // Score each page
    const pageScores = new Map();

    queryTokens.forEach(token => {
      if (this.searchIndex[token]) {
        this.searchIndex[token].forEach(hit => {
          const pageIdx = hit.pageIdx;
          const score = this.scoreHit(token, hit, query);

          if (!pageScores.has(pageIdx)) {
            pageScores.set(pageIdx, {
              score: 0,
              hits: [],
              page: this.index.pages[pageIdx]
            });
          }

          const entry = pageScores.get(pageIdx);
          entry.score += score;
          entry.hits.push({ ...hit, token, score });
        });
      }
    });

    // Combine with exact phrase matching bonus
    const phraseBonus = this.scorePhraseMatch(query);
    phraseBonus.forEach((bonus, pageIdx) => {
      if (pageScores.has(pageIdx)) {
        pageScores.get(pageIdx).score += bonus;
      }
    });

    // Convert to array and sort by score
    const results = Array.from(pageScores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, this.config.maxResults)
      .map(entry => ({
        url: entry.page.url,
        title: entry.page.title,
        excerpt: this.generateExcerpt(entry.page, queryTokens),
        score: entry.score,
        hits: entry.hits,
        queryTokens
      }));

    return results;
  }

  /**
   * Score individual hit
   */
  scoreHit(token, hit, fullQuery) {
    let score = hit.weight;

    // Type-based multiplier
    const typeMultiplier = {
      'title': 3.0,
      'heading': 2.0,
      'body': 1.0
    };
    score *= typeMultiplier[hit.type] || 1.0;

    // Fuzzy matching penalty (simple Levenshtein distance)
    const distance = this.levenshteinDistance(token, hit.token || '');
    if (distance === 0) {
      score *= 1.2; // Exact match bonus
    } else if (distance <= 1) {
      score *= 0.9; // Small typo penalty
    } else if (distance <= 2) {
      score *= 0.7; // Larger typo penalty
    } else {
      score *= 0.5; // Significant difference
    }

    return score;
  }

  /**
   * Score phrase matches (consecutive tokens)
   */
  scorePhraseMatch(query) {
    const bonusMap = new Map();

    this.index.pages.forEach((page, pageIdx) => {
      let text = page.title.toLowerCase();
      if (page.sections) {
        text += ' ' + page.sections
          .map(s => (s.heading || '') + ' ' + (s.content || ''))
          .join(' ')
          .toLowerCase();
      }

      if (text.includes(query)) {
        bonusMap.set(pageIdx, (bonusMap.get(pageIdx) || 0) + 5.0);
      }
    });

    return bonusMap;
  }

  /**
   * Generate excerpt around matched terms
   */
  generateExcerpt(page, queryTokens, contextLength = 150) {
    let text = page.title + ' ';

    if (page.sections && page.sections.length > 0) {
      // Combine content from all sections
      text += page.sections
        .map(s => (s.content || ''))
        .join(' ');
    }

    // Find first occurrence of any query token
    let startIdx = -1;
    for (const token of queryTokens) {
      const idx = text.toLowerCase().indexOf(token);
      if (idx !== -1 && (startIdx === -1 || idx < startIdx)) {
        startIdx = idx;
      }
    }

    if (startIdx === -1) {
      return text.substring(0, contextLength) + '...';
    }

    // Extract context around match
    const contextStart = Math.max(0, startIdx - contextLength / 2);
    const contextEnd = Math.min(text.length, startIdx + contextLength);
    let excerpt = text.substring(contextStart, contextEnd);

    if (contextStart > 0) excerpt = '...' + excerpt;
    if (contextEnd < text.length) excerpt = excerpt + '...';

    return excerpt;
  }

  /**
   * Simple Levenshtein distance for typo detection
   */
  levenshteinDistance(a, b) {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1, // substitution
            matrix[i][j - 1] + 1,     // insertion
            matrix[i - 1][j] + 1      // deletion
          );
        }
      }
    }

    return matrix[b.length][a.length];
  }

  /**
   * Get autocomplete suggestions
   */
  getAutocomplete(query, limit = 5) {
    if (!this.searchIndex || query.length < 1) {
      return [];
    }

    const queryLower = query.toLowerCase();
    const suggestions = new Set();

    // Find all tokens starting with query
    Object.keys(this.searchIndex).forEach(token => {
      if (token.startsWith(queryLower) && token !== queryLower) {
        suggestions.add(token);
      }
    });

    return Array.from(suggestions)
      .sort()
      .slice(0, limit);
  }

  /**
   * Setup event listeners
   */
  setupEventListeners() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;

    // Search on input (with debounce)
    searchInput.addEventListener('input', (e) => {
      this.onSearchInput(e.target.value);
    });

    // Keyboard navigation in results
    searchInput.addEventListener('keydown', (e) => {
      this.onSearchKeydown(e);
    });

    // Close results on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('#search-input') && !e.target.closest('#search-results')) {
        this.closeResults();
      }
    });
  }

  /**
   * Handle search input
   */
  onSearchInput(query) {
    // Clear debounce timer
    clearTimeout(this.debounceTimer);

    // Close results if query is empty
    if (query.trim().length === 0) {
      this.closeResults();
      return;
    }

    // Debounce search
    this.debounceTimer = setTimeout(() => {
      this.performSearch(query);
    }, this.config.debounceDelay);
  }

  /**
   * Perform search and render results
   */
  performSearch(query) {
    this.results = this.search(query);
    this.selectedResultIndex = -1;
    this.renderResults(query);
  }

  /**
   * Render search results
   */
  renderResults(query) {
    const container = document.getElementById('search-results');
    if (!container) return;

    if (this.results.length === 0) {
      container.innerHTML = `<div class="search-no-results">No results for "${query}"</div>`;
      container.classList.add('open');
      return;
    }

    const html = `
      <div class="search-results-header">
        <span class="search-result-count">${this.results.length} result${this.results.length === 1 ? '' : 's'}</span>
        <div class="search-history-toggle">
          <button class="search-clear-history" aria-label="Clear search history">⌫</button>
        </div>
      </div>
      <ul class="search-results-list" role="listbox">
        ${this.results.map((result, idx) => `
          <li class="search-result-item"
              role="option"
              data-index="${idx}"
              data-url="${result.url}">
            <div class="search-result-title">
              <a href="${result.url}">${this.highlightQuery(result.title, result.queryTokens)}</a>
            </div>
            <div class="search-result-excerpt">
              ${this.highlightQuery(result.excerpt, result.queryTokens)}
            </div>
            <div class="search-result-url">${result.url}</div>
          </li>
        `).join('')}
      </ul>
    `;

    container.innerHTML = html;
    container.classList.add('open');

    // Add click handlers
    container.querySelectorAll('.search-result-item').forEach((item, idx) => {
      item.addEventListener('click', (e) => {
        if (!e.target.closest('a')) {
          window.location.href = item.dataset.url;
        }
      });

      item.addEventListener('mouseover', () => {
        this.setSelectedResult(idx);
      });
    });

    // Clear history button
    const clearBtn = container.querySelector('.search-clear-history');
    if (clearBtn) {
      clearBtn.addEventListener('click', (e) => {
        e.preventDefault();
        this.clearSearchHistory();
      });
    }
  }

  /**
   * Highlight query terms in text
   */
  highlightQuery(text, queryTokens) {
    if (!queryTokens || queryTokens.length === 0) {
      return this.escapeHtml(text);
    }

    let result = this.escapeHtml(text);

    queryTokens.forEach(token => {
      const regex = new RegExp(`(${token})`, 'gi');
      result = result.replace(regex, `<span class="${this.config.highlightClass}">$1</span>`);
    });

    return result;
  }

  /**
   * Escape HTML special characters
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Handle keyboard navigation in results
   */
  onSearchKeydown(e) {
    const container = document.getElementById('search-results');
    if (!container || !container.classList.contains('open')) {
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        this.setSelectedResult(this.selectedResultIndex + 1);
        break;

      case 'ArrowUp':
        e.preventDefault();
        this.setSelectedResult(this.selectedResultIndex - 1);
        break;

      case 'Enter':
        e.preventDefault();
        if (this.selectedResultIndex >= 0 && this.results[this.selectedResultIndex]) {
          window.location.href = this.results[this.selectedResultIndex].url;
        }
        break;

      case 'Escape':
        e.preventDefault();
        this.closeResults();
        e.target.blur();
        break;
    }
  }

  /**
   * Set selected result index
   */
  setSelectedResult(index) {
    const items = document.querySelectorAll('.search-result-item');

    // Clamp index
    if (index < 0) index = items.length - 1;
    if (index >= items.length) index = 0;

    // Remove active class from all
    items.forEach(item => item.classList.remove('active'));

    // Add to new selected
    if (items[index]) {
      items[index].classList.add('active');
      items[index].scrollIntoView({ block: 'nearest' });
    }

    this.selectedResultIndex = index;
  }

  /**
   * Close search results
   */
  closeResults() {
    const container = document.getElementById('search-results');
    if (container) {
      container.classList.remove('open');
      container.innerHTML = '';
    }
    this.selectedResultIndex = -1;
  }

  /**
   * Search history management
   */
  loadSearchHistory() {
    try {
      const stored = localStorage.getItem('hololoom-search-history');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  saveSearchHistory() {
    try {
      localStorage.setItem(
        'hololoom-search-history',
        JSON.stringify(this.searchHistory.slice(0, this.config.maxHistorySize))
      );
    } catch {
      // Silently fail if localStorage not available
    }
  }

  addToSearchHistory(query) {
    if (!query || query.trim().length === 0) return;

    // Remove duplicates
    this.searchHistory = this.searchHistory.filter(
      item => item.toLowerCase() !== query.toLowerCase()
    );

    // Add to front
    this.searchHistory.unshift(query);

    // Keep only max items
    this.searchHistory = this.searchHistory.slice(0, this.config.maxHistorySize);

    this.saveSearchHistory();
  }

  getSearchHistory() {
    return [...this.searchHistory];
  }

  clearSearchHistory() {
    this.searchHistory = [];
    localStorage.removeItem('hololoom-search-history');
  }

  /**
   * Get search statistics
   */
  getStatistics() {
    return {
      indexedPages: this.index?.pages?.length || 0,
      indexedSections: this.index?.pages?.reduce(
        (sum, page) => sum + (page.sections?.length || 0),
        0
      ) || 0,
      indexTokens: Object.keys(this.searchIndex || {}).length,
      searchHistory: this.searchHistory.length
    };
  }
}

// Auto-initialize on page load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.hololoomSearch = new DocumentSearch('/data/search-index.json');
  });
} else {
  window.hololoomSearch = new DocumentSearch('/data/search-index.json');
}
