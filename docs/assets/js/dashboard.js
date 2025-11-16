/**
 * BigPlay Dashboard JavaScript
 * Main controller for dashboard functionality
 * Includes: wizard, search, widgets, activity feed, customization
 */

const BigPlayDashboard = (() => {
  // ============================================
  // CONSTANTS
  // ============================================

  const STORAGE_KEYS = {
    WIDGETS: 'bigplay-widgets',
    PREFERENCES: 'bigplay-preferences',
    SEARCH_HISTORY: 'bigplay-search-history',
  };

  const WIZARD_PATHS = {
    learn: {
      title: 'Start Learning HoloLoom',
      description: 'Begin with fundamentals and progress through our 5-part training course. Perfect for beginners!',
      link: '/training/part1.html',
    },
    promptly: {
      title: 'Get Started with Promptly',
      description: 'Master prompt engineering with Promptly\'s version control and optimization tools.',
      link: '/promptly/quickstart.html',
    },
    community: {
      title: 'Join the Community',
      description: 'Connect with other developers, ask questions, and share your projects.',
      link: '#',
    },
    issues: {
      title: 'Report a Bug',
      description: 'Found an issue? Help us improve by reporting it with detailed information.',
      link: '#',
    },
    contribute: {
      title: 'Start Contributing',
      description: 'Contribute code, documentation, or ideas to the HoloLoom project.',
      link: '/contributing.html',
    },
  };

  // ============================================
  // INTERNAL STATE
  // ============================================

  let state = {
    currentWizardPath: null,
    searchIndex: null,
    activeWidgets: {},
  };

  // ============================================
  // INITIALIZATION
  // ============================================

  function init() {
    loadPreferences();
    setupTheme();
    setupWizard();
    setupSearch();
    setupActivity();
    setupAnnouncements();
    setupWidgetControls();
    setupStats();
    setupEcosystem();
  }

  // ============================================
  // THEME MANAGEMENT
  // ============================================

  function setupTheme() {
    const THEME_KEY = 'hololoom-theme';
    const themeToggle = document.getElementById('themeToggle');
    const html = document.documentElement;

    if (!themeToggle) return;

    function getInitialTheme() {
      const stored = localStorage.getItem(THEME_KEY);
      if (stored) return stored;
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function setTheme(theme) {
      html.setAttribute('data-theme', theme);
      localStorage.setItem(THEME_KEY, theme);
      updateThemeButton(theme);
    }

    function updateThemeButton(theme) {
      themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
      themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    }

    function toggleTheme() {
      const current = html.getAttribute('data-theme') || getInitialTheme();
      const next = current === 'dark' ? 'light' : 'dark';
      setTheme(next);
    }

    setTheme(getInitialTheme());
    themeToggle.addEventListener('click', toggleTheme);
  }

  // ============================================
  // WIZARD FUNCTIONALITY
  // ============================================

  function setupWizard() {
    const buttons = document.querySelectorAll('.wizard-button');
    const recommendations = document.getElementById('wizard-recommendations');

    if (!buttons.length || !recommendations) return;

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        const path = button.dataset.path;
        showRecommendation(path);
        updateWizardUI(button);
      });
    });

    // Keyboard navigation for wizard
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        buttons[0].focus();
      }
    });
  }

  function showRecommendation(path) {
    const recommendation = WIZARD_PATHS[path];
    if (!recommendation) return;

    const recommendations = document.getElementById('wizard-recommendations');
    const title = document.getElementById('rec-title');
    const description = document.getElementById('rec-description');
    const link = document.getElementById('rec-link');

    if (title && description && link) {
      title.textContent = recommendation.title;
      description.textContent = recommendation.description;
      link.href = recommendation.link;
      link.textContent = 'Get Started →';

      recommendations.classList.remove('hidden');
      recommendations.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    state.currentWizardPath = path;
    savePreferences();
  }

  function updateWizardUI(button) {
    document.querySelectorAll('.wizard-button').forEach((btn) => {
      btn.classList.remove('active');
    });
    button.classList.add('active');
  }

  // ============================================
  // SEARCH FUNCTIONALITY
  // ============================================

  function setupSearch() {
    const searchInput = document.getElementById('universal-search');
    if (!searchInput) return;

    searchInput.addEventListener('input', debounce(performSearch, 300));
    searchInput.addEventListener('focus', () => {
      // Expand search results if there's cached history
      const history = getSearchHistory();
      if (history.length > 0) {
        showSearchSuggestions(history);
      }
    });

    // Keyboard shortcut: '/' to focus search
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && !e.target.matches('input, textarea')) {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  function performSearch(e) {
    const query = e.target.value.trim();
    const resultsContainer = document.getElementById('search-results');

    if (!query) {
      resultsContainer.classList.add('hidden');
      return;
    }

    // Simulated search results (in production, this would query actual docs)
    const results = searchDocumentation(query);

    if (results.length === 0) {
      resultsContainer.innerHTML = '<div class="search-result-item"><p class="search-result-title">No results found</p></div>';
    } else {
      renderSearchResults(results);
      addToSearchHistory(query);
    }

    resultsContainer.classList.remove('hidden');
  }

  function searchDocumentation(query) {
    const query_lower = query.toLowerCase();

    // Simulated search index (in production, load from search-index.json)
    const docs = [
      { title: 'Training Part 1: Foundations', category: 'Training', url: '/training/part1.html' },
      { title: 'Training Part 2: Core Concepts', category: 'Training', url: '/training/part2.html' },
      { title: 'Training Part 3: Tutorials', category: 'Training', url: '/training/part3.html' },
      { title: 'RAG System Documentation', category: 'Documentation', url: '/docs/rag' },
      { title: 'Alignment Framework v1.0', category: 'Documentation', url: '/docs/alignment' },
      { title: 'API Reference', category: 'Reference', url: '/api/' },
      { title: 'Contributing Guide', category: 'Community', url: '/contributing.html' },
      { title: 'Architecture Overview', category: 'Documentation', url: '/docs/architecture' },
      { title: 'Performance Benchmarks', category: 'Reference', url: '/performance/' },
      { title: 'Community Forums', category: 'Community', url: '#' },
    ];

    return docs.filter((doc) => doc.title.toLowerCase().includes(query_lower));
  }

  function renderSearchResults(results) {
    const resultsContainer = document.getElementById('search-results');
    let html = '';

    // Group by category
    const categories = {};
    results.forEach((result) => {
      if (!categories[result.category]) {
        categories[result.category] = [];
      }
      categories[result.category].push(result);
    });

    // Render each category
    Object.entries(categories).forEach(([category, items]) => {
      html += `<div class="search-result-category">${category}</div>`;
      items.forEach((item) => {
        html += `
          <a href="${item.url}" class="search-result-item">
            <p class="search-result-title">${escapeHtml(item.title)}</p>
            <p class="search-result-desc">${escapeHtml(category)}</p>
          </a>
        `;
      });
    });

    resultsContainer.innerHTML = html;
  }

  function showSearchSuggestions(history) {
    const resultsContainer = document.getElementById('search-results');
    let html = '<div class="search-result-category">Recent Searches</div>';

    history.slice(0, 5).forEach((query) => {
      html += `
        <div class="search-result-item" style="cursor: pointer;">
          <p class="search-result-title">${escapeHtml(query)}</p>
        </div>
      `;
    });

    resultsContainer.innerHTML = html;
    resultsContainer.classList.remove('hidden');
  }

  function addToSearchHistory(query) {
    const history = getSearchHistory();
    if (!history.includes(query)) {
      history.unshift(query);
      localStorage.setItem(STORAGE_KEYS.SEARCH_HISTORY, JSON.stringify(history.slice(0, 20)));
    }
  }

  function getSearchHistory() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEYS.SEARCH_HISTORY) || '[]');
    } catch {
      return [];
    }
  }

  // ============================================
  // ACTIVITY FEED
  // ============================================

  function setupActivity() {
    // Activity feed is static for now, but can be updated dynamically
    // In production, this would fetch from an API
    loadActivityFeed();
  }

  function loadActivityFeed() {
    // Simulated activity updates
    // In production, fetch from backend API
    const activityCards = document.querySelectorAll('.activity-card');

    activityCards.forEach((card) => {
      // Add real-time indicator
      const title = card.querySelector('.activity-title');
      if (title) {
        const indicator = document.createElement('span');
        indicator.className = 'activity-indicator';
        indicator.textContent = '●';
        indicator.style.color = 'var(--color-success)';
        indicator.style.marginLeft = '8px';
        indicator.style.fontSize = '0.8rem';
        title.appendChild(indicator);
      }
    });
  }

  // ============================================
  // ANNOUNCEMENTS
  // ============================================

  function setupAnnouncements() {
    loadAnnouncements();
  }

  async function loadAnnouncements() {
    const banner = document.getElementById('announcement-banner');
    if (!banner) return;

    try {
      // Try to load from JSON file
      const response = await fetch('/data/announcements.json');
      const announcements = await response.json();

      if (announcements && announcements.length > 0) {
        const announcement = announcements[0]; // Show most recent
        renderAnnouncement(banner, announcement);
      }
    } catch {
      // Fallback to default announcement
      const announcement = {
        title: 'Welcome to BigPlay',
        message: 'Your unified gateway to the HoloLoom ecosystem',
        type: 'info',
        icon: '🚀',
      };
      renderAnnouncement(banner, announcement);
    }
  }

  function renderAnnouncement(container, announcement) {
    const html = `
      <div style="display: flex; align-items: start; gap: var(--spacing-md);">
        <span class="announcement-icon" aria-hidden="true">${announcement.icon || '📢'}</span>
        <div>
          <p class="announcement-title">${escapeHtml(announcement.title)}</p>
          <p class="announcement-text">${escapeHtml(announcement.message)}</p>
        </div>
      </div>
    `;
    container.innerHTML = html;
    container.classList.remove('hidden');
  }

  // ============================================
  // WIDGET CONTROLS
  // ============================================

  function setupWidgetControls() {
    const resetButton = document.getElementById('reset-dashboard');
    const checkboxes = document.querySelectorAll('.settings-checkbox input');

    if (resetButton) {
      resetButton.addEventListener('click', resetDashboard);
    }

    checkboxes.forEach((checkbox) => {
      checkbox.addEventListener('change', (e) => {
        const widgetName = e.target.name;
        const isChecked = e.target.checked;
        toggleWidget(widgetName, isChecked);
      });
    });

    // Load saved preferences
    loadWidgetPreferences();
  }

  function toggleWidget(widgetName, isChecked) {
    const widgetMap = {
      'widget-activity': '.activity-section',
      'widget-ecosystem': '.ecosystem-section',
      'widget-spotlight': '.spotlight-section',
      'widget-quick-links': '.quick-links-section',
    };

    const selector = widgetMap[widgetName];
    if (!selector) return;

    const element = document.querySelector(selector);
    if (element) {
      element.style.display = isChecked ? '' : 'none';
    }

    state.activeWidgets[widgetName] = isChecked;
    localStorage.setItem(STORAGE_KEYS.WIDGETS, JSON.stringify(state.activeWidgets));
  }

  function loadWidgetPreferences() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.WIDGETS) || '{}');
      state.activeWidgets = saved;

      // Apply saved preferences
      Object.entries(saved).forEach(([widgetName, isChecked]) => {
        const checkbox = document.querySelector(`input[name="${widgetName}"]`);
        if (checkbox) {
          checkbox.checked = isChecked;
          toggleWidget(widgetName, isChecked);
        }
      });
    } catch {
      // Use defaults
      state.activeWidgets = {
        'widget-activity': true,
        'widget-ecosystem': true,
        'widget-spotlight': true,
        'widget-quick-links': true,
      };
    }
  }

  function resetDashboard() {
    if (confirm('Reset dashboard to default layout?')) {
      localStorage.removeItem(STORAGE_KEYS.WIDGETS);
      localStorage.removeItem(STORAGE_KEYS.PREFERENCES);
      location.reload();
    }
  }

  // ============================================
  // STATISTICS
  // ============================================

  function setupStats() {
    animateStats();
    loadStatsFromJSON();
  }

  async function loadStatsFromJSON() {
    try {
      const response = await fetch('/data/stats.json');
      const stats = await response.json();

      // Update stat elements
      if (stats.downloads) {
        const elem = document.getElementById('stat-downloads');
        if (elem) elem.textContent = stats.downloads;
      }
      if (stats.stars) {
        const elem = document.getElementById('stat-stars');
        if (elem) elem.textContent = `⭐ ${stats.stars}`;
      }
      if (stats.contributors) {
        const elem = document.getElementById('stat-contributors');
        if (elem) elem.textContent = stats.contributors;
      }
      if (stats.activeThreads) {
        const elem = document.getElementById('stat-active');
        if (elem) elem.textContent = stats.activeThreads;
      }
    } catch {
      // Use defaults
    }
  }

  function animateStats() {
    const stats = document.querySelectorAll('.hero-stat-value');
    stats.forEach((stat) => {
      const text = stat.textContent;
      // Extract number
      const number = parseInt(text.replace(/[^0-9]/g, '')) || 0;

      if (number > 0) {
        animateCounter(stat, 0, number, 1500);
      }
    });
  }

  function animateCounter(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const interval = setInterval(() => {
      current += increment;
      if (current >= end) {
        current = end;
        clearInterval(interval);
      }

      const text = element.textContent;
      const prefix = text.replace(/[0-9,]/g, '');
      element.textContent = prefix + Math.floor(current).toLocaleString();
    }, 16);
  }

  // ============================================
  // ECOSYSTEM VISUALIZATION
  // ============================================

  function setupEcosystem() {
    setupEcosystemCards();
  }

  function setupEcosystemCards() {
    const cards = document.querySelectorAll('.ecosystem-card');

    cards.forEach((card) => {
      card.addEventListener('click', () => {
        const component = card.dataset.component;
        const link = card.querySelector('.ecosystem-link');
        if (link) {
          location.href = link.href;
        }
      });

      // Keyboard accessibility
      card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          card.click();
        }
      });
    });
  }

  // ============================================
  // PREFERENCES
  // ============================================

  function loadPreferences() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.PREFERENCES) || '{}');
      state = { ...state, ...saved };
    } catch {
      // Use defaults
    }
  }

  function savePreferences() {
    localStorage.setItem(STORAGE_KEYS.PREFERENCES, JSON.stringify({
      currentWizardPath: state.currentWizardPath,
    }));
  }

  // ============================================
  // UTILITIES
  // ============================================

  function debounce(func, delay) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), delay);
    };
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ============================================
  // PUBLIC API
  // ============================================

  return {
    init,
  };
})();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    BigPlayDashboard.init();
  });
} else {
  BigPlayDashboard.init();
}
