/**
 * Navigation Module for HoloLoom Documentation Site
 * Handles sticky navbar, mobile menu, active states, keyboard shortcuts, and breadcrumbs
 *
 * Features:
 * - Sticky navbar with scroll detection
 * - Mobile hamburger menu toggle
 * - Active page highlighting in navigation
 * - Keyboard shortcuts (/, ?, Esc, d, arrows)
 * - Breadcrumb generation from page structure
 * - Smooth scrolling to anchor links
 * - Sidebar collapsible sections
 * - Focus management for accessibility
 *
 * @module nav
 * @version 1.0.0
 */

(function() {
  'use strict';

  // ============================================
  // CONFIGURATION
  // ============================================

  const CONFIG = {
    // Selectors
    navbar: 'nav',
    hamburgerBtn: '.hamburger',
    mobileNav: '.mobile-nav',
    navLinks: 'nav a',
    sidebarToggle: '.sidebar-toggle',
    sidebarSections: '.sidebar-section',
    searchInput: '#search-input',
    searchButton: '.search-button',

    // Breakpoints
    mobileBreakpoint: 768,

    // Scroll debounce delay (ms)
    scrollDebounceDelay: 150,

    // Classes
    activeClass: 'active',
    expandedClass: 'expanded',
    stickyClass: 'sticky',
    visibleClass: 'visible'
  };

  // ============================================
  // STATE
  // ============================================

  const state = {
    isScrolling: false,
    isMobileMenuOpen: false,
    currentPath: window.location.pathname,
    scrollTimeout: null,
    previousScrollY: 0
  };

  // ============================================
  // UTILITY FUNCTIONS
  // ============================================

  /**
   * Get elements safely without throwing errors
   * @param {string} selector - CSS selector
   * @returns {NodeList} Matching elements (empty NodeList if none found)
   */
  function safeQuery(selector) {
    try {
      return document.querySelectorAll(selector);
    } catch (e) {
      console.warn(`Invalid selector: ${selector}`);
      return [];
    }
  }

  /**
   * Get single element safely
   * @param {string} selector - CSS selector
   * @returns {Element|null} First matching element or null
   */
  function safeQueryOne(selector) {
    try {
      return document.querySelector(selector);
    } catch (e) {
      console.warn(`Invalid selector: ${selector}`);
      return null;
    }
  }

  /**
   * Check if element is in viewport
   * @param {Element} el - Element to check
   * @returns {boolean} True if element is visible
   */
  function isElementInViewport(el) {
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
  }

  /**
   * Normalize URL path for comparison
   * @param {string} path - URL path
   * @returns {string} Normalized path
   */
  function normalizePath(path) {
    // Remove trailing slash and make lowercase
    return path.toLowerCase().replace(/\/$/, '') || '/';
  }

  /**
   * Debounce function
   * @param {Function} func - Function to debounce
   * @param {number} delay - Delay in ms
   * @returns {Function} Debounced function
   */
  function debounce(func, delay) {
    let timeoutId;
    return function debounced(...args) {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
  }

  // ============================================
  // STICKY NAVBAR
  // ============================================

  /**
   * Handle navbar sticky behavior on scroll
   */
  function updateStickyNavbar() {
    const navbar = safeQueryOne(CONFIG.navbar);
    if (!navbar) return;

    const scrollY = window.scrollY;
    const navHeight = navbar.offsetHeight;

    if (scrollY > navHeight) {
      navbar.classList.add(CONFIG.stickyClass);
    } else {
      navbar.classList.remove(CONFIG.stickyClass);
    }

    state.previousScrollY = scrollY;
  }

  // Create debounced scroll handler
  const debouncedStickyUpdate = debounce(updateStickyNavbar, CONFIG.scrollDebounceDelay);

  /**
   * Initialize sticky navbar
   */
  function initStickyNavbar() {
    window.addEventListener('scroll', debouncedStickyUpdate, { passive: true });
    // Initial call
    updateStickyNavbar();
  }

  // ============================================
  // MOBILE MENU
  // ============================================

  /**
   * Toggle mobile menu visibility
   */
  function toggleMobileMenu() {
    const mobileNav = safeQueryOne(CONFIG.mobileNav);
    if (!mobileNav) return;

    state.isMobileMenuOpen = !state.isMobileMenuOpen;

    if (state.isMobileMenuOpen) {
      mobileNav.setAttribute('aria-hidden', 'false');
      mobileNav.classList.add(CONFIG.visibleClass);
      // Trap focus in menu
      focusTrapMenu(mobileNav);
    } else {
      mobileNav.setAttribute('aria-hidden', 'true');
      mobileNav.classList.remove(CONFIG.visibleClass);
    }

    updateHamburgerButton();
  }

  /**
   * Close mobile menu
   */
  function closeMobileMenu() {
    const mobileNav = safeQueryOne(CONFIG.mobileNav);
    if (!mobileNav) return;

    state.isMobileMenuOpen = false;
    mobileNav.setAttribute('aria-hidden', 'true');
    mobileNav.classList.remove(CONFIG.visibleClass);
    updateHamburgerButton();
  }

  /**
   * Update hamburger button state
   */
  function updateHamburgerButton() {
    const btn = safeQueryOne(CONFIG.hamburgerBtn);
    if (!btn) return;

    if (state.isMobileMenuOpen) {
      btn.setAttribute('aria-expanded', 'true');
      btn.classList.add(CONFIG.activeClass);
    } else {
      btn.setAttribute('aria-expanded', 'false');
      btn.classList.remove(CONFIG.activeClass);
    }
  }

  /**
   * Trap focus within mobile menu (accessibility)
   * @param {Element} menu - Menu element
   */
  function focusTrapMenu(menu) {
    const focusableElements = menu.querySelectorAll(
      'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    menu.addEventListener('keydown', (e) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        // Shift+Tab on first element -> focus last
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        // Tab on last element -> focus first
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    });
  }

  /**
   * Initialize mobile menu
   */
  function initMobileMenu() {
    const hamburger = safeQueryOne(CONFIG.hamburgerBtn);
    if (!hamburger) return;

    // Set initial aria attributes
    hamburger.setAttribute('aria-expanded', 'false');
    hamburger.setAttribute('aria-label', 'Toggle navigation menu');

    const mobileNav = safeQueryOne(CONFIG.mobileNav);
    if (mobileNav) {
      mobileNav.setAttribute('aria-hidden', 'true');
    }

    // Toggle on button click
    hamburger.addEventListener('click', toggleMobileMenu);

    // Close on link click
    safeQuery(`${CONFIG.mobileNav} a`).forEach(link => {
      link.addEventListener('click', closeMobileMenu);
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && state.isMobileMenuOpen) {
        closeMobileMenu();
      }
    });
  }

  // ============================================
  // ACTIVE PAGE HIGHLIGHTING
  // ============================================

  /**
   * Highlight active navigation item based on current page
   */
  function updateActiveNavItems() {
    const currentPath = normalizePath(state.currentPath);
    const navLinks = safeQuery(CONFIG.navLinks);

    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (!href) return;

      const linkPath = normalizePath(href);
      const isActive =
        linkPath === currentPath ||
        currentPath.startsWith(linkPath + '/') ||
        (linkPath === '/' && currentPath === '/');

      if (isActive) {
        link.classList.add(CONFIG.activeClass);
        link.setAttribute('aria-current', 'page');

        // Expand parent sections in sidebar
        let parent = link.closest(CONFIG.sidebarSections);
        while (parent) {
          parent.classList.add(CONFIG.expandedClass);
          const toggle = parent.querySelector(CONFIG.sidebarToggle);
          if (toggle) {
            toggle.setAttribute('aria-expanded', 'true');
          }
          parent = parent.parentElement.closest(CONFIG.sidebarSections);
        }
      } else {
        link.classList.remove(CONFIG.activeClass);
        link.removeAttribute('aria-current');
      }
    });
  }

  /**
   * Initialize active navigation state
   */
  function initActiveNavigation() {
    updateActiveNavItems();
  }

  // ============================================
  // SIDEBAR COLLAPSIBLE SECTIONS
  // ============================================

  /**
   * Toggle sidebar section expansion
   * @param {Element} toggle - Toggle button element
   */
  function toggleSidebarSection(toggle) {
    const section = toggle.closest(CONFIG.sidebarSections);
    if (!section) return;

    const isExpanded = section.classList.contains(CONFIG.expandedClass);
    const content = section.querySelector('ul, ol, div[class*="content"]');

    if (isExpanded) {
      section.classList.remove(CONFIG.expandedClass);
      toggle.setAttribute('aria-expanded', 'false');
      if (content) {
        content.style.maxHeight = '0';
      }
    } else {
      section.classList.add(CONFIG.expandedClass);
      toggle.setAttribute('aria-expanded', 'true');
      if (content) {
        content.style.maxHeight = content.scrollHeight + 'px';
      }
    }
  }

  /**
   * Initialize sidebar collapsible sections
   */
  function initSidebarSections() {
    const toggles = safeQuery(`${CONFIG.sidebarSections} ${CONFIG.sidebarToggle}`);

    toggles.forEach(toggle => {
      // Set initial aria attributes
      const isExpanded = toggle.closest(CONFIG.sidebarSections)
        ?.classList.contains(CONFIG.expandedClass) ?? true;
      toggle.setAttribute('aria-expanded', isExpanded);

      // Toggle on click
      toggle.addEventListener('click', (e) => {
        e.preventDefault();
        toggleSidebarSection(toggle);
      });

      // Allow arrow keys to navigate
      toggle.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
          e.preventDefault();
          const section = toggle.closest(CONFIG.sidebarSections);
          const allToggles = Array.from(safeQuery(CONFIG.sidebarToggle));
          const currentIndex = allToggles.indexOf(toggle);

          if (e.key === 'ArrowDown' && currentIndex < allToggles.length - 1) {
            allToggles[currentIndex + 1].focus();
          } else if (e.key === 'ArrowUp' && currentIndex > 0) {
            allToggles[currentIndex - 1].focus();
          }
        }
      });
    });
  }

  // ============================================
  // KEYBOARD SHORTCUTS
  // ============================================

  /**
   * Handle keyboard shortcuts
   * @param {KeyboardEvent} e - Keyboard event
   */
  function handleKeyboardShortcuts(e) {
    // Don't trigger shortcuts when typing in input fields
    if (e.target.matches('input, textarea, [contenteditable]')) {
      return;
    }

    switch (e.key) {
      case '/':
        // Focus search box
        e.preventDefault();
        const searchInput = safeQueryOne(CONFIG.searchInput);
        if (searchInput) {
          searchInput.focus();
        }
        break;

      case '?':
        // Show keyboard shortcuts help
        e.preventDefault();
        showKeyboardShortcutsModal();
        break;

      case 'Escape':
        // Close menus and modals
        e.preventDefault();
        closeMobileMenu();
        closeAllModals();
        break;

      case 'd':
        // Toggle theme (handled by theme.js)
        if (e.ctrlKey || e.metaKey) {
          // theme.js handles this
        }
        break;

      case 'ArrowLeft':
        // Go to previous page
        if (e.ctrlKey || e.metaKey) {
          navigatePreviousPage();
        }
        break;

      case 'ArrowRight':
        // Go to next page
        if (e.ctrlKey || e.metaKey) {
          navigateNextPage();
        }
        break;

      case 't':
        // Jump to table of contents
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          const toc = safeQueryOne('.table-of-contents');
          if (toc) {
            toc.scrollIntoView({ behavior: 'smooth' });
          }
        }
        break;
    }
  }

  /**
   * Show keyboard shortcuts help modal
   */
  function showKeyboardShortcutsModal() {
    const modal = safeQueryOne('.shortcuts-modal');
    if (modal) {
      modal.classList.add(CONFIG.visibleClass);
      modal.setAttribute('aria-hidden', 'false');
      // Focus first close button
      const closeBtn = modal.querySelector('.modal-close');
      if (closeBtn) closeBtn.focus();
    }
  }

  /**
   * Close all modals
   */
  function closeAllModals() {
    safeQuery('[role="dialog"], .modal').forEach(modal => {
      modal.classList.remove(CONFIG.visibleClass);
      modal.setAttribute('aria-hidden', 'true');
    });
  }

  /**
   * Navigate to previous page
   */
  function navigatePreviousPage() {
    const prevLink = safeQueryOne('a[rel="prev"], .prev-page');
    if (prevLink) {
      window.location.href = prevLink.getAttribute('href');
    }
  }

  /**
   * Navigate to next page
   */
  function navigateNextPage() {
    const nextLink = safeQueryOne('a[rel="next"], .next-page');
    if (nextLink) {
      window.location.href = nextLink.getAttribute('href');
    }
  }

  /**
   * Initialize keyboard shortcuts
   */
  function initKeyboardShortcuts() {
    document.addEventListener('keydown', handleKeyboardShortcuts);
  }

  // ============================================
  // BREADCRUMB GENERATION
  // ============================================

  /**
   * Generate breadcrumb navigation from current path
   */
  function generateBreadcrumbs() {
    const breadcrumbContainer = safeQueryOne('.breadcrumbs');
    if (!breadcrumbContainer) return;

    const parts = state.currentPath.split('/').filter(Boolean);
    const breadcrumbs = [];

    // Add home
    breadcrumbs.push({
      label: 'Home',
      url: '/'
    });

    // Add path segments
    let url = '';
    parts.forEach((part) => {
      url += '/' + part;
      breadcrumbs.push({
        label: part.charAt(0).toUpperCase() + part.slice(1),
        url: url
      });
    });

    // Build HTML
    const html = breadcrumbs.map((crumb, index) => {
      const isLast = index === breadcrumbs.length - 1;
      if (isLast) {
        return `<span aria-current="page">${escapeHtml(crumb.label)}</span>`;
      }
      return `<a href="${escapeHtml(crumb.url)}">${escapeHtml(crumb.label)}</a>`;
    }).join('<span class="separator"> / </span>');

    breadcrumbContainer.innerHTML = html;
  }

  /**
   * Escape HTML special characters
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  function escapeHtml(text) {
    const map = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  /**
   * Initialize breadcrumbs
   */
  function initBreadcrumbs() {
    generateBreadcrumbs();
  }

  // ============================================
  // SMOOTH SCROLLING
  // ============================================

  /**
   * Handle smooth scrolling to anchors
   */
  function initSmoothScrolling() {
    safeQuery('a[href^="#"]').forEach(link => {
      link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');

        // Skip if href is just '#'
        if (href === '#') return;

        const target = safeQueryOne(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });

          // Set focus to target for accessibility
          if (target.tabIndex === -1) {
            target.tabIndex = -1;
          }
          target.focus();
        }
      });
    });
  }

  // ============================================
  // INITIALIZATION
  // ============================================

  /**
   * Initialize all navigation features
   */
  function init() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    // Initialize features
    initStickyNavbar();
    initMobileMenu();
    initActiveNavigation();
    initSidebarSections();
    initKeyboardShortcuts();
    initBreadcrumbs();
    initSmoothScrolling();

    // Log initialization
    if (window.location.hash === '#nav-debug') {
      console.log('Navigation module initialized');
    }
  }

  // ============================================
  // PUBLIC API
  // ============================================

  // Expose public methods for external use
  window.HoloLoomNav = {
    toggleMobileMenu,
    closeMobileMenu,
    updateActiveNavItems,
    generateBreadcrumbs,
    closeAllModals,
    showKeyboardShortcutsModal
  };

  // Start initialization
  init();

})();
