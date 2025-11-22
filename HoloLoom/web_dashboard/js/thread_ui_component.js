/**
 * Thread UI Component
 * Milestone 1: Card-based thread visualization
 * Date: November 2025
 *
 * Provides visual representation of conversation threads with:
 * - Thread cards (name, state, message count, summary)
 * - State-based styling (inactive, active, background, archived)
 * - Smooth transitions and animations
 * - Click-to-activate functionality
 */

class ThreadCard {
    constructor(thread, callbacks = {}) {
        this.thread = thread;
        this.callbacks = callbacks;
        this.element = null;

        this._createCard();
    }

    /**
     * Create card DOM element
     */
    _createCard() {
        const card = document.createElement('div');
        card.className = `thread-card thread-card--${this.thread.state}`;
        card.dataset.threadId = this.thread.id;

        // Header
        const header = document.createElement('div');
        header.className = 'thread-card__header';

        const title = document.createElement('h3');
        title.className = 'thread-card__title';
        title.textContent = this.thread.name;

        const badge = document.createElement('span');
        badge.className = 'thread-card__badge';
        badge.textContent = this._getStateBadge();

        header.appendChild(title);
        header.appendChild(badge);

        // Summary
        const summary = document.createElement('div');
        summary.className = 'thread-card__summary';
        summary.textContent = this.thread.summary || 'No messages yet';

        // Footer
        const footer = document.createElement('div');
        footer.className = 'thread-card__footer';

        const messageCount = document.createElement('span');
        messageCount.className = 'thread-card__message-count';
        messageCount.textContent = `${this.thread.messages.length} messages`;

        const lastActive = document.createElement('span');
        lastActive.className = 'thread-card__last-active';
        lastActive.textContent = this._formatLastActive();

        footer.appendChild(messageCount);
        footer.appendChild(lastActive);

        // Actions
        const actions = document.createElement('div');
        actions.className = 'thread-card__actions';

        if (this.thread.state !== 'active') {
            const activateBtn = document.createElement('button');
            activateBtn.className = 'thread-card__btn thread-card__btn--activate';
            activateBtn.textContent = 'Activate';
            activateBtn.onclick = () => this._handleActivate();
            actions.appendChild(activateBtn);
        }

        if (this.thread.state !== 'archived') {
            const archiveBtn = document.createElement('button');
            archiveBtn.className = 'thread-card__btn thread-card__btn--archive';
            archiveBtn.textContent = 'Archive';
            archiveBtn.onclick = () => this._handleArchive();
            actions.appendChild(archiveBtn);
        }

        // Assemble card
        card.appendChild(header);
        card.appendChild(summary);
        card.appendChild(footer);
        card.appendChild(actions);

        // Click handler (activate on click)
        card.onclick = (e) => {
            if (!e.target.classList.contains('thread-card__btn')) {
                this._handleActivate();
            }
        };

        this.element = card;
    }

    /**
     * Get state badge text
     */
    _getStateBadge() {
        const badges = {
            'active': '● Active',
            'background': '○ Background',
            'inactive': '○ Inactive',
            'archived': '✓ Archived'
        };
        return badges[this.thread.state] || this.thread.state;
    }

    /**
     * Format last active timestamp
     */
    _formatLastActive() {
        const now = Date.now();
        const lastActive = new Date(this.thread.lastActive).getTime();
        const diffMs = now - lastActive;

        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${diffDays}d ago`;
    }

    /**
     * Handle activate button click
     */
    _handleActivate() {
        if (this.thread.state === 'active') return;

        if (this.callbacks.onActivate) {
            this.callbacks.onActivate(this.thread);
        }
    }

    /**
     * Handle archive button click
     */
    _handleArchive() {
        if (this.callbacks.onArchive) {
            this.callbacks.onArchive(this.thread);
        }
    }

    /**
     * Update card with new thread data
     */
    update(thread) {
        this.thread = thread;

        // Update state class
        this.element.className = `thread-card thread-card--${thread.state}`;

        // Update badge
        const badge = this.element.querySelector('.thread-card__badge');
        badge.textContent = this._getStateBadge();

        // Update summary
        const summary = this.element.querySelector('.thread-card__summary');
        summary.textContent = thread.summary || 'No messages yet';

        // Update message count
        const messageCount = this.element.querySelector('.thread-card__message-count');
        messageCount.textContent = `${thread.messages.length} messages`;

        // Update last active
        const lastActive = this.element.querySelector('.thread-card__last-active');
        lastActive.textContent = this._formatLastActive();

        // Update actions
        const actions = this.element.querySelector('.thread-card__actions');
        actions.innerHTML = '';

        if (thread.state !== 'active') {
            const activateBtn = document.createElement('button');
            activateBtn.className = 'thread-card__btn thread-card__btn--activate';
            activateBtn.textContent = 'Activate';
            activateBtn.onclick = () => this._handleActivate();
            actions.appendChild(activateBtn);
        }

        if (thread.state !== 'archived') {
            const archiveBtn = document.createElement('button');
            archiveBtn.className = 'thread-card__btn thread-card__btn--archive';
            archiveBtn.textContent = 'Archive';
            archiveBtn.onclick = () => this._handleArchive();
            actions.appendChild(archiveBtn);
        }
    }

    /**
     * Highlight card (for visual feedback)
     */
    highlight() {
        this.element.classList.add('thread-card--highlighted');
        setTimeout(() => {
            this.element.classList.remove('thread-card--highlighted');
        }, 1000);
    }

    /**
     * Remove card from DOM
     */
    destroy() {
        if (this.element && this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }
}


class ThreadListView {
    constructor(container, threadManager, callbacks = {}) {
        this.container = container;
        this.threadManager = threadManager;
        this.callbacks = callbacks;
        this.cards = new Map(); // threadId → ThreadCard

        this._createView();
    }

    /**
     * Create list view structure
     */
    _createView() {
        this.container.innerHTML = '';
        this.container.className = 'thread-list-view';

        // Header
        const header = document.createElement('div');
        header.className = 'thread-list-view__header';

        const title = document.createElement('h2');
        title.textContent = 'Conversation Threads';

        const createBtn = document.createElement('button');
        createBtn.className = 'thread-list-view__create-btn';
        createBtn.textContent = '+ New Thread';
        createBtn.onclick = () => this._handleCreateThread();

        header.appendChild(title);
        header.appendChild(createBtn);

        // Thread list container
        const listContainer = document.createElement('div');
        listContainer.className = 'thread-list-view__list';

        // Empty state
        const emptyState = document.createElement('div');
        emptyState.className = 'thread-list-view__empty';
        emptyState.textContent = 'No threads yet. Say "create a new thread about..." to get started.';
        listContainer.appendChild(emptyState);

        this.container.appendChild(header);
        this.container.appendChild(listContainer);

        this.listContainer = listContainer;
        this.emptyState = emptyState;
    }

    /**
     * Handle create thread button
     */
    _handleCreateThread() {
        const topic = prompt('Thread topic:');
        if (topic) {
            if (this.callbacks.onCreate) {
                this.callbacks.onCreate(topic);
            }
        }
    }

    /**
     * Add thread to view
     */
    addThread(thread) {
        // Remove empty state if this is first thread
        if (this.cards.size === 0 && this.emptyState) {
            this.emptyState.style.display = 'none';
        }

        // Create card
        const card = new ThreadCard(thread, {
            onActivate: (t) => {
                if (this.callbacks.onActivate) {
                    this.callbacks.onActivate(t);
                }
            },
            onArchive: (t) => {
                if (this.callbacks.onArchive) {
                    this.callbacks.onArchive(t);
                }
            }
        });

        this.cards.set(thread.id, card);

        // Insert based on state (active first, then background, then inactive, then archived)
        const stateOrder = { 'active': 0, 'background': 1, 'inactive': 2, 'archived': 3 };
        const cardStateOrder = stateOrder[thread.state] || 999;

        let inserted = false;
        for (const child of this.listContainer.children) {
            if (child.classList.contains('thread-list-view__empty')) continue;

            const childThreadId = child.dataset.threadId;
            const childCard = this.cards.get(childThreadId);
            if (childCard) {
                const childStateOrder = stateOrder[childCard.thread.state] || 999;
                if (cardStateOrder < childStateOrder) {
                    this.listContainer.insertBefore(card.element, child);
                    inserted = true;
                    break;
                }
            }
        }

        if (!inserted) {
            this.listContainer.appendChild(card.element);
        }

        // Highlight new card
        card.highlight();
    }

    /**
     * Update thread in view
     */
    updateThread(thread) {
        const card = this.cards.get(thread.id);
        if (card) {
            card.update(thread);

            // Re-sort if state changed
            this._resortCards();

            // Highlight updated card
            card.highlight();
        }
    }

    /**
     * Remove thread from view
     */
    removeThread(threadId) {
        const card = this.cards.get(threadId);
        if (card) {
            card.destroy();
            this.cards.delete(threadId);

            // Show empty state if no threads left
            if (this.cards.size === 0 && this.emptyState) {
                this.emptyState.style.display = 'block';
            }
        }
    }

    /**
     * Re-sort cards by state
     */
    _resortCards() {
        const stateOrder = { 'active': 0, 'background': 1, 'inactive': 2, 'archived': 3 };

        const sortedCards = Array.from(this.cards.values()).sort((a, b) => {
            const aOrder = stateOrder[a.thread.state] || 999;
            const bOrder = stateOrder[b.thread.state] || 999;
            return aOrder - bOrder;
        });

        // Re-insert in order
        for (const card of sortedCards) {
            this.listContainer.appendChild(card.element);
        }
    }

    /**
     * Render all threads from thread manager
     */
    renderAll() {
        // Clear existing cards
        for (const card of this.cards.values()) {
            card.destroy();
        }
        this.cards.clear();

        // Get all threads
        const threads = Array.from(this.threadManager.threads.values());

        if (threads.length === 0) {
            this.emptyState.style.display = 'block';
            return;
        }

        this.emptyState.style.display = 'none';

        // Add all threads
        for (const thread of threads) {
            this.addThread(thread);
        }
    }

    /**
     * Get metrics for display
     */
    getMetrics() {
        const metrics = this.threadManager.getMetrics();
        return {
            totalThreads: metrics.totalThreads,
            activeThreads: metrics.activeThreads,
            backgroundThreads: metrics.backgroundThreads,
            archivedThreads: metrics.archivedThreads
        };
    }
}


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ThreadCard, ThreadListView };
}
