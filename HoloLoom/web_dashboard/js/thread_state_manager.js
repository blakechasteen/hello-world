/**
 * Thread State Manager
 * Milestone 1: Multi-threaded conversation management
 * Date: November 2025
 *
 * Manages conversation thread lifecycle with state machine:
 * INACTIVE → ACTIVE → BACKGROUND → ARCHIVED
 */

class ThreadStateManager {
    constructor() {
        // Thread storage
        this.threads = new Map(); // id → Thread
        this.activeThreadId = null;

        // Thread ID counter
        this.nextThreadId = 1;

        // Configuration
        this.config = {
            maxActiveThreads: 10,
            maxBackgroundThreads: 20,
            autoArchiveAfterDays: 30,
            summarizationEnabled: true
        };

        // Metrics
        this.metrics = {
            totalThreadsCreated: 0,
            threadsArchived: 0,
            averageThreadDuration: 0,
            threadSwitches: 0
        };

        console.log('[ThreadStateManager] Initialized');
    }

    /**
     * Create a new thread
     */
    createThread(name, initialMessage = null) {
        // Generate unique ID
        const id = `thread_${this.nextThreadId++}`;

        // Create thread object
        const thread = {
            id: id,
            name: name,
            state: 'inactive',
            messages: [],
            summary: '',
            createdAt: new Date(),
            lastActive: new Date(),
            activationCount: 0,
            contextSnapshot: null
        };

        // Add initial message if provided
        if (initialMessage) {
            thread.messages.push({
                role: 'user',
                content: initialMessage,
                timestamp: new Date()
            });
        }

        // Store thread
        this.threads.set(id, thread);
        this.metrics.totalThreadsCreated++;

        console.log(`[ThreadStateManager] Created thread: ${name} (${id})`);

        return thread;
    }

    /**
     * Activate a thread (make it the current focus)
     */
    activateThread(threadId) {
        const thread = this.threads.get(threadId);
        if (!thread) {
            throw new Error(`Thread not found: ${threadId}`);
        }

        // Deactivate current active thread (move to background)
        if (this.activeThreadId && this.activeThreadId !== threadId) {
            const currentThread = this.threads.get(this.activeThreadId);
            if (currentThread) {
                this._transitionState(currentThread, 'background');
                this.metrics.threadSwitches++;
            }
        }

        // Activate new thread
        this._transitionState(thread, 'active');
        this.activeThreadId = threadId;
        thread.lastActive = new Date();
        thread.activationCount++;

        console.log(`[ThreadStateManager] Activated thread: ${thread.name} (${threadId})`);

        return thread;
    }

    /**
     * Get active thread
     */
    getActiveThread() {
        if (!this.activeThreadId) {
            return null;
        }
        return this.threads.get(this.activeThreadId);
    }

    /**
     * Add message to a thread
     */
    addMessage(threadId, role, content) {
        const thread = this.threads.get(threadId);
        if (!thread) {
            throw new Error(`Thread not found: ${threadId}`);
        }

        const message = {
            role: role, // 'user' or 'assistant'
            content: content,
            timestamp: new Date()
        };

        thread.messages.push(message);
        thread.lastActive = new Date();

        // Update summary if enabled
        if (this.config.summarizationEnabled && thread.messages.length % 10 === 0) {
            this._updateThreadSummary(thread);
        }

        return message;
    }

    /**
     * Close a thread (move to archived)
     */
    closeThread(threadId) {
        const thread = this.threads.get(threadId);
        if (!thread) {
            throw new Error(`Thread not found: ${threadId}`);
        }

        // Generate final summary
        this._updateThreadSummary(thread);

        // Transition to archived
        this._transitionState(thread, 'archived');
        this.metrics.threadsArchived++;

        // If this was the active thread, clear active state
        if (this.activeThreadId === threadId) {
            this.activeThreadId = null;
        }

        console.log(`[ThreadStateManager] Closed thread: ${thread.name} (${threadId})`);

        return thread;
    }

    /**
     * Find threads by name (fuzzy matching)
     */
    findThreadsByName(query) {
        const lowerQuery = query.toLowerCase();
        const matches = [];

        for (const thread of this.threads.values()) {
            const lowerName = thread.name.toLowerCase();

            // Exact match
            if (lowerName === lowerQuery) {
                matches.push({ thread, score: 1.0, matchType: 'exact' });
                continue;
            }

            // Contains match
            if (lowerName.includes(lowerQuery)) {
                matches.push({ thread, score: 0.8, matchType: 'contains' });
                continue;
            }

            // Word match
            const nameWords = lowerName.split(/\s+/);
            const queryWords = lowerQuery.split(/\s+/);
            const wordMatches = queryWords.filter(qw =>
                nameWords.some(nw => nw.startsWith(qw))
            );

            if (wordMatches.length > 0) {
                const score = wordMatches.length / queryWords.length;
                matches.push({ thread, score, matchType: 'words' });
            }
        }

        // Sort by score descending
        matches.sort((a, b) => b.score - a.score);

        return matches;
    }

    /**
     * Get all threads in a specific state
     */
    getThreadsByState(state) {
        const results = [];
        for (const thread of this.threads.values()) {
            if (thread.state === state) {
                results.push(thread);
            }
        }
        return results;
    }

    /**
     * Get all active and background threads
     */
    getActiveThreads() {
        return [
            ...this.getThreadsByState('active'),
            ...this.getThreadsByState('background')
        ];
    }

    /**
     * Generate summaries for all active threads
     */
    summarizeAllThreads() {
        const activeThreads = this.getActiveThreads();
        const summaries = [];

        for (const thread of activeThreads) {
            this._updateThreadSummary(thread);
            summaries.push({
                id: thread.id,
                name: thread.name,
                summary: thread.summary,
                messageCount: thread.messages.length,
                lastActive: thread.lastActive
            });
        }

        return summaries;
    }

    /**
     * Transition thread state (with validation)
     */
    _transitionState(thread, newState) {
        const oldState = thread.state;

        // Validate state transition
        const validTransitions = {
            'inactive': ['active', 'archived'],
            'active': ['background', 'archived'],
            'background': ['active', 'archived'],
            'archived': [] // Cannot transition out of archived
        };

        if (!validTransitions[oldState].includes(newState)) {
            console.warn(`[ThreadStateManager] Invalid state transition: ${oldState} → ${newState}`);
            return false;
        }

        // Update state
        thread.state = newState;

        console.log(`[ThreadStateManager] Thread ${thread.id} state: ${oldState} → ${newState}`);

        return true;
    }

    /**
     * Update thread summary (basic implementation)
     */
    _updateThreadSummary(thread) {
        // For Milestone 1, use simple heuristic
        // Milestone 2+ can integrate with LLM for better summaries

        if (thread.messages.length === 0) {
            thread.summary = 'No messages yet';
            return;
        }

        const recentMessages = thread.messages.slice(-5); // Last 5 messages
        const topics = new Set();

        // Extract keywords from messages (simple word extraction)
        for (const msg of recentMessages) {
            const words = msg.content.toLowerCase().split(/\s+/);
            const keywords = words.filter(w =>
                w.length > 4 && // Longer words
                !['about', 'would', 'could', 'should', 'there', 'where'].includes(w)
            );
            keywords.forEach(k => topics.add(k));
        }

        // Create summary
        if (topics.size > 0) {
            const topicList = Array.from(topics).slice(0, 3).join(', ');
            thread.summary = `Discussion about ${topicList} (${thread.messages.length} messages)`;
        } else {
            thread.summary = `${thread.messages.length} messages`;
        }
    }

    /**
     * Get metrics
     */
    getMetrics() {
        const activeCount = this.getThreadsByState('active').length;
        const backgroundCount = this.getThreadsByState('background').length;
        const archivedCount = this.getThreadsByState('archived').length;

        return {
            ...this.metrics,
            activeThreads: activeCount,
            backgroundThreads: backgroundCount,
            archivedThreads: archivedCount,
            totalThreads: this.threads.size
        };
    }

    /**
     * Serialize threads for persistence
     */
    serialize() {
        const data = {
            threads: Array.from(this.threads.values()),
            activeThreadId: this.activeThreadId,
            nextThreadId: this.nextThreadId,
            metrics: this.metrics
        };

        return JSON.stringify(data);
    }

    /**
     * Deserialize threads from persistence
     */
    deserialize(json) {
        try {
            const data = JSON.parse(json);

            // Restore threads
            this.threads = new Map();
            for (const thread of data.threads) {
                this.threads.set(thread.id, thread);
            }

            // Restore state
            this.activeThreadId = data.activeThreadId;
            this.nextThreadId = data.nextThreadId;
            this.metrics = data.metrics || this.metrics;

            console.log(`[ThreadStateManager] Restored ${this.threads.size} threads`);
            return true;
        } catch (error) {
            console.error('[ThreadStateManager] Deserialization failed:', error);
            return false;
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ThreadStateManager;
}
