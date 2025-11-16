/**
 * HoloLoom Community Forum System
 * Lightweight, client-side forum with localStorage persistence
 *
 * Features:
 * - Thread CRUD (Create, Read, Update, Delete)
 * - Reply threading
 * - Voting system (upvote/downvote)
 * - Search and filtering
 * - Hot score calculation
 * - User reputation system
 * - Markdown support
 *
 * Data persisted to localStorage under key 'hololoom_forums'
 */

const ForumManager = {
    // Storage key
    STORAGE_KEY: 'hololoom_forums',

    // Initialize default categories and load data
    init() {
        const existing = this.loadData();

        if (!existing || !existing.categories || existing.categories.length === 0) {
            this.saveData(this.getDefaultData());
        }
    },

    // Get default forum structure
    getDefaultData() {
        return {
            categories: [
                {
                    id: 'general',
                    name: 'General Discussion',
                    description: 'General topics and community chat',
                    threadCount: 0,
                    postCount: 0
                },
                {
                    id: 'hololoom',
                    name: 'HoloLoom Help',
                    description: 'Questions and discussions about HoloLoom features',
                    threadCount: 0,
                    postCount: 0
                },
                {
                    id: 'promptly',
                    name: 'Promptly',
                    description: 'Discuss prompt engineering and optimization',
                    threadCount: 0,
                    postCount: 0
                },
                {
                    id: 'development',
                    name: 'Development',
                    description: 'Technical development, API usage, and contributions',
                    threadCount: 0,
                    postCount: 0
                },
                {
                    id: 'announcements',
                    name: 'Announcements',
                    description: 'Important updates and announcements from the team',
                    threadCount: 0,
                    postCount: 0
                }
            ],
            threads: [
                // Example threads for demo
                {
                    id: 1,
                    categoryId: 'hololoom',
                    title: 'Welcome to the HoloLoom Community Forum',
                    content: '# Welcome!\n\nThis is the community forum for HoloLoom. Feel free to:\n\n- Ask questions about how to use HoloLoom\n- Share your experiences and tips\n- Report bugs and suggest features\n- Network with other users\n\nLet\'s build something great together!',
                    author: 'HoloLoom Team',
                    created: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
                    updated: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
                    views: 42,
                    type: 'announcement',
                    solved: false,
                    tags: ['welcome', 'getting-started'],
                    replies: [
                        {
                            id: 1,
                            threadId: 1,
                            content: 'Great to see this community growing! Looking forward to learning from everyone here.',
                            author: 'Jane Developer',
                            created: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
                            upvotes: 3,
                            downvotes: 0,
                            solution: false
                        },
                        {
                            id: 2,
                            threadId: 1,
                            content: 'Thanks for setting this up! Very excited about the HoloLoom ecosystem.',
                            author: 'Alex Researcher',
                            created: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
                            upvotes: 2,
                            downvotes: 0,
                            solution: false
                        }
                    ],
                    upvotes: 8,
                    downvotes: 0
                }
            ],
            nextThreadId: 2,
            nextReplyId: 3
        };
    },

    // ============ Storage Operations ============

    loadData() {
        const data = localStorage.getItem(this.STORAGE_KEY);
        return data ? JSON.parse(data) : null;
    },

    saveData(data) {
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
        this.updateCategoryStats();
    },

    // ============ Category Operations ============

    getCategories() {
        return this.loadData().categories;
    },

    // ============ Thread Operations (CRUD) ============

    /**
     * Get all threads
     */
    getAllThreads() {
        return this.loadData().threads;
    },

    /**
     * Get thread by ID
     */
    getThread(threadId) {
        return this.getAllThreads().find(t => t.id === threadId);
    },

    /**
     * Get threads by category
     */
    getThreadsByCategory(categoryId) {
        return this.getAllThreads().filter(t => t.categoryId === categoryId);
    },

    /**
     * Create new thread
     */
    createThread(threadData) {
        const data = this.loadData();
        const threadId = data.nextThreadId++;

        const thread = {
            id: threadId,
            categoryId: threadData.categoryId,
            title: threadData.title,
            content: threadData.content,
            author: threadData.author,
            created: new Date().toISOString(),
            updated: new Date().toISOString(),
            views: 0,
            type: threadData.type || 'discussion',
            solved: false,
            tags: threadData.tags || [],
            replies: [],
            upvotes: 0,
            downvotes: 0
        };

        data.threads.push(thread);
        this.saveData(data);

        return thread;
    },

    /**
     * Update thread
     */
    updateThread(threadId, updates) {
        const data = this.loadData();
        const thread = data.threads.find(t => t.id === threadId);

        if (thread) {
            Object.assign(thread, updates, {
                updated: new Date().toISOString()
            });
            this.saveData(data);
        }

        return thread;
    },

    /**
     * Delete thread
     */
    deleteThread(threadId) {
        const data = this.loadData();
        data.threads = data.threads.filter(t => t.id !== threadId);
        this.saveData(data);
    },

    /**
     * Increment view count
     */
    incrementViewCount(threadId) {
        const thread = this.getThread(threadId);
        if (thread) {
            thread.views++;
            this.updateThread(threadId, { views: thread.views });
        }
    },

    /**
     * Mark thread as solved (for questions)
     */
    markThreadAsSolved(threadId) {
        return this.updateThread(threadId, { solved: true });
    },

    /**
     * Mark reply as solution
     */
    markAsSolution(threadId, replyId) {
        const thread = this.getThread(threadId);
        if (thread) {
            // Clear previous solutions
            thread.replies.forEach(r => r.solution = false);

            // Set new solution
            const reply = thread.replies.find(r => r.id === replyId);
            if (reply) {
                reply.solution = true;
                this.markThreadAsSolved(threadId);
            }

            this.updateThread(threadId, { replies: thread.replies });
        }
    },

    // ============ Reply Operations (CRUD) ============

    /**
     * Create reply
     */
    createReply(threadId, replyData) {
        const data = this.loadData();
        const thread = data.threads.find(t => t.id === threadId);

        if (!thread) {
            return null;
        }

        const replyId = data.nextReplyId++;

        const reply = {
            id: replyId,
            threadId: threadId,
            content: replyData.content,
            author: replyData.author,
            created: new Date().toISOString(),
            upvotes: 0,
            downvotes: 0,
            solution: false
        };

        thread.replies.push(reply);
        this.saveData(data);

        return reply;
    },

    /**
     * Delete reply
     */
    deleteReply(threadId, replyId) {
        const thread = this.getThread(threadId);
        if (thread) {
            thread.replies = thread.replies.filter(r => r.id !== replyId);
            this.updateThread(threadId, { replies: thread.replies });
        }
    },

    // ============ Voting Operations ============

    /**
     * Upvote thread
     */
    upvoteThread(threadId) {
        const thread = this.getThread(threadId);
        if (thread) {
            thread.upvotes++;
            this.updateThread(threadId, { upvotes: thread.upvotes });
        }
    },

    /**
     * Downvote thread
     */
    downvoteThread(threadId) {
        const thread = this.getThread(threadId);
        if (thread) {
            thread.downvotes++;
            this.updateThread(threadId, { downvotes: thread.downvotes });
        }
    },

    /**
     * Upvote reply
     */
    upvoteReply(threadId, replyId) {
        const thread = this.getThread(threadId);
        if (thread) {
            const reply = thread.replies.find(r => r.id === replyId);
            if (reply) {
                reply.upvotes++;
                this.updateThread(threadId, { replies: thread.replies });
            }
        }
    },

    /**
     * Downvote reply
     */
    downvoteReply(threadId, replyId) {
        const thread = this.getThread(threadId);
        if (thread) {
            const reply = thread.replies.find(r => r.id === replyId);
            if (reply) {
                reply.downvotes++;
                this.updateThread(threadId, { replies: thread.replies });
            }
        }
    },

    // ============ Search and Filter ============

    /**
     * Search threads and replies
     */
    searchThreads(query) {
        const lowerQuery = query.toLowerCase();
        const threads = this.getAllThreads();

        return threads.filter(thread =>
            thread.title.toLowerCase().includes(lowerQuery) ||
            thread.content.toLowerCase().includes(lowerQuery) ||
            thread.tags.some(tag => tag.toLowerCase().includes(lowerQuery)) ||
            thread.replies.some(reply =>
                reply.content.toLowerCase().includes(lowerQuery)
            )
        );
    },

    /**
     * Filter threads by tag
     */
    getThreadsByTag(tag) {
        return this.getAllThreads().filter(t =>
            t.tags.some(tagItem => tagItem.toLowerCase() === tag.toLowerCase())
        );
    },

    /**
     * Get unanswered questions
     */
    getUnansweredQuestions(categoryId) {
        let threads = this.getThreadsByCategory(categoryId);
        return threads.filter(t => t.type === 'question' && t.replies.length === 0);
    },

    // ============ Sorting and Hot Score ============

    /**
     * Calculate hot score based on votes and recency
     */
    calculateHotScore(thread) {
        const ageInHours = (Date.now() - new Date(thread.created).getTime()) / (1000 * 60 * 60);
        const votes = thread.upvotes - thread.downvotes;
        const replies = thread.replies.length;

        // Hot score: votes (40%) + replies (30%) + recency (30%)
        // Recency decays exponentially (half-life of 24 hours)
        const recencyScore = Math.exp(-ageInHours / 24);
        const hotScore = (votes * 0.4) + (replies * 0.3 * Math.min(1, replies / 10)) + (recencyScore * 0.3 * 100);

        return hotScore;
    },

    /**
     * Sort threads by hot score
     */
    sortByHotScore(threads) {
        return [...threads].sort((a, b) =>
            this.calculateHotScore(b) - this.calculateHotScore(a)
        );
    },

    // ============ Activity and Statistics ============

    /**
     * Get recent activity
     */
    getRecentActivity(limit = 10) {
        const threads = this.getAllThreads();
        return threads
            .sort((a, b) => {
                const aTime = a.replies.length > 0
                    ? new Date(a.replies[a.replies.length - 1].created)
                    : new Date(a.created);
                const bTime = b.replies.length > 0
                    ? new Date(b.replies[b.replies.length - 1].created)
                    : new Date(b.created);
                return bTime - aTime;
            })
            .slice(0, limit);
    },

    /**
     * Get statistics
     */
    getStatistics() {
        const threads = this.getAllThreads();
        const totalReplies = threads.reduce((sum, t) => sum + t.replies.length, 0);
        const users = new Set();

        threads.forEach(t => {
            users.add(t.author);
            t.replies.forEach(r => users.add(r.author));
        });

        // Count threads created today
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        const todayThreads = threads.filter(t => new Date(t.created) >= today).length;

        return {
            threadCount: threads.length,
            replyCount: totalReplies,
            userCount: users.size,
            todayThreadCount: todayThreads
        };
    },

    /**
     * Get top contributors
     */
    getTopContributors(limit = 8) {
        const contributors = {};

        this.getAllThreads().forEach(thread => {
            if (!contributors[thread.author]) {
                contributors[thread.author] = {
                    name: thread.author,
                    postCount: 0,
                    reputation: 0
                };
            }
            contributors[thread.author].postCount++;
            contributors[thread.author].reputation += (thread.upvotes - thread.downvotes);

            thread.replies.forEach(reply => {
                if (!contributors[reply.author]) {
                    contributors[reply.author] = {
                        name: reply.author,
                        postCount: 0,
                        reputation: 0
                    };
                }
                contributors[reply.author].postCount++;
                contributors[reply.author].reputation += (reply.upvotes - reply.downvotes);
            });
        });

        return Object.values(contributors)
            .sort((a, b) => b.reputation - a.reputation)
            .slice(0, limit);
    },

    /**
     * Get user reputation (total upvotes on all their posts)
     */
    getUserReputation(userName) {
        let reputation = 0;

        this.getAllThreads().forEach(thread => {
            if (thread.author === userName) {
                reputation += (thread.upvotes - thread.downvotes);
            }
            thread.replies.forEach(reply => {
                if (reply.author === userName) {
                    reputation += (reply.upvotes - reply.downvotes);
                }
            });
        });

        return Math.max(0, reputation);
    },

    /**
     * Get user's threads and replies
     */
    getUserPosts(userName) {
        const threads = this.getAllThreads()
            .filter(t => t.author === userName);

        const replies = [];
        this.getAllThreads().forEach(thread => {
            thread.replies
                .filter(r => r.author === userName)
                .forEach(reply => {
                    replies.push({
                        ...reply,
                        threadId: thread.id,
                        threadTitle: thread.title
                    });
                });
        });

        return { threads, replies };
    },

    /**
     * Update category statistics
     */
    updateCategoryStats() {
        const data = this.loadData();

        data.categories.forEach(category => {
            const threads = data.threads.filter(t => t.categoryId === category.id);
            category.threadCount = threads.length;
            category.postCount = threads.reduce((sum, t) => sum + t.replies.length + 1, 0);
        });

        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
    },

    // ============ Data Export/Import ============

    /**
     * Export forum data to JSON
     */
    exportData() {
        const data = this.loadData();
        return JSON.stringify(data, null, 2);
    },

    /**
     * Import forum data from JSON
     */
    importData(jsonString) {
        try {
            const data = JSON.parse(jsonString);
            if (data.categories && data.threads) {
                this.saveData(data);
                return true;
            }
            return false;
        } catch (e) {
            console.error('Import failed:', e);
            return false;
        }
    },

    /**
     * Reset to default data (for demo/testing)
     */
    reset() {
        this.saveData(this.getDefaultData());
    },

    /**
     * Clear all data
     */
    clearAll() {
        localStorage.removeItem(this.STORAGE_KEY);
    }
};

// Initialize on script load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        ForumManager.init();
    });
} else {
    ForumManager.init();
}
