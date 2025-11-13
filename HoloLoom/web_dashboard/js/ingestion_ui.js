/**
 * Data Ingestion UI
 * ==================
 * No-code data loading interface for HoloLoom.
 *
 * Features:
 * - YouTube video ingestion (URL paste-and-process)
 * - File upload interface (text, PDF, etc.)
 * - Web URL scraping
 * - Batch ingestion queue monitoring
 * - Progress tracking
 *
 * Philosophy: Framework → Elegance → Remove Code Barrier
 */

class IngestionUI {
    constructor(apiBase = 'http://localhost:8000') {
        this.apiBase = apiBase;
        this.updateInterval = 2000; // 2 seconds (frequent for progress updates)
        this.intervalId = null;
        this.activeJobs = new Map(); // jobId → job data
    }

    /**
     * Initialize ingestion UI
     */
    async initialize() {
        console.log('Initializing Ingestion UI...');

        // Initial queue load
        await this.refreshQueue();

        // Start polling
        this.intervalId = setInterval(() => this.refreshQueue(), this.updateInterval);

        // Setup event listeners
        this.setupEventListeners();

        console.log('✓ Ingestion UI initialized');
    }

    /**
     * Stop polling and cleanup
     */
    destroy() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // YouTube ingestion
        const youtubeBtn = document.getElementById('youtube-ingest-btn');
        if (youtubeBtn) {
            youtubeBtn.addEventListener('click', () => this.handleYouTubeIngest());
        }

        // File upload
        const fileInput = document.getElementById('file-upload-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        // Web URL ingestion
        const webBtn = document.getElementById('web-ingest-btn');
        if (webBtn) {
            webBtn.addEventListener('click', () => this.handleWebIngest());
        }

        // Clear completed button
        const clearBtn = document.getElementById('queue-clear-completed-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearCompletedJobs());
        }
    }

    /**
     * Handle YouTube ingestion
     */
    async handleYouTubeIngest() {
        const input = document.getElementById('youtube-url-input');
        if (!input) return;

        const url = input.value.trim();
        if (!url) {
            this.showError('Please enter a YouTube URL');
            return;
        }

        // Extract video ID
        const videoId = this.extractYouTubeVideoId(url);
        if (!videoId) {
            this.showError('Invalid YouTube URL');
            return;
        }

        try {
            // Show loading
            const btn = document.getElementById('youtube-ingest-btn');
            const originalText = btn.textContent;
            btn.textContent = 'Processing...';
            btn.disabled = true;

            const response = await fetch(`${this.apiBase}/ingestion/youtube`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    url: videoId,
                    chunk_duration: 60.0,
                    languages: ['en']
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Ingestion failed');
            }

            const result = await response.json();
            console.log('YouTube ingestion started:', result);

            // Clear input
            input.value = '';

            // Show success
            this.showSuccess(`YouTube ingestion started (Job ID: ${result.job_id})`);

            // Refresh queue
            await this.refreshQueue();

            // Restore button
            btn.textContent = originalText;
            btn.disabled = false;
        } catch (error) {
            console.error('YouTube ingestion failed:', error);
            this.showError(error.message);

            // Restore button
            const btn = document.getElementById('youtube-ingest-btn');
            btn.textContent = 'Ingest';
            btn.disabled = false;
        }
    }

    /**
     * Handle file upload
     */
    async handleFileUpload(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        const file = files[0];

        try {
            // TODO: Implement file upload API endpoint
            console.log('File upload:', file.name, file.size, file.type);

            this.showWarning('File upload not yet implemented (coming in Phase 3)');

            // Clear input
            event.target.value = '';
        } catch (error) {
            console.error('File upload failed:', error);
            this.showError(error.message);
        }
    }

    /**
     * Handle web URL ingestion
     */
    async handleWebIngest() {
        const input = document.getElementById('web-url-input');
        if (!input) return;

        const url = input.value.trim();
        if (!url) {
            this.showError('Please enter a URL');
            return;
        }

        try {
            // TODO: Implement web ingestion API endpoint
            console.log('Web ingestion:', url);

            this.showWarning('Web URL ingestion not yet implemented (coming in Phase 3)');

            // Clear input
            input.value = '';
        } catch (error) {
            console.error('Web ingestion failed:', error);
            this.showError(error.message);
        }
    }

    /**
     * Refresh ingestion queue
     */
    async refreshQueue() {
        try {
            const response = await fetch(`${this.apiBase}/ingestion/status`);

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const data = await response.json();
            this.updateQueue(data.queue || []);
        } catch (error) {
            console.error('Failed to refresh queue:', error);
            // Don't show error for queue refresh (non-critical)
        }
    }

    /**
     * Update queue display
     */
    updateQueue(jobs) {
        const container = document.getElementById('ingestion-queue-container');
        if (!container) return;

        // Update active jobs map
        jobs.forEach(job => {
            this.activeJobs.set(job.job_id, job);
        });

        if (jobs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V7h14v12zm-8-2h2v-4h4v-2h-4V7h-2v4H7v2h4z"/>
                    </svg>
                    <div>No ingestion jobs in queue</div>
                    <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--secondary);">
                        Start by ingesting a YouTube video above
                    </div>
                </div>
            `;
            return;
        }

        // Render jobs
        let html = '<div class="ingestion-jobs">';

        jobs.forEach(job => {
            html += this.renderJob(job);
        });

        html += '</div>';
        container.innerHTML = html;

        // Update total count
        const countEl = document.getElementById('queue-job-count');
        if (countEl) {
            countEl.textContent = `${jobs.length} job${jobs.length !== 1 ? 's' : ''}`;
        }
    }

    /**
     * Render job card
     */
    renderJob(job) {
        const statusClass = job.status === 'completed' ? 'success'
            : job.status === 'failed' ? 'danger'
            : job.status === 'processing' ? 'info'
            : 'secondary';

        const timestamp = new Date(job.timestamp * 1000).toLocaleString();

        return `
            <div class="job-card">
                <div class="job-header">
                    <div class="job-type"><strong>${job.type.toUpperCase()}</strong></div>
                    <span class="badge ${statusClass}">${job.status}</span>
                </div>
                <div class="job-url">${this.truncate(job.url || job.file || job.source || 'Unknown', 50)}</div>
                <div class="job-footer">
                    <span style="font-size: 0.75rem; color: var(--secondary);">${timestamp}</span>
                    <span style="font-size: 0.75rem; color: var(--secondary);">ID: ${job.job_id}</span>
                </div>
            </div>
        `;
    }

    /**
     * Clear completed jobs
     */
    clearCompletedJobs() {
        // Filter out completed jobs from active jobs map
        for (const [jobId, job] of this.activeJobs.entries()) {
            if (job.status === 'completed' || job.status === 'failed') {
                this.activeJobs.delete(jobId);
            }
        }

        // Update display
        this.updateQueue(Array.from(this.activeJobs.values()));
    }

    /**
     * Extract YouTube video ID from URL
     */
    extractYouTubeVideoId(url) {
        // Handle various YouTube URL formats
        const patterns = [
            /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\s?]+)/,
            /^([a-zA-Z0-9_-]{11})$/ // Direct video ID
        ];

        for (const pattern of patterns) {
            const match = url.match(pattern);
            if (match && match[1]) {
                return match[1];
            }
        }

        return null;
    }

    /**
     * Show success message
     */
    showSuccess(message) {
        const container = document.getElementById('ingestion-message-container');
        if (container) {
            container.innerHTML = `
                <div class="success-message">
                    ${this.escapeHtml(message)}
                </div>
            `;
            container.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                container.style.display = 'none';
            }, 5000);
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        const container = document.getElementById('ingestion-message-container');
        if (container) {
            container.innerHTML = `
                <div class="error">
                    <div class="error-title">Error</div>
                    <div>${this.escapeHtml(message)}</div>
                </div>
            `;
            container.style.display = 'block';

            // Auto-hide after 10 seconds
            setTimeout(() => {
                container.style.display = 'none';
            }, 10000);
        }
    }

    /**
     * Show warning message
     */
    showWarning(message) {
        const container = document.getElementById('ingestion-message-container');
        if (container) {
            container.innerHTML = `
                <div class="warning-box">
                    <strong>⚠ ${this.escapeHtml(message)}</strong>
                </div>
            `;
            container.style.display = 'block';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                container.style.display = 'none';
            }, 5000);
        }
    }

    /**
     * Truncate text
     */
    truncate(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Export for use in control panel
window.IngestionUI = IngestionUI;
