/**
 * WORKFLOW GALLERY - JavaScript
 * ==============================
 * Client-side functionality for workflow discovery and deployment
 *
 * Features:
 * - Real-time search
 * - Multi-dimension filtering
 * - Dynamic sorting
 * - One-click deployment
 * - Modal interactions
 *
 * Author: HoloLoom Team
 * Date: November 17, 2025
 */

// ============================================
// SAMPLE WORKFLOW DATA
// ============================================

const SAMPLE_WORKFLOWS = [
    {
        id: "inbox-triage",
        name: "Inbox Triage",
        category: "Email & Communication",
        categorySlug: "email",
        author: "@hololoom",
        icon: "📧",
        rating: 4.8,
        reviewCount: 1234,
        deployments: 10543,
        successRate: 0.95,
        difficulty: "beginner",
        timeSaved: "2 hours/day",
        timeSavedMinutes: 120,
        description: "Automatically triage emails, draft responses, and route urgent messages to Slack. Handles 200+ emails/day with 95% accuracy.",
        longDescription: "The Inbox Triage workflow transforms email management from a time sink into a seamless automated process. Using advanced NLP and sentiment analysis, it classifies emails into urgent, respond, archive, or spam categories. For routine emails, it drafts context-aware responses that maintain your writing style. Urgent emails trigger immediate Slack notifications with priority context.",
        testimonial: "Changed my life! I actually look forward to checking email now.",
        testimonialAuthor: "@sarah",
        tags: ["email", "productivity", "automation", "nlp"],
        featured: true,
        addedDate: "2025-11-01",
        diagram: [
            {type: "fetch", label: "📥", name: "Gmail"},
            {type: "classify", label: "🏷️", name: "Classify"},
            {type: "route", label: "🔀", name: "Route"},
            {type: "output", label: "📤", name: "Deliver"}
        ],
        requiredIntegrations: ["gmail", "slack"],
        setupSteps: [
            "Configure Gmail API credentials",
            "Set up Slack webhook URL",
            "Customize classification rules",
            "Test with sample emails"
        ],
        metrics: {
            avgExecutionTime: "2 minutes",
            costPerRun: "$0.03",
            emailsPerDay: 200
        }
    },
    {
        id: "meeting-summary",
        name: "Meeting Summarization",
        category: "Email & Communication",
        categorySlug: "email",
        author: "@productivitypro",
        icon: "📝",
        rating: 4.9,
        reviewCount: 987,
        deployments: 8234,
        successRate: 0.97,
        difficulty: "beginner",
        timeSaved: "30 min/meeting",
        timeSavedMinutes: 30,
        description: "Transcribe meetings, extract action items, and distribute notes automatically. Never take notes again.",
        longDescription: "This workflow revolutionizes meeting productivity. It connects to Zoom, Google Meet, or Teams to automatically record and transcribe meetings. Advanced NLP extracts key discussion points, decisions made, and action items with assigned owners. The summary is automatically sent to attendees via email and Slack within minutes of the meeting ending.",
        testimonial: "Saves me 30 minutes every meeting. The action items are spot-on!",
        testimonialAuthor: "@michael",
        tags: ["meetings", "transcription", "productivity"],
        featured: true,
        addedDate: "2025-11-05",
        diagram: [
            {type: "record", label: "🎙️", name: "Record"},
            {type: "transcribe", label: "📄", name: "Transcribe"},
            {type: "extract", label: "🎯", name: "Extract"},
            {type: "distribute", label: "📨", name: "Send"}
        ],
        requiredIntegrations: ["zoom", "slack", "gmail"],
        setupSteps: [
            "Connect to Zoom/Meet/Teams",
            "Configure attendee list sources",
            "Set up distribution channels",
            "Test with a sample meeting"
        ],
        metrics: {
            avgExecutionTime: "2-3 minutes",
            costPerRun: "$0.05",
            meetingsPerWeek: 10
        }
    },
    {
        id: "bug-triage",
        name: "Bug Triage Automation",
        category: "Developer Tools",
        categorySlug: "dev",
        author: "@techcorp",
        icon: "🐛",
        rating: 4.8,
        reviewCount: 654,
        deployments: 6789,
        successRate: 0.93,
        difficulty: "intermediate",
        timeSaved: "30 min/bug",
        timeSavedMinutes: 30,
        description: "Auto-classify bug severity, assign to teams, and notify stakeholders. Handles 500+ bugs/month.",
        longDescription: "Bug Triage Automation eliminates the bottleneck of manual bug classification. It analyzes bug reports using ML to determine severity (critical/high/medium/low), identifies the relevant team based on component analysis, and routes bugs to the right developers. Critical bugs trigger immediate PagerDuty alerts while lower-priority bugs are batched for daily review.",
        testimonial: "Our best productivity investment in 5 years. Saves 233 hours/month.",
        testimonialAuthor: "@techcorpcto",
        tags: ["bugs", "automation", "devops", "triage"],
        featured: true,
        addedDate: "2025-11-03",
        diagram: [
            {type: "fetch", label: "📥", name: "Fetch"},
            {type: "analyze", label: "🔍", name: "Analyze"},
            {type: "classify", label: "🏷️", name: "Classify"},
            {type: "assign", label: "👤", name: "Assign"},
            {type: "notify", label: "🔔", name: "Alert"}
        ],
        requiredIntegrations: ["github", "jira", "pagerduty"],
        setupSteps: [
            "Connect to GitHub/Jira",
            "Configure team routing rules",
            "Set severity thresholds",
            "Set up notification channels"
        ],
        metrics: {
            avgExecutionTime: "30 seconds",
            costPerRun: "$0.02",
            bugsPerMonth: 500
        }
    },
    {
        id: "report-generator",
        name: "Weekly Report Generator",
        category: "Data & Analytics",
        categorySlug: "data",
        author: "@dataanalyst",
        icon: "📊",
        rating: 4.7,
        reviewCount: 543,
        deployments: 5123,
        successRate: 0.91,
        difficulty: "intermediate",
        timeSaved: "4 hours/week",
        timeSavedMinutes: 240,
        description: "Pull data from multiple sources, analyze trends, generate visualizations, and create PDF reports automatically.",
        longDescription: "Transform data reporting from a weekly chore into an automated process. This workflow connects to your databases, APIs, and spreadsheets to pull the latest metrics. It performs trend analysis, generates charts and visualizations, and compiles everything into a professional PDF report. The report is automatically distributed to stakeholders every Monday morning.",
        testimonial: "Fridays used to be report day. Now I just review and send. 4 hours back!",
        testimonialAuthor: "@analytics_lead",
        tags: ["reports", "analytics", "automation", "data"],
        featured: false,
        addedDate: "2025-11-07",
        diagram: [
            {type: "fetch", label: "📥", name: "Fetch Data"},
            {type: "analyze", label: "📈", name: "Analyze"},
            {type: "visualize", label: "📊", name: "Chart"},
            {type: "generate", label: "📄", name: "PDF"},
            {type: "distribute", label: "📨", name: "Send"}
        ],
        requiredIntegrations: ["postgres", "google_sheets", "gmail"],
        setupSteps: [
            "Configure data source connections",
            "Define metrics and KPIs",
            "Customize report template",
            "Set distribution schedule"
        ],
        metrics: {
            avgExecutionTime: "5 minutes",
            costPerRun: "$0.10",
            reportsPerMonth: 4
        }
    },
    {
        id: "competitive-intel",
        name: "Competitive Intelligence Monitor",
        category: "Data & Analytics",
        categorySlug: "data",
        author: "@marketintel",
        icon: "🔍",
        rating: 4.9,
        reviewCount: 432,
        deployments: 4567,
        successRate: 0.96,
        difficulty: "advanced",
        timeSaved: "10 hours/week",
        timeSavedMinutes: 600,
        description: "Monitor competitor websites 24/7, track pricing changes, feature updates, and news mentions. Get instant alerts.",
        longDescription: "Stay ahead of the competition with automated intelligence gathering. This workflow continuously monitors competitor websites, pricing pages, product announcements, and news mentions. It uses computer vision to detect visual changes and NLP to analyze content updates. When significant changes are detected, you get immediate Slack alerts with detailed change summaries.",
        testimonial: "Never miss a competitor move. Worth every penny!",
        testimonialAuthor: "@strategy_vp",
        tags: ["competitive", "monitoring", "intelligence", "automation"],
        featured: false,
        addedDate: "2025-11-10",
        diagram: [
            {type: "scrape", label: "🕷️", name: "Scrape"},
            {type: "analyze", label: "🔍", name: "Analyze"},
            {type: "detect", label: "🎯", name: "Detect"},
            {type: "alert", label: "🔔", name: "Alert"}
        ],
        requiredIntegrations: ["apify", "slack", "airtable"],
        setupSteps: [
            "Add competitor URLs to monitor",
            "Configure change detection thresholds",
            "Set up alert channels",
            "Test with sample changes"
        ],
        metrics: {
            avgExecutionTime: "10 minutes (hourly)",
            costPerRun: "$0.15",
            checksPerDay: 24
        }
    },
    {
        id: "code-review",
        name: "Code Review Assistant",
        category: "Developer Tools",
        categorySlug: "dev",
        author: "@devtools",
        icon: "👨‍💻",
        rating: 4.6,
        reviewCount: 389,
        deployments: 4321,
        successRate: 0.89,
        difficulty: "intermediate",
        timeSaved: "20 min/PR",
        timeSavedMinutes: 20,
        description: "Automatically review pull requests, check code style, suggest improvements, and comment inline.",
        longDescription: "Accelerate code review cycles with AI-powered analysis. This workflow automatically reviews every pull request, checking for common issues like security vulnerabilities, style violations, performance anti-patterns, and missing test coverage. It posts detailed inline comments with improvement suggestions and assigns a review score. Human reviewers can focus on architecture and business logic.",
        testimonial: "Catches 80% of issues before human review. Massive time saver!",
        testimonialAuthor: "@senior_dev",
        tags: ["code-review", "automation", "quality", "devops"],
        featured: false,
        addedDate: "2025-11-12",
        diagram: [
            {type: "fetch", label: "📥", name: "PR"},
            {type: "analyze", label: "🔍", name: "Analyze"},
            {type: "check", label: "✓", name: "Check"},
            {type: "comment", label: "💬", name: "Comment"}
        ],
        requiredIntegrations: ["github", "sonarqube"],
        setupSteps: [
            "Connect to GitHub repository",
            "Configure code quality rules",
            "Set review thresholds",
            "Test with sample PR"
        ],
        metrics: {
            avgExecutionTime: "2 minutes",
            costPerRun: "$0.04",
            prsPerWeek: 30
        }
    },
    {
        id: "social-scheduler",
        name: "Social Media Scheduler",
        category: "Content Creation",
        categorySlug: "content",
        author: "@contentpro",
        icon: "📱",
        rating: 4.8,
        reviewCount: 678,
        deployments: 3456,
        successRate: 0.94,
        difficulty: "beginner",
        timeSaved: "5 hours/week",
        timeSavedMinutes: 300,
        description: "Generate social media posts, optimize timing, and schedule across Twitter, LinkedIn, and Facebook.",
        longDescription: "Transform your social media presence with intelligent automation. This workflow generates engaging posts based on your content calendar, optimizes posting times based on audience analytics, and schedules content across multiple platforms. It can repurpose blog posts, create thread variations, and maintain consistent brand voice across channels.",
        testimonial: "10x our reach with consistent posting. Growth has been incredible!",
        testimonialAuthor: "@marketing_mgr",
        tags: ["social-media", "content", "marketing", "automation"],
        featured: false,
        addedDate: "2025-11-08",
        diagram: [
            {type: "generate", label: "✍️", name: "Generate"},
            {type: "optimize", label: "🎯", name: "Optimize"},
            {type: "schedule", label: "📅", name: "Schedule"},
            {type: "post", label: "📤", name: "Post"}
        ],
        requiredIntegrations: ["twitter", "linkedin", "facebook", "buffer"],
        setupSteps: [
            "Connect social media accounts",
            "Set posting schedule preferences",
            "Configure content templates",
            "Test with sample posts"
        ],
        metrics: {
            avgExecutionTime: "30 seconds per post",
            costPerRun: "$0.02",
            postsPerWeek: 20
        }
    },
    {
        id: "doc-generator",
        name: "Documentation Generator",
        category: "Developer Tools",
        categorySlug: "dev",
        icon: "📚",
        author: "@docwriter",
        rating: 4.7,
        reviewCount: 298,
        deployments: 2987,
        successRate: 0.92,
        difficulty: "intermediate",
        timeSaved: "4 hours/release",
        timeSavedMinutes: 240,
        description: "Parse code, extract APIs, generate documentation, and publish to your docs site automatically.",
        longDescription: "Keep documentation in perfect sync with your codebase. This workflow analyzes your source code to extract API signatures, parameters, return types, and inline comments. It generates beautiful, searchable documentation in Markdown or HTML format and publishes it to your docs site. Documentation is automatically updated with every release.",
        testimonial: "Docs are always up-to-date now. Developers actually read them!",
        testimonialAuthor: "@tech_lead",
        tags: ["documentation", "automation", "devops"],
        featured: false,
        addedDate: "2025-11-06",
        diagram: [
            {type: "parse", label: "📖", name: "Parse"},
            {type: "extract", label: "🔍", name: "Extract"},
            {type: "generate", label: "📄", name: "Generate"},
            {type: "publish", label: "🌐", name: "Publish"}
        ],
        requiredIntegrations: ["github", "readthedocs"],
        setupSteps: [
            "Connect to code repository",
            "Configure documentation style",
            "Set up publishing target",
            "Test with sample code"
        ],
        metrics: {
            avgExecutionTime: "3 minutes",
            costPerRun: "$0.05",
            releasesPerMonth: 4
        }
    },
    {
        id: "customer-support",
        name: "Customer Support Automation",
        category: "Email & Communication",
        categorySlug: "email",
        author: "@supportdesk",
        icon: "💬",
        rating: 4.5,
        reviewCount: 567,
        deployments: 2876,
        successRate: 0.88,
        difficulty: "intermediate",
        timeSaved: "3 hours/day",
        timeSavedMinutes: 180,
        description: "Classify support tickets, draft responses for common issues, and escalate complex cases automatically.",
        longDescription: "Handle 70% of support tickets automatically while maintaining quality. This workflow categorizes incoming tickets by issue type, searches your knowledge base for relevant solutions, and drafts personalized responses. Simple issues are resolved automatically while complex cases are escalated to human agents with full context and suggested solutions.",
        testimonial: "Reduced response time from 4 hours to 10 minutes. Customers love it!",
        testimonialAuthor: "@support_lead",
        tags: ["support", "automation", "customer-service"],
        featured: false,
        addedDate: "2025-11-09",
        diagram: [
            {type: "fetch", label: "📥", name: "Tickets"},
            {type: "classify", label: "🏷️", name: "Classify"},
            {type: "search", label: "🔍", name: "KB Search"},
            {type: "draft", label: "✍️", name: "Draft"},
            {type: "route", label: "🔀", name: "Route"}
        ],
        requiredIntegrations: ["zendesk", "intercom"],
        setupSteps: [
            "Connect to support platform",
            "Import knowledge base",
            "Configure routing rules",
            "Test with sample tickets"
        ],
        metrics: {
            avgExecutionTime: "45 seconds",
            costPerRun: "$0.03",
            ticketsPerDay: 100
        }
    },
    {
        id: "translation-pipeline",
        name: "Translation Pipeline",
        category: "Content Creation",
        categorySlug: "content",
        author: "@globalcontent",
        icon: "🌍",
        rating: 4.6,
        reviewCount: 234,
        deployments: 2134,
        successRate: 0.90,
        difficulty: "beginner",
        timeSaved: "2 hours/document",
        timeSavedMinutes: 120,
        description: "Translate documents into multiple languages, maintain formatting, and validate translations automatically.",
        longDescription: "Expand your global reach with professional-quality translations. This workflow detects the source language, translates content into your target languages using neural machine translation, preserves document formatting, and validates translations for accuracy. It handles documents, marketing materials, and website content with cultural adaptation.",
        testimonial: "Saved $500 per document vs. professional translation. Quality is excellent!",
        testimonialAuthor: "@global_marketing",
        tags: ["translation", "internationalization", "content"],
        featured: false,
        addedDate: "2025-11-11",
        diagram: [
            {type: "detect", label: "🔍", name: "Detect"},
            {type: "translate", label: "🌐", name: "Translate"},
            {type: "validate", label: "✓", name: "Validate"},
            {type: "format", label: "📄", name: "Format"}
        ],
        requiredIntegrations: ["deepl", "google_translate"],
        setupSteps: [
            "Select source and target languages",
            "Configure translation API",
            "Set quality thresholds",
            "Test with sample document"
        ],
        metrics: {
            avgExecutionTime: "1 minute per 1000 words",
            costPerRun: "$0.20",
            documentsPerWeek: 5
        }
    }
];

// ============================================
// STATE MANAGEMENT
// ============================================

let state = {
    workflows: [...SAMPLE_WORKFLOWS],
    filteredWorkflows: [...SAMPLE_WORKFLOWS],
    selectedCategory: 'all',
    selectedDifficulty: 'all',
    sortBy: 'popular',
    searchQuery: ''
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    renderWorkflows();
});

function initializeEventListeners() {
    // Search
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', handleSearch);

    // Category filters
    const categoryBtns = document.querySelectorAll('[data-category]');
    categoryBtns.forEach(btn => {
        btn.addEventListener('click', () => handleCategoryFilter(btn.dataset.category));
    });

    // Difficulty filters
    const difficultyBtns = document.querySelectorAll('[data-difficulty]');
    difficultyBtns.forEach(btn => {
        btn.addEventListener('click', () => handleDifficultyFilter(btn.dataset.difficulty));
    });

    // Sort
    const sortSelect = document.getElementById('sortSelect');
    sortSelect.addEventListener('change', (e) => handleSort(e.target.value));

    // Close modals on background click
    document.getElementById('detailModal').addEventListener('click', (e) => {
        if (e.target.id === 'detailModal') closeDetailModal();
    });
    document.getElementById('deployModal').addEventListener('click', (e) => {
        if (e.target.id === 'deployModal') closeDeployModal();
    });
}

// ============================================
// FILTERING & SEARCH
// ============================================

function handleSearch(e) {
    state.searchQuery = e.target.value.toLowerCase();
    applyFilters();
}

function handleCategoryFilter(category) {
    state.selectedCategory = category;

    // Update UI
    document.querySelectorAll('[data-category]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.category === category);
    });

    applyFilters();
}

function handleDifficultyFilter(difficulty) {
    state.selectedDifficulty = difficulty;

    // Update UI
    document.querySelectorAll('[data-difficulty]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.difficulty === difficulty);
    });

    applyFilters();
}

function handleSort(sortBy) {
    state.sortBy = sortBy;
    applyFilters();
}

function applyFilters() {
    let filtered = [...state.workflows];

    // Search filter
    if (state.searchQuery) {
        filtered = filtered.filter(w =>
            w.name.toLowerCase().includes(state.searchQuery) ||
            w.description.toLowerCase().includes(state.searchQuery) ||
            w.tags.some(tag => tag.includes(state.searchQuery))
        );
    }

    // Category filter
    if (state.selectedCategory !== 'all') {
        filtered = filtered.filter(w => w.categorySlug === state.selectedCategory);
    }

    // Difficulty filter
    if (state.selectedDifficulty !== 'all') {
        filtered = filtered.filter(w => w.difficulty === state.selectedDifficulty);
    }

    // Sort
    filtered.sort((a, b) => {
        switch (state.sortBy) {
            case 'popular':
                return b.deployments - a.deployments;
            case 'newest':
                return new Date(b.addedDate) - new Date(a.addedDate);
            case 'timeSaved':
                return b.timeSavedMinutes - a.timeSavedMinutes;
            case 'rating':
                return b.rating - a.rating;
            default:
                return 0;
        }
    });

    state.filteredWorkflows = filtered;
    renderWorkflows();
}

// ============================================
// RENDERING
// ============================================

function renderWorkflows() {
    const featuredWorkflows = state.filteredWorkflows.filter(w => w.featured);
    const allWorkflows = state.filteredWorkflows;

    // Render featured section
    const featuredGrid = document.getElementById('featuredGrid');
    if (featuredWorkflows.length > 0) {
        document.getElementById('featuredSection').style.display = 'block';
        featuredGrid.innerHTML = featuredWorkflows.map(renderWorkflowCard).join('');
    } else {
        document.getElementById('featuredSection').style.display = 'none';
    }

    // Render all workflows
    const workflowGrid = document.getElementById('workflowGrid');
    const emptyState = document.getElementById('emptyState');

    if (allWorkflows.length > 0) {
        document.getElementById('allSection').style.display = 'block';
        emptyState.style.display = 'none';
        workflowGrid.innerHTML = allWorkflows.map(renderWorkflowCard).join('');
    } else {
        document.getElementById('allSection').style.display = 'none';
        emptyState.style.display = 'block';
    }

    // Add click handlers
    attachCardEventListeners();
}

function renderWorkflowCard(workflow) {
    const stars = '★'.repeat(Math.floor(workflow.rating)) +
                  (workflow.rating % 1 >= 0.5 ? '½' : '') +
                  '☆'.repeat(5 - Math.ceil(workflow.rating));

    const difficultyClass = workflow.difficulty;
    const difficultyIcon = {
        beginner: '⚡',
        intermediate: '⚙️',
        advanced: '🚀'
    }[workflow.difficulty];

    return `
        <div class="workflow-card" data-id="${workflow.id}">
            <div class="workflow-header">
                <span class="workflow-icon">${workflow.icon}</span>
                <div class="workflow-title-group">
                    <h3 class="workflow-title">${workflow.name}</h3>
                    <p class="workflow-author">by ${workflow.author}</p>
                </div>
            </div>

            <div class="workflow-rating">
                <span class="stars">${stars}</span>
                <span class="rating-value">${workflow.rating.toFixed(1)}/5.0</span>
                <span class="rating-count">(${workflow.reviewCount.toLocaleString()} ratings)</span>
            </div>

            <div class="workflow-diagram">
                <div class="mini-diagram">
                    ${workflow.diagram.map((node, i) => `
                        ${i > 0 ? '<span class="mini-arrow">→</span>' : ''}
                        <div class="mini-node" title="${node.name}">${node.label}</div>
                    `).join('')}
                </div>
            </div>

            <div class="workflow-metrics">
                <div class="metric">
                    <span class="metric-icon">💡</span>
                    <span class="metric-value">${workflow.timeSaved}</span>
                </div>
                <div class="metric">
                    <span class="metric-icon">🚀</span>
                    <span class="metric-value">${workflow.deployments.toLocaleString()} deploys</span>
                </div>
                <div class="metric">
                    <span class="metric-icon">✅</span>
                    <span class="metric-value">${Math.round(workflow.successRate * 100)}% success</span>
                </div>
                <div class="metric">
                    <span class="metric-icon">${difficultyIcon}</span>
                    <span class="difficulty ${difficultyClass}">${workflow.difficulty}</span>
                </div>
            </div>

            <div class="workflow-testimonial">
                "${workflow.testimonial}"
                <div class="testimonial-author">— ${workflow.testimonialAuthor}</div>
            </div>

            <div class="workflow-tags">
                ${workflow.tags.slice(0, 4).map(tag =>
                    `<span class="tag">${tag}</span>`
                ).join('')}
            </div>

            <div class="workflow-actions">
                <button class="btn btn-primary" onclick="openDeployModal('${workflow.id}')">
                    Use This Workflow
                </button>
                <button class="btn btn-secondary" onclick="openDetailModal('${workflow.id}')">
                    Learn More
                </button>
            </div>
        </div>
    `;
}

function attachCardEventListeners() {
    // Cards are clickable
    document.querySelectorAll('.workflow-card').forEach(card => {
        card.style.cursor = 'pointer';
        // Card click opens detail (but not if clicking buttons)
        card.addEventListener('click', (e) => {
            if (!e.target.closest('button')) {
                const id = card.dataset.id;
                openDetailModal(id);
            }
        });
    });
}

// ============================================
// MODALS
// ============================================

function openDetailModal(workflowId) {
    const workflow = state.workflows.find(w => w.id === workflowId);
    if (!workflow) return;

    const modal = document.getElementById('detailModal');
    const modalHeader = document.getElementById('modalHeader');
    const modalBody = document.getElementById('modalBody');

    modalHeader.innerHTML = `
        <div>
            <h2>${workflow.icon} ${workflow.name}</h2>
            <p style="color: var(--gray-600); margin-top: 0.5rem;">by ${workflow.author}</p>
        </div>
    `;

    modalBody.innerHTML = `
        <div class="modal-section">
            <h3 class="modal-section-title">Overview</h3>
            <p>${workflow.longDescription}</p>
        </div>

        <div class="modal-section">
            <h3 class="modal-section-title">How It Works</h3>
            <ol style="padding-left: 1.5rem; line-height: 1.8;">
                ${workflow.diagram.map(node =>
                    `<li><strong>${node.name}</strong>: ${node.type} operation</li>`
                ).join('')}
            </ol>
        </div>

        <div class="modal-section">
            <h3 class="modal-section-title">Setup Instructions</h3>
            <ol style="padding-left: 1.5rem; line-height: 1.8;">
                ${workflow.setupSteps.map(step => `<li>${step}</li>`).join('')}
            </ol>
        </div>

        <div class="modal-section">
            <h3 class="modal-section-title">Required Integrations</h3>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                ${workflow.requiredIntegrations.map(int =>
                    `<span class="tag">${int}</span>`
                ).join('')}
            </div>
        </div>

        <div class="modal-section">
            <h3 class="modal-section-title">Performance Metrics</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <strong>Avg Execution Time:</strong><br>
                    ${workflow.metrics.avgExecutionTime}
                </div>
                <div>
                    <strong>Cost Per Run:</strong><br>
                    ${workflow.metrics.costPerRun}
                </div>
                <div>
                    <strong>Success Rate:</strong><br>
                    ${Math.round(workflow.successRate * 100)}%
                </div>
            </div>
        </div>

        <div class="modal-section">
            <h3 class="modal-section-title">User Reviews</h3>
            <div class="workflow-testimonial">
                "${workflow.testimonial}"
                <div class="testimonial-author">— ${workflow.testimonialAuthor}</div>
            </div>
            <p style="text-align: center; margin-top: 1rem;">
                <strong>${workflow.rating.toFixed(1)}/5.0</strong>
                (${workflow.reviewCount.toLocaleString()} reviews)
            </p>
        </div>

        <div class="modal-section" style="text-align: center;">
            <button class="btn btn-primary" onclick="closeDetailModal(); openDeployModal('${workflow.id}')">
                Use This Workflow
            </button>
        </div>
    `;

    modal.classList.add('active');
}

function closeDetailModal() {
    document.getElementById('detailModal').classList.remove('active');
}

function openDeployModal(workflowId) {
    const workflow = state.workflows.find(w => w.id === workflowId);
    if (!workflow) return;

    const modal = document.getElementById('deployModal');
    const modalBody = document.getElementById('deployModalBody');

    document.getElementById('deployModalTitle').textContent = `Deploy "${workflow.name}"`;

    modalBody.innerHTML = `
        <form id="deployForm" onsubmit="handleDeploy(event, '${workflow.id}')">
            <div class="deploy-step">
                <h3 class="step-title">Step 1: Configure Credentials</h3>
                ${workflow.requiredIntegrations.map(integration => `
                    <div class="input-group">
                        <label class="input-label">${integration.toUpperCase()} API Key:</label>
                        <input type="text" class="input-field" placeholder="Enter your ${integration} API key" required>
                        <small style="color: var(--gray-500);">
                            <a href="#" style="color: var(--primary);">Where do I find this?</a>
                        </small>
                    </div>
                `).join('')}
            </div>

            <div class="deploy-step">
                <h3 class="step-title">Step 2: Customize Settings (Optional)</h3>
                <div class="input-group">
                    <label class="input-label">Workflow Name (for your records):</label>
                    <input type="text" class="input-field" value="${workflow.name}" placeholder="My ${workflow.name}">
                </div>
                <div class="input-group">
                    <label class="input-label">Execution Schedule:</label>
                    <select class="input-field">
                        <option value="manual">Manual (run on-demand)</option>
                        <option value="hourly">Every hour</option>
                        <option value="daily">Daily at 9:00 AM</option>
                        <option value="weekly">Weekly on Monday</option>
                    </select>
                </div>
            </div>

            <div class="deploy-step">
                <h3 class="step-title">Step 3: Choose Deployment</h3>
                <div class="radio-group">
                    <label class="radio-option">
                        <input type="radio" name="deployment" value="cloud" checked>
                        <div class="radio-content">
                            <div class="radio-title">☁️ HoloLoom Cloud (Recommended)</div>
                            <div class="radio-description">
                                Easiest option. Hosted on our infrastructure.
                                No setup required. Free tier available.
                            </div>
                        </div>
                    </label>

                    <label class="radio-option">
                        <input type="radio" name="deployment" value="heroku">
                        <div class="radio-content">
                            <div class="radio-title">🚀 Heroku</div>
                            <div class="radio-description">
                                Deploy to your Heroku account.
                                Estimated cost: $7/month (Hobby tier).
                            </div>
                        </div>
                    </label>

                    <label class="radio-option">
                        <input type="radio" name="deployment" value="local">
                        <div class="radio-content">
                            <div class="radio-title">💻 Local Docker</div>
                            <div class="radio-description">
                                Run on your own machine. Free but requires Docker installed.
                            </div>
                        </div>
                    </label>
                </div>

                <div class="cost-estimate">
                    <strong>Estimated Monthly Cost:</strong> $0-7
                    (depending on deployment option)
                </div>
            </div>

            <div style="display: flex; gap: 1rem; justify-content: flex-end; margin-top: 2rem;">
                <button type="button" class="btn btn-secondary" onclick="closeDeployModal()">
                    Cancel
                </button>
                <button type="submit" class="btn btn-primary">
                    Deploy Workflow
                </button>
            </div>
        </form>
    `;

    modal.classList.add('active');
}

function closeDeployModal() {
    document.getElementById('deployModal').classList.remove('active');
}

// ============================================
// DEPLOYMENT
// ============================================

async function handleDeploy(event, workflowId) {
    event.preventDefault();

    const workflow = state.workflows.find(w => w.id === workflowId);
    const formData = new FormData(event.target);
    const deployment = formData.get('deployment');

    // Show loading state
    showToast('Deploying workflow...', 'info');

    // Simulate API call
    try {
        // In production, this would call:
        // const response = await fetch('/api/workflows/deploy', {
        //     method: 'POST',
        //     body: JSON.stringify({ workflowId, deployment, credentials: ... })
        // });

        await new Promise(resolve => setTimeout(resolve, 2000));

        closeDeployModal();
        showToast(`✓ "${workflow.name}" deployed successfully!`, 'success');

        // Could redirect to dashboard
        // window.location.href = `/dashboard?workflow=${workflowId}`;

    } catch (error) {
        showToast(`✗ Deployment failed: ${error.message}`, 'error');
    }
}

// ============================================
// TOAST NOTIFICATIONS
// ============================================

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');

    toast.className = `toast ${type}`;
    toastMessage.textContent = message;
    toast.classList.add('active');

    setTimeout(() => {
        toast.classList.remove('active');
    }, 3000);
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatTimeSaved(minutes) {
    if (minutes < 60) {
        return `${minutes} min`;
    } else if (minutes < 1440) {
        const hours = Math.floor(minutes / 60);
        return `${hours} hour${hours > 1 ? 's' : ''}`;
    } else {
        const days = Math.floor(minutes / 1440);
        return `${days} day${days > 1 ? 's' : ''}`;
    }
}

function calculateROI(timeSavedMinutes, costPerMonth = 7, hourlyRate = 100) {
    const monthlyHoursSaved = (timeSavedMinutes / 60) * 22; // 22 working days
    const monthlyValue = monthlyHoursSaved * hourlyRate;
    const roi = monthlyValue / costPerMonth;
    return {
        value: monthlyValue,
        roi: roi,
        hoursSaved: monthlyHoursSaved
    };
}

// ============================================
// EXPORT FOR TESTING
// ============================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        SAMPLE_WORKFLOWS,
        applyFilters,
        calculateROI
    };
}
