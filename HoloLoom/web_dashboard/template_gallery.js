/**
 * HoloLoom Template Gallery
 * Advanced template management, discovery, and loading system
 *
 * Features:
 * - Dynamic template loading from filesystem
 * - Search and filtering with fuzzy matching
 * - Template metadata discovery
 * - Preview generation with workflow visualization
 * - Template recommendations based on use case
 * - Analytics and usage tracking
 */

class TemplateGallery {
    constructor() {
        this.templates = [];
        this.filteredTemplates = [];
        this.currentCategory = 'all';
        this.searchQuery = '';
        this.selectedTemplate = null;
        this.templateCache = new Map();
        this.usageStats = this.loadUsageStats();
    }

    /**
     * Initialize template gallery
     */
    async init() {
        try {
            await this.loadTemplatesFromFilesystem();
            this.renderCategoryTabs();
            this.setupEventListeners();
            this.displayTemplates();
            this.trackPageView();
        } catch (error) {
            console.error('Failed to initialize template gallery:', error);
            this.showError('Failed to load templates');
        }
    }

    /**
     * Load templates from example_workflows directory
     */
    async loadTemplatesFromFilesystem() {
        const templateFiles = [
            'research_pipeline.json',
            'safety_gated_query.json',
            'bdr_outbound_sequence.json',
            'crm/lead_scoring_simple.json',
            'crm/multi_factor_scoring.json',
            'crm/daily_action_list.json',
            'llm/customer_support_triage.json',
            'llm/content_creation.json'
        ];

        this.templates = [];

        for (const file of templateFiles) {
            try {
                const response = await fetch(`example_workflows/${file}`);
                const workflow = await response.json();

                // Extract metadata from workflow
                const template = this.extractTemplateMetadata(workflow, file);
                this.templates.push(template);
                this.templateCache.set(template.id, workflow);
            } catch (error) {
                console.warn(`Failed to load template: ${file}`, error);
            }
        }

        // Sort by popularity (usage count)
        this.templates.sort((a, b) => {
            const usageA = this.usageStats[a.id] || 0;
            const usageB = this.usageStats[b.id] || 0;
            return usageB - usageA;
        });
    }

    /**
     * Extract metadata from workflow JSON
     */
    extractTemplateMetadata(workflow, filename) {
        const id = filename.replace('.json', '').split('/').pop();
        const baseMetadata = {
            id,
            filename,
            name: workflow.name || this.humanizeName(id),
            agents: workflow.nodes?.length || 0,
            connections: workflow.connections?.length || 0,
            complexity: this.calculateComplexity(workflow),
            estimatedTime: this.estimateExecutionTime(workflow),
        };

        // Try to infer category from filename
        let category = 'research';
        if (filename.includes('crm')) category = 'crm';
        else if (filename.includes('safety')) category = 'safety';
        else if (filename.includes('support')) category = 'support';
        else if (filename.includes('content')) category = 'content';

        return {
            ...baseMetadata,
            category,
            description: this.generateDescription(workflow),
            icon: this.getCategoryIcon(category),
            tags: this.extractTags(workflow),
            status: this.getTemplateStatus(id),
        };
    }

    /**
     * Calculate workflow complexity (1-3)
     */
    calculateComplexity(workflow) {
        const nodes = workflow.nodes?.length || 0;
        const connections = workflow.connections?.length || 0;
        const hasDecisions = nodes > 6 || connections > 8;
        const hasBranching = connections > nodes;

        if (nodes <= 3) return 1;
        if (nodes <= 6 && !hasDecisions) return 2;
        return 3;
    }

    /**
     * Estimate execution time based on workflow
     */
    estimateExecutionTime(workflow) {
        const nodes = workflow.nodes?.length || 0;
        const avgTimePerNode = 500; // 500ms per node

        const totalMs = nodes * avgTimePerNode;
        if (totalMs < 60000) return '<1 min';
        const minutes = Math.ceil(totalMs / 60000);
        const maxMinutes = minutes + 2;
        return `${minutes}-${maxMinutes} min`;
    }

    /**
     * Generate description from workflow
     */
    generateDescription(workflow) {
        const descriptions = {
            'Research Pipeline': 'Multi-query research with synthesis and refinement for comprehensive exploration.',
            'Safety-Gated Query': 'Query with safety checks and conditional branching for risk-aware execution.',
            'BDR Outbound Sequence': 'Business development outreach automation with CRM integration.',
            'Lead Scoring': 'Intelligent lead qualification and scoring for sales prioritization.',
            'Daily Action List': 'Generate prioritized daily tasks from CRM data and context.',
            'Customer Support Triage': 'Intelligent routing and triage of customer support tickets.',
            'Content Creation': 'Multi-stage content generation with research, drafting, and refinement.',
        };

        return descriptions[workflow.name] || 'Pre-built workflow for automation tasks';
    }

    /**
     * Extract tags from workflow agents
     */
    extractTags(workflow) {
        const agentTypes = new Set();
        const tags = [];

        workflow.nodes?.forEach(node => {
            if (node.agentType) {
                agentTypes.add(node.agentType);
            }
        });

        // Map agent types to tags
        const agentToTag = {
            'hololoom': 'Query',
            'multiquery': 'Multi-Query',
            'safety': 'Safety',
            'synthesizer': 'Synthesis',
            'refiner': 'Refinement',
            'response': 'Output',
            'conditional': 'Branching',
        };

        agentTypes.forEach(type => {
            const tag = agentToTag[type];
            if (tag && !tags.includes(tag)) tags.push(tag);
        });

        return tags.length > 0 ? tags : ['Workflow'];
    }

    /**
     * Get template status (new, popular, beta, etc.)
     */
    getTemplateStatus(id) {
        const newTemplates = ['safety_gated_query', 'content_creation'];
        const betaTemplates = ['content_creation'];
        const popularTemplates = ['research_pipeline', 'lead_scoring_simple'];

        if (newTemplates.includes(id)) return 'new';
        if (betaTemplates.includes(id)) return 'beta';
        if (popularTemplates.includes(id)) return 'popular';
        return null;
    }

    /**
     * Get category icon
     */
    getCategoryIcon(category) {
        const icons = {
            'research': '🔍',
            'crm': '👥',
            'support': '🎧',
            'content': '✍️',
            'safety': '🛡️',
        };
        return icons[category] || '📋';
    }

    /**
     * Humanize template ID to readable name
     */
    humanizeName(id) {
        return id
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    /**
     * Filter templates by category and search
     */
    getFilteredTemplates() {
        return this.templates.filter(t => {
            const matchesCategory = this.currentCategory === 'all' || t.category === this.currentCategory;
            const matchesSearch = this.matchesSearch(t);
            return matchesCategory && matchesSearch;
        });
    }

    /**
     * Fuzzy search matching
     */
    matchesSearch(template) {
        if (!this.searchQuery) return true;

        const query = this.searchQuery.toLowerCase();
        const fields = [
            template.name,
            template.description,
            ...template.tags,
        ];

        return fields.some(field =>
            field.toLowerCase().includes(query)
        );
    }

    /**
     * Get template recommendations based on use case
     */
    getRecommendations(useCase) {
        const recommendations = {
            'sales': ['lead_scoring_simple', 'multi_factor_scoring', 'daily_action_list'],
            'marketing': ['content_creation', 'research_pipeline'],
            'support': ['customer_support_triage', 'safety_gated_query'],
            'research': ['research_pipeline', 'safety_gated_query'],
            'security': ['safety_gated_query'],
        };

        const ids = recommendations[useCase] || [];
        return this.templates.filter(t => ids.includes(t.id));
    }

    /**
     * Render category filter tabs
     */
    renderCategoryTabs() {
        // Implementation in HTML (kept for reference)
        console.log('Category tabs rendered');
    }

    /**
     * Display templates in grid
     */
    displayTemplates() {
        this.filteredTemplates = this.getFilteredTemplates();
        console.log(`Displaying ${this.filteredTemplates.length} templates`);
    }

    /**
     * Preview template with workflow visualization
     */
    async previewTemplate(templateId) {
        const template = this.templates.find(t => t.id === templateId);
        if (!template) return;

        this.selectedTemplate = template;

        try {
            const workflow = this.templateCache.get(template.id) ||
                            await this.loadTemplateWorkflow(template.filename);

            this.generatePreviewDiagram(workflow);
        } catch (error) {
            console.error('Failed to preview template:', error);
        }
    }

    /**
     * Load template workflow JSON
     */
    async loadTemplateWorkflow(filename) {
        const response = await fetch(`example_workflows/${filename}`);
        return await response.json();
    }

    /**
     * Generate workflow diagram for preview
     */
    generatePreviewDiagram(workflow) {
        // Simple ASCII diagram generation
        const nodes = workflow.nodes || [];
        const connections = workflow.connections || [];

        let diagram = '';
        nodes.forEach((node, index) => {
            diagram += `[${node.agentType}]`;
            if (index < nodes.length - 1) diagram += ' → ';
        });

        return diagram;
    }

    /**
     * Load selected template in builder
     */
    loadTemplate(template) {
        // Track usage
        this.trackTemplateUsage(template.id);

        // Save to session storage
        sessionStorage.setItem('selectedTemplate', template.filename);

        // Redirect to builder
        window.location.href = `workflow_builder.html?template=${template.filename}`;
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Implementation in HTML (kept for reference)
        console.log('Event listeners configured');
    }

    /**
     * Usage Statistics
     */
    loadUsageStats() {
        try {
            const stats = localStorage.getItem('templateUsageStats');
            return stats ? JSON.parse(stats) : {};
        } catch {
            return {};
        }
    }

    /**
     * Track template usage
     */
    trackTemplateUsage(templateId) {
        const stats = this.loadUsageStats();
        stats[templateId] = (stats[templateId] || 0) + 1;
        localStorage.setItem('templateUsageStats', JSON.stringify(stats));
    }

    /**
     * Track page view for analytics
     */
    trackPageView() {
        const event = {
            type: 'template_gallery_view',
            timestamp: new Date().toISOString(),
            templateCount: this.templates.length,
        };
        console.log('Analytics:', event);
    }

    /**
     * Show error message
     */
    showError(message) {
        const grid = document.getElementById('templatesGrid');
        grid.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: #e74c3c;">
                <div style="font-size: 48px; margin-bottom: 15px;">⚠️</div>
                <div style="font-weight: 600; margin-bottom: 5px;">${message}</div>
            </div>
        `;
    }

    /**
     * Export template as JSON
     */
    exportTemplate(templateId) {
        const workflow = this.templateCache.get(templateId);
        if (!workflow) return;

        const json = JSON.stringify(workflow, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `template_${templateId}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Import template from file
     */
    async importTemplate(file) {
        try {
            const json = await file.text();
            const workflow = JSON.parse(json);
            const template = this.extractTemplateMetadata(workflow, file.name);
            return { template, workflow };
        } catch (error) {
            console.error('Failed to import template:', error);
            throw error;
        }
    }

    /**
     * Get template statistics
     */
    getStatistics() {
        return {
            totalTemplates: this.templates.length,
            categories: [...new Set(this.templates.map(t => t.category))],
            avgComplexity: (
                this.templates.reduce((sum, t) => sum + t.complexity, 0) /
                this.templates.length
            ).toFixed(1),
            avgAgents: (
                this.templates.reduce((sum, t) => sum + t.agents, 0) /
                this.templates.length
            ).toFixed(1),
        };
    }
}

// Initialize gallery when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.gallery = new TemplateGallery();
        window.gallery.init();
    });
} else {
    window.gallery = new TemplateGallery();
    window.gallery.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TemplateGallery;
}
