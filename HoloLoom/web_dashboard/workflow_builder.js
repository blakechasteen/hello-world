/**
 * HoloLoom Workflow Builder - Interactive Agent Workflow Designer
 *
 * Features:
 * - Drag-and-drop agent placement
 * - Visual connection drawing
 * - Real-time workflow execution
 * - Export/import workflows as JSON
 * - Multi-agent orchestration
 */

// Global state
let nodes = [];
let connections = [];
let selectedNode = null;
let draggedNode = null;
let connectionStart = null;
let nextNodeId = 1;
let executionState = {
    running: false,
    currentNode: null,
    results: {}
};

// Version control state
let versionHistory = [];
let currentVersion = 1;
let currentBranch = 'main';
let branches = {
    'main': {
        versions: [],
        head: null
    }
};

// Agent type definitions
const agentDefinitions = {
    hololoom: {
        name: 'HoloLoom Query',
        type: 'query',
        color: '#667eea',
        inputs: ['query'],
        outputs: ['spacetime'],
        config: {
            pattern: { type: 'select', options: ['bare', 'fast', 'fused'], default: 'fast' },
            return_trace: { type: 'boolean', default: true }
        }
    },
    search: {
        name: 'Memory Search',
        type: 'query',
        color: '#667eea',
        inputs: ['query'],
        outputs: ['memories'],
        config: {
            max_results: { type: 'number', default: 10, min: 1, max: 100 },
            similarity_threshold: { type: 'number', default: 0.7, min: 0, max: 1, step: 0.1 }
        }
    },
    multiquery: {
        name: 'Multi-Query',
        type: 'query',
        color: '#667eea',
        inputs: ['query'],
        outputs: ['subqueries'],
        config: {
            max_subqueries: { type: 'number', default: 5, min: 2, max: 10 },
            mode: { type: 'select', options: ['research', 'verify', 'plan_execute'], default: 'research' }
        }
    },
    embedder: {
        name: 'Matryoshka Embedder',
        type: 'process',
        color: '#f093fb',
        inputs: ['text'],
        outputs: ['embeddings'],
        config: {
            scales: { type: 'text', default: '96,192,384' },
            normalize: { type: 'boolean', default: true }
        }
    },
    synthesizer: {
        name: 'Synthesizer',
        type: 'process',
        color: '#f093fb',
        inputs: ['text'],
        outputs: ['synthesis'],
        config: {
            extract_entities: { type: 'boolean', default: true },
            extract_motifs: { type: 'boolean', default: true }
        }
    },
    refiner: {
        name: 'Recursive Refiner',
        type: 'process',
        color: '#f093fb',
        inputs: ['spacetime'],
        outputs: ['refined'],
        config: {
            strategy: { type: 'select', options: ['refine', 'critique', 'verify', 'elegance'], default: 'refine' },
            max_iterations: { type: 'number', default: 3, min: 1, max: 10 }
        }
    },
    store: {
        name: 'Memory Store',
        type: 'memory',
        color: '#4facfe',
        inputs: ['data'],
        outputs: ['stored'],
        config: {
            backend: { type: 'select', options: ['inmemory', 'hybrid', 'hyperspace'], default: 'inmemory' }
        }
    },
    retrieve: {
        name: 'Context Retriever',
        type: 'memory',
        color: '#4facfe',
        inputs: ['query'],
        outputs: ['context'],
        config: {
            k: { type: 'number', default: 5, min: 1, max: 50 },
            use_fusion: { type: 'boolean', default: false }
        }
    },
    fusion: {
        name: 'Knowledge Fusion',
        type: 'memory',
        color: '#4facfe',
        inputs: ['query'],
        outputs: ['expanded'],
        config: {
            max_depth: { type: 'number', default: 2, min: 1, max: 5 },
            min_importance: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.1 }
        }
    },
    thompson: {
        name: 'Thompson Sampler',
        type: 'decision',
        color: '#43e97b',
        inputs: ['options'],
        outputs: ['selected'],
        config: {
            exploration_rate: { type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 }
        }
    },
    convergence: {
        name: 'Convergence Engine',
        type: 'decision',
        color: '#43e97b',
        inputs: ['features'],
        outputs: ['decision'],
        config: {
            strategy: { type: 'select', options: ['argmax', 'epsilon_greedy', 'bayesian_blend'], default: 'epsilon_greedy' }
        }
    },
    safety: {
        name: 'Safety Guardrails',
        type: 'decision',
        color: '#43e97b',
        inputs: ['action'],
        outputs: ['gated'],
        config: {
            risk_threshold: { type: 'select', options: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], default: 'MEDIUM' },
            enable_human_in_loop: { type: 'boolean', default: false }
        }
    },
    response: {
        name: 'Response Generator',
        type: 'output',
        color: '#fa709a',
        inputs: ['data'],
        outputs: ['response'],
        config: {
            format: { type: 'select', options: ['text', 'json', 'markdown'], default: 'text' }
        }
    },
    format: {
        name: 'Format Converter',
        type: 'output',
        color: '#fa709a',
        inputs: ['data'],
        outputs: ['formatted'],
        config: {
            output_format: { type: 'select', options: ['json', 'markdown', 'html', 'yaml'], default: 'json' }
        }
    },
    conditional: {
        name: 'Conditional Branch',
        type: 'control',
        color: '#30cfd0',
        inputs: ['condition'],
        outputs: ['true', 'false'],
        config: {
            condition_type: { type: 'select', options: ['confidence', 'count', 'custom'], default: 'confidence' },
            threshold: { type: 'number', default: 0.75, min: 0, max: 1, step: 0.05 }
        }
    },
    loop: {
        name: 'Loop Iterator',
        type: 'control',
        color: '#30cfd0',
        inputs: ['data'],
        outputs: ['iteration'],
        config: {
            max_iterations: { type: 'number', default: 10, min: 1, max: 100 },
            break_condition: { type: 'text', default: 'confidence > 0.9' }
        }
    },
    parallel: {
        name: 'Parallel Executor',
        type: 'control',
        color: '#30cfd0',
        inputs: ['tasks'],
        outputs: ['results'],
        config: {
            max_concurrent: { type: 'number', default: 5, min: 1, max: 20 }
        }
    },
    // ========== LLM AGENTS ==========
    llm_prompt: {
        name: 'LLM Prompt',
        type: 'llm',
        color: '#ff6b6b',
        inputs: ['prompt', 'context'],
        outputs: ['response', 'usage'],
        config: {
            provider: {
                type: 'select',
                options: ['openai', 'anthropic', 'ollama', 'local'],
                default: 'openai'
            },
            model: {
                type: 'select',
                options: [
                    'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo',
                    'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku',
                    'llama3', 'mistral', 'gemma'
                ],
                default: 'gpt-4'
            },
            temperature: { type: 'number', default: 0.7, min: 0, max: 2, step: 0.1 },
            max_tokens: { type: 'number', default: 1000, min: 1, max: 4000, step: 100 },
            system_prompt: {
                type: 'textarea',
                default: 'You are a helpful assistant.',
                placeholder: 'System instructions for the LLM...'
            },
            user_prompt_template: {
                type: 'textarea',
                default: '${input.text}',
                placeholder: 'User prompt template. Use ${variable} for substitution.'
            }
        }
    },
    structured_llm: {
        name: 'Structured Output LLM',
        type: 'llm',
        color: '#ff6b6b',
        inputs: ['prompt'],
        outputs: ['structured_data', 'raw_response'],
        config: {
            provider: {
                type: 'select',
                options: ['openai', 'anthropic'],
                default: 'openai'
            },
            model: {
                type: 'select',
                options: ['gpt-4', 'gpt-4-turbo', 'claude-3-opus', 'claude-3-sonnet'],
                default: 'gpt-4'
            },
            temperature: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.1 },
            output_schema: {
                type: 'json',
                default: JSON.stringify({
                    type: 'object',
                    properties: {
                        name: { type: 'string', description: 'Person name' },
                        email: { type: 'string', format: 'email' },
                        sentiment: { type: 'string', enum: ['positive', 'neutral', 'negative'] }
                    },
                    required: ['name', 'email', 'sentiment']
                }, null, 2),
                placeholder: 'JSON Schema defining the expected output structure'
            },
            enforce_schema: { type: 'boolean', default: true },
            retry_on_invalid: { type: 'boolean', default: true },
            max_retries: { type: 'number', default: 3, min: 1, max: 5 }
        }
    },
    prompt_chain: {
        name: 'Prompt Chain',
        type: 'llm',
        color: '#ff6b6b',
        inputs: ['initial_input'],
        outputs: ['final_response', 'intermediate_steps'],
        config: {
            provider: {
                type: 'select',
                options: ['openai', 'anthropic', 'ollama'],
                default: 'openai'
            },
            model: {
                type: 'select',
                options: ['gpt-4', 'gpt-4-turbo', 'claude-3-opus', 'claude-3-sonnet'],
                default: 'gpt-4'
            },
            temperature: { type: 'number', default: 0.7, min: 0, max: 2, step: 0.1 },
            chain_steps: {
                type: 'json',
                default: JSON.stringify([
                    {
                        name: 'extract',
                        prompt: 'Extract key points from: ${input}',
                        temperature: 0.3
                    },
                    {
                        name: 'analyze',
                        prompt: 'Analyze these points: ${extract.output}',
                        temperature: 0.7
                    },
                    {
                        name: 'synthesize',
                        prompt: 'Synthesize insights from the analysis: ${analyze.output}',
                        temperature: 0.9
                    }
                ], null, 2),
                placeholder: 'Array of chain steps with name, prompt, and optional temperature'
            },
            preserve_all_steps: { type: 'boolean', default: true }
        }
    },
    few_shot: {
        name: 'Few-Shot Learner',
        type: 'llm',
        color: '#ff6b6b',
        inputs: ['query', 'examples'],
        outputs: ['response', 'confidence'],
        config: {
            provider: {
                type: 'select',
                options: ['openai', 'anthropic'],
                default: 'openai'
            },
            model: {
                type: 'select',
                options: ['gpt-4', 'gpt-4-turbo', 'claude-3-opus'],
                default: 'gpt-4'
            },
            temperature: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.1 },
            task_description: {
                type: 'textarea',
                default: 'Given the examples below, perform the same task on the new input.',
                placeholder: 'Description of the task the model should learn from examples'
            },
            examples: {
                type: 'json',
                default: JSON.stringify([
                    { input: 'Great product, love it!', output: 'positive' },
                    { input: 'Terrible experience, very disappointed.', output: 'negative' },
                    { input: 'It works fine, nothing special.', output: 'neutral' }
                ], null, 2),
                placeholder: 'Array of {input, output} example pairs'
            },
            num_examples: { type: 'number', default: 3, min: 1, max: 10 },
            auto_select_examples: {
                type: 'boolean',
                default: false,
                description: 'Automatically select most relevant examples from memory'
            }
        }
    },
    llm_consensus: {
        name: 'Multi-Model Consensus',
        type: 'llm',
        color: '#ff6b6b',
        inputs: ['prompt'],
        outputs: ['consensus_response', 'agreement_score', 'individual_responses'],
        config: {
            models: {
                type: 'json',
                default: JSON.stringify([
                    { provider: 'openai', model: 'gpt-4', weight: 1.0 },
                    { provider: 'anthropic', model: 'claude-3-opus', weight: 1.0 },
                    { provider: 'anthropic', model: 'claude-3-sonnet', weight: 0.8 }
                ], null, 2),
                placeholder: 'Array of {provider, model, weight} configs'
            },
            consensus_strategy: {
                type: 'select',
                options: ['majority_vote', 'weighted_average', 'all_agree', 'best_of_n'],
                default: 'majority_vote'
            },
            temperature: { type: 'number', default: 0.7, min: 0, max: 2, step: 0.1 },
            require_unanimous: { type: 'boolean', default: false },
            min_agreement_threshold: { type: 'number', default: 0.6, min: 0, max: 1, step: 0.1 }
        }
    },
    rag_prompt: {
        name: 'RAG Prompt',
        type: 'llm',
        color: '#ff6b6b',
        inputs: ['query', 'context'],
        outputs: ['response', 'sources', 'confidence'],
        config: {
            provider: {
                type: 'select',
                options: ['openai', 'anthropic'],
                default: 'openai'
            },
            model: {
                type: 'select',
                options: ['gpt-4', 'gpt-4-turbo', 'claude-3-opus', 'claude-3-sonnet'],
                default: 'gpt-4'
            },
            temperature: { type: 'number', default: 0.3, min: 0, max: 1, step: 0.1 },
            retrieval_k: { type: 'number', default: 5, min: 1, max: 20 },
            cite_sources: { type: 'boolean', default: true },
            source_format: {
                type: 'select',
                options: ['inline', 'footnotes', 'appendix'],
                default: 'inline'
            },
            system_prompt: {
                type: 'textarea',
                default: 'Answer the question based on the provided context. Cite sources when making claims. If the context does not contain enough information, say so explicitly.',
                placeholder: 'System instructions for RAG prompting'
            },
            require_citations: { type: 'boolean', default: false },
            hallucination_check: { type: 'boolean', default: true }
        }
    },

    // ========== TOOL AGENTS ==========
    tool_call: {
        name: 'Tool Call',
        type: 'tool',
        color: '#ffa726',
        inputs: ['tool_name', 'arguments'],
        outputs: ['result', 'status'],
        config: {
            tool_name: {
                type: 'text',
                default: '',
                placeholder: 'Name of the tool to call (e.g., calculator, search)'
            },
            arguments_template: {
                type: 'textarea',
                default: '{}',
                placeholder: 'JSON arguments for the tool. Use ${variable} for substitution.'
            },
            timeout_seconds: {
                type: 'number',
                default: 30,
                min: 1,
                max: 300,
                step: 1
            },
            retry_on_failure: {
                type: 'boolean',
                default: true
            },
            max_retries: {
                type: 'number',
                default: 3,
                min: 1,
                max: 10
            }
        }
    },

    api_request: {
        name: 'API Request',
        type: 'tool',
        color: '#ffa726',
        inputs: ['url', 'params'],
        outputs: ['response', 'status_code'],
        config: {
            method: {
                type: 'select',
                options: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
                default: 'GET'
            },
            url_template: {
                type: 'text',
                default: '',
                placeholder: 'https://api.example.com/endpoint?param=${value}'
            },
            headers: {
                type: 'textarea',
                default: '{"Content-Type": "application/json"}',
                placeholder: 'JSON object with HTTP headers'
            },
            body_template: {
                type: 'textarea',
                default: '',
                placeholder: 'Request body (for POST/PUT). Use ${variable} for substitution.'
            },
            timeout_seconds: {
                type: 'number',
                default: 30,
                min: 1,
                max: 300
            },
            follow_redirects: {
                type: 'boolean',
                default: true
            },
            verify_ssl: {
                type: 'boolean',
                default: true
            }
        }
    },

    code_executor: {
        name: 'Code Executor',
        type: 'tool',
        color: '#ffa726',
        inputs: ['code', 'context'],
        outputs: ['result', 'stdout', 'stderr'],
        config: {
            language: {
                type: 'select',
                options: ['python', 'javascript', 'bash'],
                default: 'python'
            },
            code_template: {
                type: 'textarea',
                default: '',
                placeholder: 'Code to execute. Use ${variable} to inject variables.'
            },
            execution_mode: {
                type: 'select',
                options: ['safe', 'restricted', 'sandbox'],
                default: 'safe'
            },
            timeout_seconds: {
                type: 'number',
                default: 10,
                min: 1,
                max: 60
            },
            capture_output: {
                type: 'boolean',
                default: true
            },
            allowed_modules: {
                type: 'textarea',
                default: 'math,json,datetime,re',
                placeholder: 'Comma-separated list of allowed Python modules'
            }
        }
    },

    file_ops: {
        name: 'File Operations',
        type: 'tool',
        color: '#ffa726',
        inputs: ['operation', 'path', 'content'],
        outputs: ['result', 'status'],
        config: {
            operation: {
                type: 'select',
                options: ['read', 'write', 'append', 'delete', 'exists', 'list'],
                default: 'read'
            },
            file_path_template: {
                type: 'text',
                default: '',
                placeholder: 'Path to file. Use ${variable} for substitution.'
            },
            content_template: {
                type: 'textarea',
                default: '',
                placeholder: 'Content to write (for write/append operations)'
            },
            encoding: {
                type: 'select',
                options: ['utf-8', 'ascii', 'latin-1'],
                default: 'utf-8'
            },
            create_dirs: {
                type: 'boolean',
                default: true
            },
            overwrite: {
                type: 'boolean',
                default: false
            }
        }
    },
    // ========== GROUP/CONTAINER NODES (Wave 4.4) ==========
    group: {
        name: 'Group Container',
        type: 'group',
        color: '#6c757d',
        inputs: ['input'],
        outputs: ['output'],
        config: {
            label: {
                type: 'text',
                default: 'Group',
                placeholder: 'Group label'
            },
            description: {
                type: 'textarea',
                default: '',
                placeholder: 'Description of this group'
            },
            collapsed: {
                type: 'boolean',
                default: false
            },
            color: {
                type: 'select',
                options: ['gray', 'blue', 'green', 'orange', 'purple', 'red'],
                default: 'gray'
            }
        },
        isContainer: true,
        minWidth: 300,
        minHeight: 200
    },
    subworkflow: {
        name: 'Sub-Workflow',
        type: 'group',
        color: '#6c757d',
        inputs: ['input'],
        outputs: ['output'],
        config: {
            label: {
                type: 'text',
                default: 'Sub-Workflow',
                placeholder: 'Sub-workflow name'
            },
            workflow_id: {
                type: 'text',
                default: '',
                placeholder: 'ID of saved workflow to embed'
            },
            collapsed: {
                type: 'boolean',
                default: false
            }
        },
        isContainer: true,
        minWidth: 300,
        minHeight: 200
    },
    // ========== COMPOSITE WORKFLOW (Wave 7.2) ==========
    composite: {
        name: 'Composite Workflow',
        type: 'composite',
        color: '#9b59b6',
        inputs: ['input'],
        outputs: ['output'],
        config: {
            label: {
                type: 'text',
                default: 'Composite Workflow',
                placeholder: 'Workflow name'
            },
            workflow_id: {
                type: 'workflow_select',
                default: '',
                placeholder: 'Select a saved workflow'
            },
            input_mapping: {
                type: 'json',
                default: '{}',
                placeholder: '{"input_name": "source_path"}'
            },
            output_mapping: {
                type: 'json',
                default: '{}',
                placeholder: '{"output_name": "target_path"}'
            },
            show_preview: {
                type: 'boolean',
                default: true
            }
        },
        description: 'A workflow within a workflow. Double-click to drill down.',
        isComposite: true,
        minWidth: 220,
        minHeight: 120
    }
};

// Initialize drag-and-drop
document.addEventListener('DOMContentLoaded', () => {
    initializeDragAndDrop();
    initializeCanvas();
    setupEventListeners();
    initializeMobileFeatures();
});

/**
 * Initialize mobile-specific features
 * - Bottom sheet for agent palette
 * - Floating action button for adding nodes
 * - Properties panel toggle on tablet
 * - Touch-friendly UI adjustments
 */
function initializeMobileFeatures() {
    // Detect if device is mobile/tablet
    const isMobile = window.matchMedia('(max-width: 768px)').matches;

    if (!isMobile) return; // Skip on desktop

    const agentPalette = document.querySelector('.agent-palette');
    const propertiesPanel = document.querySelector('.properties-panel');

    // Phone: Bottom sheet animation
    if (window.matchMedia('(max-width: 480px)').matches) {
        // Add drag handle for bottom sheet
        const paletteHeader = agentPalette.querySelector('.palette-header');
        if (paletteHeader) {
            paletteHeader.style.cursor = 'grab';
            paletteHeader.style.userSelect = 'none';

            // Allow drag to close on mobile
            let touchStart = 0;
            paletteHeader.addEventListener('touchstart', (e) => {
                touchStart = e.touches[0].clientY;
            }, { passive: true });

            paletteHeader.addEventListener('touchend', (e) => {
                const touchEnd = e.changedTouches[0].clientY;
                const diff = touchEnd - touchStart;

                // Swipe down to close palette
                if (diff > 50) {
                    agentPalette.classList.remove('mobile-open');
                }
            }, { passive: true });
        }

        // Create floating action button for adding nodes
        const fab = document.createElement('button');
        fab.className = 'fab-add-node mobile-only';
        fab.innerHTML = '➕';
        fab.title = 'Add Node (tap to open templates)';
        fab.onclick = () => showTemplatesModal();
        document.body.appendChild(fab);

        // Make palette toggleable with tap
        const canvas = document.getElementById('canvas');
        canvas.addEventListener('tap-hold', (e) => {
            // Long press to open palette (optional feature for quick access)
        }, { passive: true });
    }

    // Tablet: Side panel toggle
    if (window.matchMedia('(min-width: 481px) and (max-width: 768px)').matches) {
        // Create toggle button for properties panel
        const toggle = document.createElement('button');
        toggle.className = 'properties-toggle';
        toggle.innerHTML = '⚙️';
        toggle.title = 'Toggle Properties';
        toggle.onclick = () => {
            propertiesPanel.classList.toggle('mobile-open');
            toggle.classList.toggle('hidden');
        };
        document.body.appendChild(toggle);

        // Close properties panel when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.properties-panel') &&
                !e.target.closest('.properties-toggle')) {
                propertiesPanel.classList.remove('mobile-open');
                toggle.classList.remove('hidden');
            }
        });
    }

    // Handle viewport resize to update mobile state
    window.addEventListener('resize', () => {
        const newIsMobile = window.matchMedia('(max-width: 768px)').matches;
        if (isMobile !== newIsMobile) {
            // Force page reload on orientation change (optional)
            // location.reload();
        }
    });
}

/**
 * Support touch-based template dragging for mobile
 * Implements tap-and-hold to show context menu instead of drag
 */
function setupMobileDragDrop() {
    const templates = document.querySelectorAll('.agent-template');

    templates.forEach(template => {
        let pressTimer = null;
        let startX, startY;

        // Touch start - begin timer for long press
        template.addEventListener('touchstart', (e) => {
            if (e.touches.length !== 1) return;

            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;

            // 500ms long press
            pressTimer = setTimeout(() => {
                const agentType = template.dataset.agent;
                showAgentPreview(agentType);
            }, 500);
        }, { passive: true });

        // Touch move - cancel long press if user moves
        template.addEventListener('touchmove', (e) => {
            const touch = e.touches[0];
            const dist = Math.sqrt(
                Math.pow(touch.clientX - startX, 2) +
                Math.pow(touch.clientY - startY, 2)
            );

            // Cancel if moved more than 10px
            if (dist > 10) {
                clearTimeout(pressTimer);
            }
        }, { passive: true });

        // Touch end - cancel if still pressing
        template.addEventListener('touchend', () => {
            clearTimeout(pressTimer);
        }, { passive: true });

        // Touch cancel - cleanup
        template.addEventListener('touchcancel', () => {
            clearTimeout(pressTimer);
        }, { passive: true });
    });
}

/**
 * Show agent preview/confirmation for mobile users
 * Before adding to canvas
 */
function showAgentPreview(agentType) {
    const definition = agentDefinitions[agentType];
    if (!definition) return;

    const modal = document.createElement('div');
    modal.className = 'modal show';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">Add ${definition.name}</div>
            <div class="modal-body">
                <div style="font-size: 48px; margin-bottom: 15px; text-align: center;">
                    ${getAgentIcon(definition.type)}
                </div>
                <p style="color: #666; text-align: center; margin-bottom: 15px;">
                    Add this agent to your workflow
                </p>
                <div style="background: #f5f5f5; padding: 10px; border-radius: 6px; font-size: 12px; color: #666;">
                    <strong>Inputs:</strong> ${definition.inputs.join(', ')}<br>
                    <strong>Outputs:</strong> ${definition.outputs.join(', ')}
                </div>
            </div>
            <div class="modal-footer">
                <button class="modal-btn secondary" onclick="this.closest('.modal').remove()">Cancel</button>
                <button class="modal-btn primary" onclick="
                    const canvas = document.getElementById('canvas');
                    const rect = canvas.getBoundingClientRect();
                    createNode('${agentType}',
                        rect.width / 2,
                        rect.height / 2);
                    this.closest('.modal').remove();
                    document.querySelector('.agent-palette').classList.remove('mobile-open');
                ">Add to Canvas</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

/**
 * Get emoji icon for agent type
 */
function getAgentIcon(type) {
    const icons = {
        'query': '❓',
        'process': '⚙️',
        'memory': '💾',
        'decision': '🔮',
        'output': '📤',
        'control': '🔀',
        'llm': '🧠',
        'tool': '🔧',
        'group': '📁',
        'composite': '📦'
    };
    return icons[type] || '📌';
}

function initializeDragAndDrop() {
    const templates = document.querySelectorAll('.agent-template');

    // Desktop: Drag-and-drop initialization
    templates.forEach(template => {
        template.addEventListener('dragstart', (e) => {
            const agentType = template.dataset.agent;
            e.dataTransfer.setData('agentType', agentType);
            e.dataTransfer.effectAllowed = 'copy';
        });
    });

    // Mobile: Long-press initialization
    if (window.matchMedia('(max-width: 768px)').matches) {
        setupMobileDragDrop();
    }
}

function initializeCanvas() {
    const canvas = document.getElementById('canvas');

    // Drop handler
    canvas.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    canvas.addEventListener('drop', (e) => {
        e.preventDefault();
        const agentType = e.dataTransfer.getData('agentType');
        if (agentType) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left + canvas.parentElement.scrollLeft;
            const y = e.clientY - rect.top + canvas.parentElement.scrollTop;
            createNode(agentType, x, y);
        }
    });

    // Click to deselect
    canvas.addEventListener('click', (e) => {
        if (e.target === canvas) {
            deselectAll();
        }
    });
}

function createNode(agentType, x, y) {
    const definition = agentDefinitions[agentType];
    if (!definition) return;

    const node = {
        id: `node-${nextNodeId++}`,
        agentType: agentType,
        x: x,
        y: y,
        config: {},
        definition: definition
    };

    // Initialize config with defaults
    Object.entries(definition.config).forEach(([key, configDef]) => {
        node.config[key] = configDef.default;
    });

    nodes.push(node);
    renderNode(node);
    saveUndoState('add_node', `Added ${definition.name}`);
    showToast(`Added ${definition.name}`, 'success');
}

function renderNode(node) {
    const canvas = document.getElementById('canvas');
    const def = node.definition;

    // Check if this is a group/container node
    if (def.isContainer || def.type === 'group') {
        renderGroupNode(node);
        return;
    }

    const nodeEl = document.createElement('div');
    nodeEl.className = 'workflow-node';
    nodeEl.id = node.id;
    nodeEl.style.left = node.x + 'px';
    nodeEl.style.top = node.y + 'px';

    // Header
    const header = document.createElement('div');
    header.className = 'node-header';

    const icon = document.createElement('div');
    icon.className = `node-icon agent-color-${def.type}`;
    icon.textContent = getAgentIcon(def.type);

    const title = document.createElement('div');
    title.className = 'node-title';
    title.textContent = def.name;

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'node-delete';
    deleteBtn.textContent = '×';
    deleteBtn.onclick = () => deleteNode(node.id);

    header.appendChild(icon);
    header.appendChild(title);
    header.appendChild(deleteBtn);

    // Config section
    const configSection = document.createElement('div');
    configSection.className = 'node-config';

    Object.entries(def.config).forEach(([key, configDef]) => {
        const configItem = createConfigInput(node, key, configDef);
        configSection.appendChild(configItem);
    });

    // Connection ports
    const inputPort = document.createElement('div');
    inputPort.className = 'node-port input';
    inputPort.dataset.nodeId = node.id;
    inputPort.dataset.portType = 'input';
    inputPort.addEventListener('click', (e) => handlePortClick(e, node.id, 'input'));

    const outputPort = document.createElement('div');
    outputPort.className = 'node-port output';
    outputPort.dataset.nodeId = node.id;
    outputPort.dataset.portType = 'output';
    outputPort.addEventListener('click', (e) => handlePortClick(e, node.id, 'output'));

    nodeEl.appendChild(header);
    nodeEl.appendChild(configSection);
    nodeEl.appendChild(inputPort);
    nodeEl.appendChild(outputPort);

    // Wave 5.5: Breakpoint indicator
    if (node.breakpoint) {
        const breakpointIndicator = document.createElement('div');
        breakpointIndicator.className = 'breakpoint-indicator';
        breakpointIndicator.title = 'Breakpoint set - click to remove';
        breakpointIndicator.onclick = (e) => {
            e.stopPropagation();
            node.breakpoint = false;
            renderWorkflow();
            showToast('Breakpoint removed', 'info');
        };
        nodeEl.appendChild(breakpointIndicator);
        nodeEl.classList.add('has-breakpoint');
    }

    // Wave 5.5: Execution state indicator
    if (node.executionState) {
        nodeEl.classList.add(`execution-${node.executionState}`);
        if (node.executionState === 'error' && node.executionError) {
            nodeEl.title = `Error: ${node.executionError}`;
        }
    }

    // Make draggable
    makeNodeDraggable(nodeEl, node);

    // Click to select
    nodeEl.addEventListener('click', (e) => {
        e.stopPropagation();
        selectNode(node.id);
    });

    canvas.appendChild(nodeEl);
}

/**
 * Render a group/container node (Wave 4.4)
 */
function renderGroupNode(node) {
    const canvas = document.getElementById('canvas');
    const def = node.definition;

    // Initialize group-specific properties if not present
    if (!node.children) node.children = [];
    if (!node.width) node.width = def.minWidth || 300;
    if (!node.height) node.height = def.minHeight || 200;

    const nodeEl = document.createElement('div');
    const colorClass = `group-color-${node.config.color || 'gray'}`;
    nodeEl.className = `workflow-node group-node ${colorClass}`;
    if (node.config.collapsed) {
        nodeEl.classList.add('collapsed');
    }
    nodeEl.id = node.id;
    nodeEl.style.left = node.x + 'px';
    nodeEl.style.top = node.y + 'px';
    nodeEl.style.width = node.width + 'px';
    nodeEl.style.height = node.config.collapsed ? 'auto' : (node.height + 'px');

    // Group Header
    const header = document.createElement('div');
    header.className = 'group-header';

    // Collapse/Expand button
    const collapseBtn = document.createElement('button');
    collapseBtn.className = 'group-collapse-btn' + (node.config.collapsed ? ' collapsed' : '');
    collapseBtn.innerHTML = '▼';
    collapseBtn.onclick = (e) => {
        e.stopPropagation();
        toggleGroupCollapse(node.id);
    };

    // Editable label
    const labelContainer = document.createElement('div');
    labelContainer.className = 'group-label';
    const labelInput = document.createElement('input');
    labelInput.type = 'text';
    labelInput.value = node.config.label || 'Group';
    labelInput.onclick = (e) => e.stopPropagation();
    labelInput.onchange = (e) => {
        node.config.label = e.target.value;
        saveUndoState('edit_group', `Renamed group to "${e.target.value}"`);
    };
    labelContainer.appendChild(labelInput);

    // Node count badge
    const nodeCount = document.createElement('span');
    nodeCount.className = 'group-node-count';
    nodeCount.textContent = `${node.children.length} node${node.children.length !== 1 ? 's' : ''}`;
    nodeCount.id = `${node.id}-count`;

    // Action buttons
    const actions = document.createElement('div');
    actions.className = 'group-actions';

    const colorBtn = document.createElement('button');
    colorBtn.className = 'group-action-btn';
    colorBtn.innerHTML = '🎨';
    colorBtn.title = 'Change color';
    colorBtn.onclick = (e) => {
        e.stopPropagation();
        cycleGroupColor(node.id);
    };

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'group-action-btn delete';
    deleteBtn.innerHTML = '✕';
    deleteBtn.title = 'Delete group';
    deleteBtn.onclick = (e) => {
        e.stopPropagation();
        deleteGroupNode(node.id);
    };

    actions.appendChild(colorBtn);
    actions.appendChild(deleteBtn);

    header.appendChild(collapseBtn);
    header.appendChild(labelContainer);
    header.appendChild(nodeCount);
    header.appendChild(actions);

    // Group Content area
    const content = document.createElement('div');
    content.className = 'group-content' + (node.config.collapsed ? ' collapsed' : '');
    content.id = `${node.id}-content`;

    // Drop zone indicator
    const dropZone = document.createElement('div');
    dropZone.className = 'group-drop-zone';
    const dropText = document.createElement('span');
    dropText.className = 'group-drop-zone-text';
    dropText.textContent = 'Drop nodes here';
    dropZone.appendChild(dropText);
    content.appendChild(dropZone);

    // Connection ports for group
    const inputPort = document.createElement('div');
    inputPort.className = 'node-port input';
    inputPort.dataset.nodeId = node.id;
    inputPort.dataset.portType = 'input';
    inputPort.addEventListener('click', (e) => handlePortClick(e, node.id, 'input'));

    const outputPort = document.createElement('div');
    outputPort.className = 'node-port output';
    outputPort.dataset.nodeId = node.id;
    outputPort.dataset.portType = 'output';
    outputPort.addEventListener('click', (e) => handlePortClick(e, node.id, 'output'));

    // Resize handle
    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'group-resize-handle';
    makeGroupResizable(resizeHandle, nodeEl, node);

    nodeEl.appendChild(header);
    nodeEl.appendChild(content);
    nodeEl.appendChild(inputPort);
    nodeEl.appendChild(outputPort);
    nodeEl.appendChild(resizeHandle);

    // Make draggable (by header only)
    makeGroupDraggable(header, nodeEl, node);

    // Setup drop zone for receiving nodes
    setupGroupDropZone(nodeEl, node);

    // Click to select
    nodeEl.addEventListener('click', (e) => {
        e.stopPropagation();
        selectNode(node.id);
    });

    canvas.appendChild(nodeEl);

    // Render children inside group
    renderGroupChildren(node);
}

/**
 * Toggle group collapse/expand state
 */
function toggleGroupCollapse(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    node.config.collapsed = !node.config.collapsed;

    const nodeEl = document.getElementById(nodeId);
    const content = document.getElementById(`${nodeId}-content`);
    const collapseBtn = nodeEl.querySelector('.group-collapse-btn');

    if (node.config.collapsed) {
        nodeEl.classList.add('collapsed');
        content.classList.add('collapsed');
        collapseBtn.classList.add('collapsed');
        nodeEl.style.height = 'auto';
    } else {
        nodeEl.classList.remove('collapsed');
        content.classList.remove('collapsed');
        collapseBtn.classList.remove('collapsed');
        nodeEl.style.height = node.height + 'px';
    }

    // Update connections
    renderConnections();
    saveUndoState('toggle_collapse', `${node.config.collapsed ? 'Collapsed' : 'Expanded'} group`);
}

/**
 * Cycle through group colors
 */
function cycleGroupColor(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    const colors = ['gray', 'blue', 'green', 'orange', 'purple', 'red'];
    const currentIndex = colors.indexOf(node.config.color || 'gray');
    const nextIndex = (currentIndex + 1) % colors.length;
    const newColor = colors[nextIndex];

    node.config.color = newColor;

    const nodeEl = document.getElementById(nodeId);
    colors.forEach(c => nodeEl.classList.remove(`group-color-${c}`));
    nodeEl.classList.add(`group-color-${newColor}`);

    saveUndoState('change_color', `Changed group color to ${newColor}`);
}

/**
 * Delete a group node and optionally ungroup its children
 */
function deleteGroupNode(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Ungroup children first (move them out of the group)
    if (node.children && node.children.length > 0) {
        node.children.forEach(childId => {
            const child = nodes.find(n => n.id === childId);
            if (child) {
                // Convert relative position to absolute
                child.x = node.x + (child.relativeX || 0);
                child.y = node.y + 56 + (child.relativeY || 0); // 56 = header height
                delete child.parentGroup;
                delete child.relativeX;
                delete child.relativeY;

                // Re-render child at new position
                const childEl = document.getElementById(childId);
                if (childEl) {
                    childEl.style.left = child.x + 'px';
                    childEl.style.top = child.y + 'px';
                    // Move child out of group to canvas
                    document.getElementById('canvas').appendChild(childEl);
                }
            }
        });
    }

    // Now delete the group node itself
    deleteNode(nodeId);
}

/**
 * Make group resizable
 */
function makeGroupResizable(handle, nodeEl, node) {
    let isResizing = false;
    let startX, startY, startWidth, startHeight;

    handle.addEventListener('mousedown', (e) => {
        e.stopPropagation();
        isResizing = true;
        startX = e.clientX;
        startY = e.clientY;
        startWidth = node.width;
        startHeight = node.height;
        document.body.style.cursor = 'se-resize';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        const minWidth = node.definition.minWidth || 300;
        const minHeight = node.definition.minHeight || 200;

        node.width = Math.max(minWidth, startWidth + dx);
        node.height = Math.max(minHeight, startHeight + dy);

        nodeEl.style.width = node.width + 'px';
        if (!node.config.collapsed) {
            nodeEl.style.height = node.height + 'px';
        }
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            document.body.style.cursor = '';
            saveUndoState('resize_group', 'Resized group');
        }
    });
}

/**
 * Make group draggable by header
 */
function makeGroupDraggable(header, nodeEl, node) {
    let isDragging = false;
    let startX, startY, nodeStartX, nodeStartY;

    header.addEventListener('mousedown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;

        isDragging = true;
        nodeEl.classList.add('dragging');

        const rect = document.getElementById('canvas').getBoundingClientRect();
        const zoom = currentZoom;

        startX = e.clientX;
        startY = e.clientY;
        nodeStartX = node.x;
        nodeStartY = node.y;
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const dx = (e.clientX - startX) / currentZoom;
        const dy = (e.clientY - startY) / currentZoom;

        node.x = nodeStartX + dx;
        node.y = nodeStartY + dy;

        nodeEl.style.left = node.x + 'px';
        nodeEl.style.top = node.y + 'px';

        // Move children with group
        if (node.children) {
            node.children.forEach(childId => {
                const child = nodes.find(n => n.id === childId);
                if (child) {
                    const childEl = document.getElementById(childId);
                    if (childEl) {
                        childEl.style.left = (node.x + (child.relativeX || 0)) + 'px';
                        childEl.style.top = (node.y + 56 + (child.relativeY || 0)) + 'px';
                    }
                }
            });
        }

        renderConnections();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            nodeEl.classList.remove('dragging');
            saveUndoState('move_group', 'Moved group');
        }
    });
}

/**
 * Setup drop zone for receiving nodes into group
 */
function setupGroupDropZone(groupEl, groupNode) {
    groupEl.addEventListener('dragover', (e) => {
        e.preventDefault();
        groupEl.classList.add('drag-over');
    });

    groupEl.addEventListener('dragleave', (e) => {
        if (!groupEl.contains(e.relatedTarget)) {
            groupEl.classList.remove('drag-over');
        }
    });

    groupEl.addEventListener('drop', (e) => {
        e.preventDefault();
        groupEl.classList.remove('drag-over');

        const nodeId = e.dataTransfer.getData('nodeId');
        if (nodeId && nodeId !== groupNode.id) {
            addNodeToGroup(nodeId, groupNode.id);
        }
    });
}

/**
 * Add a node to a group
 */
function addNodeToGroup(nodeId, groupId) {
    const node = nodes.find(n => n.id === nodeId);
    const group = nodes.find(n => n.id === groupId);
    if (!node || !group) return;

    // Check if node is already in another group
    if (node.parentGroup) {
        removeNodeFromGroup(nodeId, node.parentGroup);
    }

    // Check if trying to add a group to itself or create circular reference
    if (node.definition.isContainer && isGroupAncestor(groupId, nodeId)) {
        showToast('Cannot create circular group reference', 'error');
        return;
    }

    // Calculate relative position within group
    node.relativeX = node.x - group.x;
    node.relativeY = node.y - group.y - 56; // 56 = header height
    node.parentGroup = groupId;

    // Add to group's children
    if (!group.children) group.children = [];
    if (!group.children.includes(nodeId)) {
        group.children.push(nodeId);
    }

    // Move node element inside group content
    const nodeEl = document.getElementById(nodeId);
    const groupContent = document.getElementById(`${groupId}-content`);
    if (nodeEl && groupContent) {
        nodeEl.style.left = node.relativeX + 'px';
        nodeEl.style.top = node.relativeY + 'px';
        groupContent.appendChild(nodeEl);
    }

    // Update group node count
    updateGroupNodeCount(groupId);

    renderConnections();
    saveUndoState('add_to_group', `Added node to group`);
}

/**
 * Remove a node from a group
 */
function removeNodeFromGroup(nodeId, groupId) {
    const node = nodes.find(n => n.id === nodeId);
    const group = nodes.find(n => n.id === groupId);
    if (!node || !group) return;

    // Convert relative position to absolute
    node.x = group.x + (node.relativeX || 0);
    node.y = group.y + 56 + (node.relativeY || 0);
    delete node.parentGroup;
    delete node.relativeX;
    delete node.relativeY;

    // Remove from group's children
    if (group.children) {
        group.children = group.children.filter(id => id !== nodeId);
    }

    // Move node element back to canvas
    const nodeEl = document.getElementById(nodeId);
    const canvas = document.getElementById('canvas');
    if (nodeEl && canvas) {
        nodeEl.style.left = node.x + 'px';
        nodeEl.style.top = node.y + 'px';
        canvas.appendChild(nodeEl);
    }

    // Update group node count
    updateGroupNodeCount(groupId);

    renderConnections();
}

/**
 * Check if groupA is an ancestor of groupB (to prevent circular references)
 */
function isGroupAncestor(groupAId, groupBId) {
    const groupB = nodes.find(n => n.id === groupBId);
    if (!groupB || !groupB.children) return false;

    if (groupB.children.includes(groupAId)) return true;

    for (const childId of groupB.children) {
        const child = nodes.find(n => n.id === childId);
        if (child && child.definition.isContainer) {
            if (isGroupAncestor(groupAId, childId)) return true;
        }
    }

    return false;
}

/**
 * Update group node count badge
 */
function updateGroupNodeCount(groupId) {
    const group = nodes.find(n => n.id === groupId);
    if (!group) return;

    const countEl = document.getElementById(`${groupId}-count`);
    if (countEl) {
        const count = group.children ? group.children.length : 0;
        countEl.textContent = `${count} node${count !== 1 ? 's' : ''}`;
    }
}

/**
 * Render children inside a group
 */
function renderGroupChildren(groupNode) {
    if (!groupNode.children || groupNode.children.length === 0) return;

    const groupContent = document.getElementById(`${groupNode.id}-content`);
    if (!groupContent) return;

    groupNode.children.forEach(childId => {
        const child = nodes.find(n => n.id === childId);
        if (!child) return;

        // Check if child element already exists
        let childEl = document.getElementById(childId);
        if (!childEl) {
            // Render the child node (it will be added to canvas, we need to move it)
            const def = child.definition;
            if (def.isContainer || def.type === 'group') {
                renderGroupNode(child);
            } else {
                // Create a temporary renderNode that doesn't add to canvas
                childEl = createNodeElement(child);
            }
            childEl = document.getElementById(childId);
        }

        if (childEl) {
            // Move to group content and set relative position
            childEl.style.left = (child.relativeX || 0) + 'px';
            childEl.style.top = (child.relativeY || 0) + 'px';
            groupContent.appendChild(childEl);
        }
    });
}

/**
 * Create a node element without adding to canvas (for group children)
 */
function createNodeElement(node) {
    const def = node.definition;

    const nodeEl = document.createElement('div');
    nodeEl.className = 'workflow-node';
    nodeEl.id = node.id;
    nodeEl.style.left = node.x + 'px';
    nodeEl.style.top = node.y + 'px';

    // Header
    const header = document.createElement('div');
    header.className = 'node-header';

    const icon = document.createElement('div');
    icon.className = `node-icon agent-color-${def.type}`;
    icon.textContent = getAgentIcon(def.type);

    const title = document.createElement('div');
    title.className = 'node-title';
    title.textContent = def.name;

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'node-delete';
    deleteBtn.textContent = '×';
    deleteBtn.onclick = () => deleteNode(node.id);

    header.appendChild(icon);
    header.appendChild(title);
    header.appendChild(deleteBtn);

    // Config section
    const configSection = document.createElement('div');
    configSection.className = 'node-config';

    Object.entries(def.config).forEach(([key, configDef]) => {
        const configItem = createConfigInput(node, key, configDef);
        configSection.appendChild(configItem);
    });

    // Connection ports
    const inputPort = document.createElement('div');
    inputPort.className = 'node-port input';
    inputPort.dataset.nodeId = node.id;
    inputPort.dataset.portType = 'input';
    inputPort.addEventListener('click', (e) => handlePortClick(e, node.id, 'input'));

    const outputPort = document.createElement('div');
    outputPort.className = 'node-port output';
    outputPort.dataset.nodeId = node.id;
    outputPort.dataset.portType = 'output';
    outputPort.addEventListener('click', (e) => handlePortClick(e, node.id, 'output'));

    nodeEl.appendChild(header);
    nodeEl.appendChild(configSection);
    nodeEl.appendChild(inputPort);
    nodeEl.appendChild(outputPort);

    // Wave 5.5: Breakpoint indicator
    if (node.breakpoint) {
        const breakpointIndicator = document.createElement('div');
        breakpointIndicator.className = 'breakpoint-indicator';
        breakpointIndicator.title = 'Breakpoint set - click to remove';
        breakpointIndicator.onclick = (e) => {
            e.stopPropagation();
            node.breakpoint = false;
            renderWorkflow();
            showToast('Breakpoint removed', 'info');
        };
        nodeEl.appendChild(breakpointIndicator);
        nodeEl.classList.add('has-breakpoint');
    }

    // Wave 5.5: Execution state indicator
    if (node.executionState) {
        nodeEl.classList.add(`execution-${node.executionState}`);
        if (node.executionState === 'error' && node.executionError) {
            nodeEl.title = `Error: ${node.executionError}`;
        }
    }

    // Make draggable
    makeNodeDraggable(nodeEl, node);

    // Click to select
    nodeEl.addEventListener('click', (e) => {
        e.stopPropagation();
        selectNode(node.id);
    });

    // Add to canvas temporarily
    document.getElementById('canvas').appendChild(nodeEl);

    return nodeEl;
}

/**
 * Create a group from selected nodes (G shortcut)
 * If nodes are selected, creates a group containing them.
 * If no nodes selected, creates an empty group at canvas center.
 */
function createGroupFromSelection() {
    const selectedNodes = nodes.filter(n => n.selected);
    const canvas = document.getElementById('canvas');
    const canvasRect = canvas.getBoundingClientRect();

    // Create group node
    const groupId = generateNodeId();
    const groupDef = agentDefinitions.group;

    let groupX, groupY, groupWidth, groupHeight;
    const padding = 40;

    if (selectedNodes.length > 0) {
        // Calculate bounding box of selected nodes
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        selectedNodes.forEach(n => {
            const nodeEl = document.querySelector(`[data-node-id="${n.id}"]`);
            const nodeWidth = nodeEl ? nodeEl.offsetWidth : 200;
            const nodeHeight = nodeEl ? nodeEl.offsetHeight : 100;

            minX = Math.min(minX, n.x);
            minY = Math.min(minY, n.y);
            maxX = Math.max(maxX, n.x + nodeWidth);
            maxY = Math.max(maxY, n.y + nodeHeight);
        });

        // Add padding around the group
        groupX = minX - padding;
        groupY = minY - padding - 40; // Extra space for header
        groupWidth = Math.max(groupDef.minWidth, maxX - minX + padding * 2);
        groupHeight = Math.max(groupDef.minHeight, maxY - minY + padding * 2 + 40);
    } else {
        // Create empty group at canvas center
        groupX = (canvasRect.width / 2 - groupDef.minWidth / 2) / zoomLevel - panOffsetX;
        groupY = (canvasRect.height / 2 - groupDef.minHeight / 2) / zoomLevel - panOffsetY;
        groupWidth = groupDef.minWidth;
        groupHeight = groupDef.minHeight;
    }

    // Create group node object
    const groupNode = {
        id: groupId,
        type: 'group',
        agent: 'group',
        definition: groupDef,
        x: groupX,
        y: groupY,
        width: groupWidth,
        height: groupHeight,
        config: {
            label: selectedNodes.length > 0 ? `Group (${selectedNodes.length} nodes)` : 'New Group',
            description: '',
            collapsed: false,
            color: 'gray'
        },
        children: [],
        selected: false
    };

    // Add to nodes array
    nodes.push(groupNode);

    // Add selected nodes as children
    if (selectedNodes.length > 0) {
        selectedNodes.forEach(child => {
            // Convert to relative positioning
            child.relativeX = child.x - groupX;
            child.relativeY = child.y - groupY;
            child.parentGroup = groupId;
            groupNode.children.push(child.id);

            // Deselect the child node
            child.selected = false;
            const childEl = document.querySelector(`[data-node-id="${child.id}"]`);
            if (childEl) {
                childEl.classList.remove('selected');
            }
        });
    }

    // Save undo state
    saveUndoState('add_node', `Created group${selectedNodes.length > 0 ? ` with ${selectedNodes.length} nodes` : ''}`);

    // Render the group
    renderGroupNode(groupNode);

    // Select the new group
    selectNode(groupId);

    console.log(`Created group ${groupId} with ${selectedNodes.length} children`);
}

function createConfigInput(node, key, configDef) {
    const container = document.createElement('div');
    container.style.marginBottom = '8px';

    const label = document.createElement('label');
    label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    container.appendChild(label);

    let input;
    if (configDef.type === 'select') {
        input = document.createElement('select');
        configDef.options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt;
            option.textContent = opt;
            option.selected = opt === node.config[key];
            input.appendChild(option);
        });
    } else if (configDef.type === 'boolean') {
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = node.config[key];
        input.style.width = 'auto';
    } else if (configDef.type === 'number') {
        input = document.createElement('input');
        input.type = 'number';
        input.value = node.config[key];
        if (configDef.min !== undefined) input.min = configDef.min;
        if (configDef.max !== undefined) input.max = configDef.max;
        if (configDef.step !== undefined) input.step = configDef.step;
    } else {
        input = document.createElement('input');
        input.type = 'text';
        input.value = node.config[key];
    }

    input.addEventListener('change', (e) => {
        node.config[key] = configDef.type === 'boolean' ? e.target.checked :
                          configDef.type === 'number' ? parseFloat(e.target.value) :
                          e.target.value;
        updateProperties();
    });

    container.appendChild(input);
    return container;
}

function makeNodeDraggable(element, node) {
    let isDragging = false;
    let startX, startY, initialX, initialY;
    let isTouch = false;

    // Mouse drag handlers
    element.addEventListener('mousedown', (e) => {
        if (e.target.closest('.node-delete') || e.target.closest('.node-port') ||
            e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
            return;
        }

        isDragging = true;
        isTouch = false;
        startX = e.clientX;
        startY = e.clientY;
        initialX = node.x;
        initialY = node.y;
        element.classList.add('dragging');

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging || isTouch) return;

        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        node.x = initialX + dx;
        node.y = initialY + dy;

        element.style.left = node.x + 'px';
        element.style.top = node.y + 'px';

        updateConnections();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging && !isTouch) {
            isDragging = false;
            element.classList.remove('dragging');
        }
    });

    // Touch drag handlers for mobile devices
    element.addEventListener('touchstart', (e) => {
        // Ignore if on interactive elements
        if (e.target.closest('.node-delete') || e.target.closest('.node-port') ||
            e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
            return;
        }

        // Only handle single touch
        if (e.touches.length !== 1) return;

        isDragging = true;
        isTouch = true;
        const touch = e.touches[0];
        startX = touch.clientX;
        startY = touch.clientY;
        initialX = node.x;
        initialY = node.y;
        element.classList.add('dragging');

        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchmove', (e) => {
        if (!isDragging || !isTouch) return;

        if (e.touches.length !== 1) {
            isDragging = false;
            return;
        }

        const touch = e.touches[0];
        const dx = touch.clientX - startX;
        const dy = touch.clientY - startY;

        node.x = initialX + dx;
        node.y = initialY + dy;

        element.style.left = node.x + 'px';
        element.style.top = node.y + 'px';

        updateConnections();
        e.preventDefault();
    }, { passive: false });

    document.addEventListener('touchend', () => {
        if (isDragging && isTouch) {
            isDragging = false;
            isTouch = false;
            element.classList.remove('dragging');
        }
    });

    // Handle touch cancel (e.g., swipe interruption)
    document.addEventListener('touchcancel', () => {
        if (isDragging && isTouch) {
            isDragging = false;
            isTouch = false;
            element.classList.remove('dragging');
        }
    });
}

function handlePortClick(e, nodeId, portType) {
    e.stopPropagation();

    if (connectionStart === null) {
        // Start connection
        connectionStart = { nodeId, portType };
        showToast('Click output port to complete connection', 'success');
    } else {
        // Complete connection
        if (connectionStart.portType === 'output' && portType === 'input') {
            createConnection(connectionStart.nodeId, nodeId);
        } else if (connectionStart.portType === 'input' && portType === 'output') {
            createConnection(nodeId, connectionStart.nodeId);
        } else {
            showToast('Connect output to input', 'error');
        }
        connectionStart = null;
    }
}

function createConnection(fromNodeId, toNodeId) {
    // Prevent duplicate connections
    const exists = connections.find(c => c.from === fromNodeId && c.to === toNodeId);
    if (exists) {
        showToast('Connection already exists', 'error');
        return;
    }

    // Prevent self-connections
    if (fromNodeId === toNodeId) {
        showToast('Cannot connect node to itself', 'error');
        return;
    }

    connections.push({
        id: `conn-${connections.length + 1}`,
        from: fromNodeId,
        to: toNodeId
    });

    // Get node names for description
    const fromNode = nodes.find(n => n.id === fromNodeId);
    const toNode = nodes.find(n => n.id === toNodeId);
    const fromName = fromNode?.definition?.name || 'node';
    const toName = toNode?.definition?.name || 'node';

    saveUndoState('connect', `Connected ${fromName} → ${toName}`);
    updateConnections();
    showToast('Connection created', 'success');
}

function updateConnections() {
    const svg = document.getElementById('connectionsLayer');
    const canvas = document.getElementById('canvas');

    // Resize SVG to match canvas
    const rect = canvas.getBoundingClientRect();
    svg.setAttribute('width', rect.width);
    svg.setAttribute('height', rect.height);

    // Clear existing paths
    svg.querySelectorAll('path').forEach(p => p.remove());

    connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);
        if (!fromNode || !toNode) return;

        const fromEl = document.getElementById(conn.from);
        const toEl = document.getElementById(conn.to);
        if (!fromEl || !toEl) return;

        // Calculate port positions
        const fromX = fromNode.x + fromEl.offsetWidth;
        const fromY = fromNode.y + fromEl.offsetHeight / 2;
        const toX = toNode.x;
        const toY = toNode.y + toEl.offsetHeight / 2;

        // Create curved path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const dx = toX - fromX;
        const controlX1 = fromX + Math.abs(dx) * 0.5;
        const controlX2 = toX - Math.abs(dx) * 0.5;

        const d = `M ${fromX} ${fromY} C ${controlX1} ${fromY}, ${controlX2} ${toY}, ${toX} ${toY}`;

        path.setAttribute('d', d);
        path.setAttribute('class', 'connection-line');
        path.setAttribute('marker-end', 'url(#arrowhead)');
        path.dataset.connectionId = conn.id;

        path.addEventListener('click', () => deleteConnection(conn.id));

        svg.appendChild(path);
    });
}

// Alias for compatibility and handles group connection routing
function renderConnections() {
    const svg = document.getElementById('connectionsLayer');
    const canvas = document.getElementById('canvas');

    // Resize SVG to match canvas
    const rect = canvas.getBoundingClientRect();
    svg.setAttribute('width', rect.width);
    svg.setAttribute('height', rect.height);

    // Clear existing paths
    svg.querySelectorAll('path').forEach(p => p.remove());

    connections.forEach(conn => {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);
        if (!fromNode || !toNode) return;

        // Check if nodes are inside groups
        const fromGroup = fromNode.parentGroup ? nodes.find(n => n.id === fromNode.parentGroup) : null;
        const toGroup = toNode.parentGroup ? nodes.find(n => n.id === toNode.parentGroup) : null;

        // Skip rendering if node is in a collapsed group (unless both ends are in the same collapsed group)
        const fromCollapsed = fromGroup && fromGroup.config.collapsed;
        const toCollapsed = toGroup && toGroup.config.collapsed;

        // If both are in the same collapsed group, skip (internal connection hidden)
        if (fromCollapsed && toCollapsed && fromGroup.id === toGroup.id) {
            return;
        }

        // Get source element - use group if collapsed, otherwise use actual node
        let fromEl, toEl;
        let sourceNode = fromNode;
        let targetNode = toNode;

        if (fromCollapsed) {
            fromEl = document.getElementById(fromGroup.id);
            sourceNode = fromGroup;
        } else {
            fromEl = document.getElementById(conn.from);
        }

        if (toCollapsed) {
            toEl = document.getElementById(toGroup.id);
            targetNode = toGroup;
        } else {
            toEl = document.getElementById(conn.to);
        }

        if (!fromEl || !toEl) return;

        // Calculate port positions
        let fromX, fromY, toX, toY;

        // For group nodes, use the group's position + size
        if (fromCollapsed) {
            fromX = fromGroup.x + (fromGroup.width || fromEl.offsetWidth);
            fromY = fromGroup.y + (fromGroup.height || fromEl.offsetHeight) / 2;
        } else {
            fromX = fromNode.x + fromEl.offsetWidth;
            fromY = fromNode.y + fromEl.offsetHeight / 2;
        }

        if (toCollapsed) {
            toX = toGroup.x;
            toY = toGroup.y + (toGroup.height || toEl.offsetHeight) / 2;
        } else {
            toX = toNode.x;
            toY = toNode.y + toEl.offsetHeight / 2;
        }

        // Create curved path
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const dx = toX - fromX;
        const controlX1 = fromX + Math.abs(dx) * 0.5;
        const controlX2 = toX - Math.abs(dx) * 0.5;

        const d = `M ${fromX} ${fromY} C ${controlX1} ${fromY}, ${controlX2} ${toY}, ${toX} ${toY}`;

        path.setAttribute('d', d);

        // Style differently if crossing a group boundary
        const crossesGroup = (fromGroup && !toGroup) || (!fromGroup && toGroup) ||
                            (fromGroup && toGroup && fromGroup.id !== toGroup.id);

        if (crossesGroup) {
            path.setAttribute('class', 'connection-line group-crossing');
        } else {
            path.setAttribute('class', 'connection-line');
        }

        path.setAttribute('marker-end', 'url(#arrowhead)');
        path.dataset.connectionId = conn.id;

        // Store original endpoints for reference
        path.dataset.originalFrom = conn.from;
        path.dataset.originalTo = conn.to;

        path.addEventListener('click', () => deleteConnection(conn.id));

        svg.appendChild(path);
    });
}

function selectNode(nodeId) {
    deselectAll();
    selectedNode = nodeId;

    const nodeEl = document.getElementById(nodeId);
    if (nodeEl) {
        nodeEl.classList.add('selected');
    }

    updateProperties();
}

function deselectAll() {
    document.querySelectorAll('.workflow-node').forEach(n => n.classList.remove('selected'));
    selectedNode = null;
    updateProperties();
}

function updateProperties() {
    const panel = document.getElementById('propertiesContent');

    if (!selectedNode) {
        panel.innerHTML = `
            <div style="text-align: center; padding: 40px 20px; color: #999;">
                <div style="font-size: 48px; margin-bottom: 10px;">👈</div>
                <div>Select an agent to view properties</div>
            </div>
        `;
        return;
    }

    const node = nodes.find(n => n.id === selectedNode);
    if (!node) return;

    const def = node.definition;
    let html = `
        <div class="property-section">
            <div class="property-label">Agent Type</div>
            <div class="property-value" style="font-weight: 600; color: ${def.color};">${def.name}</div>
        </div>

        <div class="property-section">
            <div class="property-label">Node ID</div>
            <div class="property-value" style="font-family: monospace; font-size: 11px;">${node.id}</div>
        </div>

        <div class="property-section">
            <div class="property-label">Inputs</div>
            <div class="property-value" style="font-size: 11px;">${def.inputs.join(', ')}</div>
        </div>

        <div class="property-section">
            <div class="property-label">Outputs</div>
            <div class="property-value" style="font-size: 11px;">${def.outputs.join(', ')}</div>
        </div>

        <div style="border-top: 1px solid #e0e0e0; margin: 15px 0;"></div>

        <div class="property-section">
            <div class="property-label" style="font-size: 14px; font-weight: 600; margin-bottom: 15px;">⚙️ Configuration</div>
        </div>
    `;

    // Create editable form controls based on config definition
    Object.entries(def.config).forEach(([key, configDef]) => {
        const currentValue = node.config[key];
        const inputId = `config-${node.id}-${key}`;

        html += `<div class="property-section" style="margin-bottom: 15px;">
            <label for="${inputId}" style="display: block; font-size: 12px; font-weight: 500; margin-bottom: 5px; color: #333;">
                ${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </label>`;

        // Generate appropriate input based on type
        switch (configDef.type) {
            case 'text':
                html += `<input
                    type="text"
                    id="${inputId}"
                    class="property-input"
                    value="${currentValue || ''}"
                    placeholder="${configDef.placeholder || ''}"
                    onchange="updateNodeConfig('${node.id}', '${key}', this.value)"
                />`;
                break;

            case 'textarea':
                html += `<textarea
                    id="${inputId}"
                    class="property-textarea"
                    placeholder="${configDef.placeholder || ''}"
                    onchange="updateNodeConfig('${node.id}', '${key}', this.value)"
                    rows="4"
                >${currentValue || ''}</textarea>`;
                break;

            case 'number':
                html += `<input
                    type="number"
                    id="${inputId}"
                    class="property-input"
                    value="${currentValue}"
                    min="${configDef.min || ''}"
                    max="${configDef.max || ''}"
                    step="${configDef.step || 1}"
                    onchange="updateNodeConfig('${node.id}', '${key}', parseFloat(this.value))"
                />`;
                break;

            case 'select':
                html += `<select
                    id="${inputId}"
                    class="property-select"
                    onchange="updateNodeConfig('${node.id}', '${key}', this.value)"
                >`;
                configDef.options.forEach(option => {
                    html += `<option value="${option}" ${currentValue === option ? 'selected' : ''}>${option}</option>`;
                });
                html += `</select>`;
                break;

            case 'boolean':
                html += `<label style="display: flex; align-items: center; cursor: pointer;">
                    <input
                        type="checkbox"
                        id="${inputId}"
                        ${currentValue ? 'checked' : ''}
                        onchange="updateNodeConfig('${node.id}', '${key}', this.checked)"
                        style="margin-right: 8px; cursor: pointer;"
                    />
                    <span style="font-size: 12px; color: #666;">Enable ${key.replace(/_/g, ' ')}</span>
                </label>`;
                break;

            default:
                html += `<input
                    type="text"
                    id="${inputId}"
                    class="property-input"
                    value="${currentValue || ''}"
                    onchange="updateNodeConfig('${node.id}', '${key}', this.value)"
                />`;
        }

        html += `</div>`;
    });

    panel.innerHTML = html;
}

function updateNodeConfig(nodeId, key, value) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    node.config[key] = value;
    showToast(`Updated ${key}`, 'success');
    saveUndoState('update_config', `Updated ${key} on ${node.definition?.name || 'node'}`);
}

function deleteNode(nodeId) {
    if (!confirm('Delete this node?')) return;

    // Get node name before deletion
    const node = nodes.find(n => n.id === nodeId);
    const nodeName = node?.definition?.name || 'node';

    // Save state BEFORE deletion
    saveUndoState('delete_node', `Deleted ${nodeName}`);

    // Remove connections
    connections = connections.filter(c => c.from !== nodeId && c.to !== nodeId);

    // Remove node
    nodes = nodes.filter(n => n.id !== nodeId);

    const nodeEl = document.getElementById(nodeId);
    if (nodeEl) nodeEl.remove();

    updateConnections();
    if (selectedNode === nodeId) {
        deselectAll();
    }

    showToast('Node deleted', 'success');
}

function deleteConnection(connId) {
    // Save state BEFORE deletion
    saveUndoState('disconnect', 'Deleted connection');

    connections = connections.filter(c => c.id !== connId);
    updateConnections();
    showToast('Connection deleted', 'success');
}

function clearCanvas() {
    if (!confirm('Clear all nodes and connections?')) return;

    // Save state BEFORE clearing
    saveUndoState('clear', `Cleared ${nodes.length} nodes and ${connections.length} connections`);

    nodes = [];
    connections = [];
    selectedNode = null;

    document.getElementById('canvas').innerHTML = '';
    updateConnections();
    deselectAll();
    showToast('Canvas cleared', 'success');
}

// Clear canvas without confirmation (for internal use by undo/redo)
function clearCanvasOnly() {
    document.getElementById('canvas').innerHTML = '';
    updateConnections();
}

function exportWorkflow(format = 'json') {
    const workflowName = document.querySelector('.workflow-title').textContent || 'Untitled Workflow';

    switch (format) {
        case 'python':
            exportAsPython(workflowName);
            break;
        case 'yaml':
            exportAsYAML(workflowName);
            break;
        case 'json':
        default:
            exportAsJSON(workflowName);
            break;
    }
}

function exportAsJSON(workflowName) {
    const workflow = {
        version: '1.0',
        name: workflowName,
        nodes: nodes.map(n => ({
            id: n.id,
            agentType: n.agentType,
            x: n.x,
            y: n.y,
            config: n.config
        })),
        connections: connections
    };

    const json = JSON.stringify(workflow, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow-${Date.now()}.json`;
    a.click();

    showToast('Workflow exported as JSON', 'success');
}

/**
 * Export workflow as executable Python code
 */
function exportAsPython(workflowName) {
    const sortedNodes = topologicalSort();
    if (!sortedNodes) {
        showToast('Cannot export: workflow contains cycles', 'error');
        return;
    }

    // Generate Python code
    let code = `"""
HoloLoom Workflow: ${workflowName}
Generated: ${new Date().toISOString()}

This script executes a HoloLoom workflow pipeline.
"""

import asyncio
from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode


async def run_workflow(input_query: str):
    """
    Execute the ${workflowName} workflow.

    Args:
        input_query: The initial query to process

    Returns:
        Final workflow result
    """
    config = Config.fused()
    results = {}

    async with HoloLoom(config=config) as loom:
`;

    // Generate code for each node in topological order
    sortedNodes.forEach((node, index) => {
        const indent = '        ';
        const varName = `result_${node.id}`;
        const inputSources = getNodeInputSources(node.id);

        code += `\n${indent}# Node ${index + 1}: ${node.definition?.name || node.agentType}\n`;

        switch (node.agentType) {
            case 'hololoom':
                const pattern = node.config?.pattern || 'fast';
                const inputVar = inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query';
                code += `${indent}${varName} = await loom.query(\n`;
                code += `${indent}    ${inputVar},\n`;
                code += `${indent}    pattern="${pattern}"\n`;
                code += `${indent})\n`;
                break;

            case 'search':
                const maxResults = node.config?.max_results || 10;
                const threshold = node.config?.similarity_threshold || 0.7;
                code += `${indent}${varName} = await loom.recall(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    k=${maxResults},\n`;
                code += `${indent}    threshold=${threshold}\n`;
                code += `${indent})\n`;
                break;

            case 'multiquery':
                const mode = node.config?.mode || 'research';
                const maxSub = node.config?.max_subqueries || 5;
                code += `${indent}orchestrator = await create_agentic_orchestrator(config)\n`;
                code += `${indent}${varName} = await orchestrator.reason(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    mode=ReasoningMode.${mode.toUpperCase()},\n`;
                code += `${indent}    max_steps=${maxSub}\n`;
                code += `${indent})\n`;
                break;

            case 'embedder':
                const scales = node.config?.scales || '96,192,384';
                code += `${indent}${varName} = await loom.embed(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    scales=[${scales}]\n`;
                code += `${indent})\n`;
                break;

            case 'synthesizer':
                code += `${indent}${varName} = await loom.synthesize(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    extract_entities=${node.config?.extract_entities !== false},\n`;
                code += `${indent}    extract_motifs=${node.config?.extract_motifs !== false}\n`;
                code += `${indent})\n`;
                break;

            case 'refiner':
                const strategy = node.config?.strategy || 'refine';
                code += `${indent}${varName} = await loom.refine(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    strategy="${strategy}"\n`;
                code += `${indent})\n`;
                break;

            case 'memory_store':
                code += `${indent}${varName} = await loom.experience(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'}\n`;
                code += `${indent})\n`;
                break;

            case 'context_retriever':
                code += `${indent}${varName} = await loom.recall(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    k=${node.config?.max_context || 20}\n`;
                code += `${indent})\n`;
                break;

            case 'knowledge_fusion':
                code += `${indent}${varName} = await loom.fuse_knowledge(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    max_hops=${node.config?.max_hops || 3}\n`;
                code += `${indent})\n`;
                break;

            case 'thompson_sampler':
                code += `${indent}# Thompson Sampling decision\n`;
                code += `${indent}${varName} = loom.policy.thompson_sample(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'}\n`;
                code += `${indent})\n`;
                break;

            case 'convergence_engine':
                const strategy2 = node.config?.strategy || 'bayesian_blend';
                code += `${indent}${varName} = loom.convergence.collapse(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    strategy="${strategy2}"\n`;
                code += `${indent})\n`;
                break;

            case 'safety_guardrails':
                code += `${indent}from HoloLoom.alignment import SafetyGuardrails\n`;
                code += `${indent}guardrails = SafetyGuardrails()\n`;
                code += `${indent}${varName} = await guardrails.gate_action(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'}\n`;
                code += `${indent})\n`;
                break;

            case 'response_generator':
                code += `${indent}${varName} = await loom.generate_response(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'}\n`;
                code += `${indent})\n`;
                break;

            case 'format_converter':
                const format = node.config?.output_format || 'markdown';
                code += `${indent}${varName} = loom.format(\n`;
                code += `${indent}    ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'},\n`;
                code += `${indent}    format="${format}"\n`;
                code += `${indent})\n`;
                break;

            default:
                code += `${indent}# Custom node: ${node.agentType}\n`;
                code += `${indent}${varName} = ${inputSources.length > 0 ? `results['${inputSources[0]}']` : 'input_query'}\n`;
        }

        code += `${indent}results['${node.id}'] = ${varName}\n`;
    });

    // Return the final result
    const lastNode = sortedNodes[sortedNodes.length - 1];
    code += `\n        return results['${lastNode.id}']\n\n`;

    // Add main entry point
    code += `
if __name__ == "__main__":
    # Example usage
    query = "What is Thompson Sampling?"
    result = asyncio.run(run_workflow(query))
    print(f"Result: {result}")
`;

    // Download the file
    const blob = new Blob([code], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow_${workflowName.toLowerCase().replace(/\s+/g, '_')}.py`;
    a.click();

    showToast('Workflow exported as Python', 'success');
}

/**
 * Export workflow as YAML configuration
 */
function exportAsYAML(workflowName) {
    let yaml = `# HoloLoom Workflow Configuration
# Generated: ${new Date().toISOString()}

version: "1.0"
name: "${workflowName}"

nodes:
`;

    // Export nodes
    nodes.forEach(node => {
        yaml += `  - id: "${node.id}"
    type: ${node.agentType}
    name: "${node.definition?.name || node.agentType}"
    position:
      x: ${node.x}
      y: ${node.y}
`;

        // Add config if present
        if (node.config && Object.keys(node.config).length > 0) {
            yaml += `    config:\n`;
            Object.entries(node.config).forEach(([key, value]) => {
                if (typeof value === 'string') {
                    yaml += `      ${key}: "${value}"\n`;
                } else if (typeof value === 'boolean') {
                    yaml += `      ${key}: ${value}\n`;
                } else {
                    yaml += `      ${key}: ${value}\n`;
                }
            });
        }
        yaml += '\n';
    });

    // Export connections
    yaml += `connections:\n`;
    connections.forEach(conn => {
        yaml += `  - from: "${conn.from}"
    to: "${conn.to}"
`;
    });

    // Add execution metadata
    yaml += `
# Execution settings
execution:
  mode: sequential
  timeout_seconds: 300
  retry_on_failure: true
  max_retries: 3
`;

    // Download the file
    const blob = new Blob([yaml], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow_${workflowName.toLowerCase().replace(/\s+/g, '_')}.yaml`;
    a.click();

    showToast('Workflow exported as YAML', 'success');
}

/**
 * Topological sort for workflow nodes (for Python code generation)
 */
function topologicalSort() {
    const sorted = [];
    const visited = new Set();
    const visiting = new Set();

    function visit(nodeId) {
        if (visiting.has(nodeId)) {
            return false; // Cycle detected
        }
        if (visited.has(nodeId)) {
            return true;
        }

        visiting.add(nodeId);

        // Find all nodes that this node depends on (incoming connections)
        const dependencies = connections
            .filter(c => c.to === nodeId)
            .map(c => c.from);

        for (const depId of dependencies) {
            if (!visit(depId)) {
                return false;
            }
        }

        visiting.delete(nodeId);
        visited.add(nodeId);

        const node = nodes.find(n => n.id === nodeId);
        if (node) sorted.push(node);

        return true;
    }

    // Visit all nodes
    for (const node of nodes) {
        if (!visit(node.id)) {
            return null; // Cycle detected
        }
    }

    return sorted;
}

/**
 * Get input sources for a node (nodes connected to its inputs)
 */
function getNodeInputSources(nodeId) {
    return connections
        .filter(c => c.to === nodeId)
        .map(c => c.from);
}

/**
 * Show export format selector modal
 */
function showExportModal() {
    const existingModal = document.getElementById('exportModal');
    if (existingModal) {
        existingModal.classList.add('show');
        return;
    }

    const modal = document.createElement('div');
    modal.id = 'exportModal';
    modal.className = 'modal show';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 400px;">
            <div class="modal-header">
                <h3>Export Workflow</h3>
                <button class="modal-close" onclick="closeExportModal()">&times;</button>
            </div>
            <div class="modal-body" style="padding: 20px;">
                <p style="margin-bottom: 15px; color: #666;">Choose export format:</p>
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <button class="export-option-btn" onclick="exportWorkflow('json'); closeExportModal();">
                        <span style="font-size: 24px;">📄</span>
                        <div>
                            <strong>JSON</strong>
                            <small>Workflow definition (import/export)</small>
                        </div>
                    </button>
                    <button class="export-option-btn" onclick="exportWorkflow('python'); closeExportModal();">
                        <span style="font-size: 24px;">🐍</span>
                        <div>
                            <strong>Python</strong>
                            <small>Executable Python script</small>
                        </div>
                    </button>
                    <button class="export-option-btn" onclick="exportWorkflow('yaml'); closeExportModal();">
                        <span style="font-size: 24px;">📝</span>
                        <div>
                            <strong>YAML</strong>
                            <small>Configuration file format</small>
                        </div>
                    </button>
                </div>
            </div>
        </div>
    `;

    // Add styles for export buttons
    const style = document.createElement('style');
    style.textContent = `
        .export-option-btn {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }
        .export-option-btn:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .export-option-btn strong {
            display: block;
            color: #333;
        }
        .export-option-btn small {
            color: #666;
            font-size: 12px;
        }
    `;
    document.head.appendChild(style);

    document.body.appendChild(modal);
}

function closeExportModal() {
    const modal = document.getElementById('exportModal');
    if (modal) {
        modal.classList.remove('show');
    }
}

function importWorkflow() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';

    input.onchange = (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();

        reader.onload = (event) => {
            try {
                const workflow = JSON.parse(event.target.result);
                loadWorkflow(workflow);
                showToast('Workflow imported', 'success');
            } catch (err) {
                showToast('Invalid workflow file', 'error');
                console.error(err);
            }
        };

        reader.readAsText(file);
    };

    input.click();
}

function loadWorkflow(workflow) {
    // Save state BEFORE import
    saveUndoState('import', `Imported workflow: ${workflow.name || 'Untitled'}`);

    // Clear without confirmation (internal use)
    nodes = [];
    connections = [];
    selectedNode = null;
    clearCanvasOnly();
    deselectAll();

    document.querySelector('.workflow-title').textContent = workflow.name || 'Imported Workflow';

    workflow.nodes.forEach(nodeData => {
        const node = {
            id: nodeData.id,
            agentType: nodeData.agentType,
            x: nodeData.x,
            y: nodeData.y,
            config: nodeData.config,
            definition: agentDefinitions[nodeData.agentType]
        };
        nodes.push(node);
        renderNode(node);
    });

    connections = workflow.connections || [];
    updateConnections();
}

async function executeWorkflow() {
    if (nodes.length === 0) {
        showToast('No nodes to execute', 'error');
        return;
    }

    // Find starting nodes (nodes with no incoming connections)
    const startNodes = nodes.filter(node =>
        !connections.some(c => c.to === node.id)
    );

    if (startNodes.length === 0) {
        showToast('No starting nodes found (all nodes have inputs)', 'error');
        return;
    }

    executionState.running = true;
    executionState.results = {};
    updateExecutionStatus('running', 'Executing workflow...');

    const log = document.getElementById('executionLog');
    log.innerHTML = '';

    try {
        // Execute nodes in topological order
        const executedNodes = new Set();
        const queue = [...startNodes];

        while (queue.length > 0) {
            const node = queue.shift();
            if (executedNodes.has(node.id)) continue;

            // Check if all dependencies are executed
            const dependencies = connections
                .filter(c => c.to === node.id)
                .map(c => c.from);

            const allDepsExecuted = dependencies.every(dep => executedNodes.has(dep));
            if (!allDepsExecuted && dependencies.length > 0) {
                queue.push(node); // Re-queue
                continue;
            }

            // Execute node
            await executeNode(node);
            executedNodes.add(node.id);

            // Add next nodes to queue
            const nextNodes = connections
                .filter(c => c.from === node.id)
                .map(c => nodes.find(n => n.id === c.to))
                .filter(n => n);

            queue.push(...nextNodes);
        }

        updateExecutionStatus('success', `Executed ${executedNodes.size} nodes`);
        showToast('Workflow execution complete', 'success');

    } catch (err) {
        updateExecutionStatus('error', `Error: ${err.message}`);
        showToast('Workflow execution failed', 'error');
        console.error(err);
    } finally {
        executionState.running = false;
    }
}

async function executeNode(node) {
    const def = node.definition;

    addLog(`Executing ${def.name} (${node.id})...`);

    // Simulate execution (in real implementation, this would call actual agents)
    await new Promise(resolve => setTimeout(resolve, 500));

    // Mock result based on agent type
    const result = {
        nodeId: node.id,
        agentType: node.agentType,
        timestamp: new Date().toISOString(),
        config: node.config,
        output: generateMockOutput(node)
    };

    executionState.results[node.id] = result;
    addLog(`✓ ${def.name} completed`);

    return result;
}

function generateMockOutput(node) {
    const def = node.definition;

    switch (def.type) {
        case 'query':
            return { query: 'Sample query', confidence: 0.85 };
        case 'process':
            return { processed: true, embeddings: [0.1, 0.2, 0.3] };
        case 'memory':
            return { stored: true, count: 5 };
        case 'decision':
            return { decision: 'option_1', confidence: 0.9 };
        case 'output':
            return { response: 'Generated response' };
        case 'control':
            return { branch: 'true' };
        default:
            return { status: 'success' };
    }
}

function updateExecutionStatus(status, text) {
    const dot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    dot.className = 'status-dot';
    if (status) dot.classList.add(status);
    statusText.textContent = text;
}

function addLog(message) {
    const log = document.getElementById('executionLog');
    const entry = document.createElement('div');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    entry.style.marginBottom = '4px';
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function getAgentIcon(type) {
    const icons = {
        query: '🔍',
        process: '⚙️',
        memory: '💾',
        decision: '🎯',
        output: '📤',
        control: '🔀',
        llm: '🧠',
        tool: '🔧',
        group: '📁',
        composite: '📦'
    };
    return icons[type] || '📌';
}

function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Don't handle hotkeys if typing in input/textarea
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
            return;
        }

        // Delete selected node
        if (e.key === 'Delete' && selectedNode) {
            deleteNode(selectedNode);
        }

        // Duplicate selected node (D)
        if (e.key === 'd' && selectedNode) {
            e.preventDefault();
            duplicateNode(selectedNode);
        }

        // Copy selected node (Ctrl+C)
        if ((e.ctrlKey || e.metaKey) && e.key === 'c' && selectedNode) {
            e.preventDefault();
            copyNode(selectedNode);
        }

        // Paste node (Ctrl+V)
        if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
            e.preventDefault();
            pasteNode();
        }

        // Select all (Ctrl+A)
        if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
            e.preventDefault();
            selectAllNodes();
        }

        // Undo (Ctrl+Z)
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            undo();
        }

        // Redo (Ctrl+Shift+Z or Ctrl+Y)
        if ((e.ctrlKey || e.metaKey) && (e.shiftKey && e.key === 'z' || e.key === 'y')) {
            e.preventDefault();
            redo();
        }

        // Export (Ctrl+S)
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            exportWorkflow();
        }

        // Execute (Ctrl+Enter or F5)
        if (((e.ctrlKey || e.metaKey) && e.key === 'Enter') || e.key === 'F5') {
            e.preventDefault();
            executeWorkflow();
        }

        // Templates (T)
        if (e.key === 't') {
            e.preventDefault();
            showTemplatesModal();
        }

        // Clear canvas (Ctrl+Shift+Delete)
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Delete') {
            e.preventDefault();
            if (confirm('Clear entire canvas?')) {
                clearCanvas();
            }
        }

        // Deselect (Escape)
        if (e.key === 'Escape') {
            deselectAll();
            connectionStart = null;
        }

        // Show hotkeys help (?)
        if (e.key === '?' && !e.shiftKey) {
            e.preventDefault();
            showHotkeysModal();
        }

        // Toggle history panel (H)
        if (e.key === 'h' || e.key === 'H') {
            e.preventDefault();
            toggleHistoryPanel();
        }

        // Create group from selected nodes (G)
        if (e.key === 'g' || e.key === 'G') {
            e.preventDefault();
            createGroupFromSelection();
        }
    });
}

// Modal Management Functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('show');
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('show');
    }
}

// Close modal when clicking outside
document.addEventListener('click', function(event) {
    const modals = document.querySelectorAll('.modal.show');
    modals.forEach(modal => {
        const modalContent = modal.querySelector('.modal-content');
        if (!modalContent.contains(event.target) && event.target.classList.contains('modal')) {
            modal.classList.remove('show');
        }
    });
});

// Version Control Functions
function showSaveVersionModal() {
    document.getElementById('commitMessage').value = '';
    document.getElementById('commitDescription').value = '';
    openModal('saveVersionModal');
}

function showCreateBranchModal() {
    document.getElementById('branchName').value = '';
    openModal('createBranchModal');
}

function showVersionHistoryModal() {
    loadVersionHistory();
    openModal('versionHistoryModal');
}

function showDiffModal() {
    loadDiffVersions();
    openModal('diffModal');
}

async function saveWorkflowVersion() {
    const message = document.getElementById('commitMessage').value.trim();
    if (!message) {
        showToast('Commit message is required', 'error');
        return;
    }

    const description = document.getElementById('commitDescription').value.trim();

    try {
        const workflow = {
            version: '1.0',
            name: document.querySelector('.workflow-title').textContent,
            nodes: nodes.map(n => ({
                id: n.id,
                agentType: n.agentType,
                x: n.x,
                y: n.y,
                config: n.config
            })),
            connections: connections
        };

        const response = await fetch('/api/workflow/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                workflow: workflow,
                message: message,
                description: description,
                branch: currentBranch,
                timestamp: new Date().toISOString()
            })
        });

        if (response.ok) {
            const result = await response.json();
            currentVersion = result.version;
            updateVersionIndicator();
            versionHistory.push({
                version: result.version,
                message: message,
                description: description,
                branch: currentBranch,
                timestamp: result.timestamp,
                workflow: workflow
            });

            closeModal('saveVersionModal');
            showToast(`Version v${result.version} saved successfully`, 'success');
        } else {
            showToast('Failed to save version', 'error');
        }
    } catch (error) {
        console.error('Save version error:', error);
        showToast('Error saving version: ' + error.message, 'error');
    }
}

async function createBranch() {
    const branchName = document.getElementById('branchName').value.trim();
    if (!branchName) {
        showToast('Branch name is required', 'error');
        return;
    }

    if (/^[a-zA-Z0-9._/-]+$/.test(branchName) === false) {
        showToast('Invalid branch name. Use letters, numbers, dots, hyphens, underscores, or slashes', 'error');
        return;
    }

    try {
        const response = await fetch('/api/workflow/branch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                branch_name: branchName,
                from_branch: currentBranch,
                from_version: currentVersion
            })
        });

        if (response.ok) {
            const result = await response.json();

            // Update local state
            if (!branches[branchName]) {
                branches[branchName] = {
                    versions: [],
                    head: currentVersion
                };
            }

            currentBranch = branchName;
            updateVersionIndicator();
            closeModal('createBranchModal');
            showToast(`Branch '${branchName}' created successfully`, 'success');
        } else {
            showToast('Failed to create branch', 'error');
        }
    } catch (error) {
        console.error('Create branch error:', error);
        showToast('Error creating branch: ' + error.message, 'error');
    }
}

async function loadVersionHistory() {
    try {
        const response = await fetch('/api/workflow/versions');
        if (response.ok) {
            const data = await response.json();
            versionHistory = data.versions;
        }
    } catch (error) {
        console.error('Load versions error:', error);
    }

    renderVersionHistory();
}

function renderVersionHistory() {
    const versionList = document.getElementById('versionList');
    versionList.innerHTML = '';

    if (versionHistory.length === 0) {
        versionList.innerHTML = '<li style="padding: 20px; text-align: center; color: #999;">No versions yet. Save a version to get started!</li>';
        return;
    }

    // Sort by version descending
    const sorted = [...versionHistory].sort((a, b) => b.version - a.version);

    sorted.forEach(version => {
        const li = document.createElement('li');
        li.className = 'version-item';
        if (version.version === currentVersion) {
            li.classList.add('active');
        }

        const timestamp = new Date(version.timestamp).toLocaleString();

        li.innerHTML = `
            <div class="version-number">v${version.version} ${version.message}</div>
            <div class="version-message">${version.description || 'No description'}</div>
            <div class="version-time">${timestamp}
                <span class="version-branch">${version.branch || 'main'}</span>
            </div>
        `;

        li.addEventListener('click', () => loadVersion(version));
        versionList.appendChild(li);
    });
}

async function loadVersion(version) {
    if (!confirm(`Load version v${version.version}? This will replace current workflow.`)) {
        return;
    }

    try {
        loadWorkflow(version.workflow);
        currentVersion = version.version;
        currentBranch = version.branch || 'main';
        updateVersionIndicator();
        closeModal('versionHistoryModal');
        showToast(`Loaded version v${version.version}`, 'success');
    } catch (error) {
        console.error('Load version error:', error);
        showToast('Error loading version', 'error');
    }
}

async function loadDiffVersions() {
    const fromSelect = document.getElementById('diffFromVersion');
    const toSelect = document.getElementById('diffToVersion');

    fromSelect.innerHTML = '<option>Select version...</option>';
    toSelect.innerHTML = '<option>Select version...</option>';

    const sorted = [...versionHistory].sort((a, b) => a.version - b.version);

    sorted.forEach(version => {
        const optionFrom = document.createElement('option');
        optionFrom.value = version.version;
        optionFrom.textContent = `v${version.version} - ${version.message}`;
        fromSelect.appendChild(optionFrom);

        const optionTo = document.createElement('option');
        optionTo.value = version.version;
        optionTo.textContent = `v${version.version} - ${version.message}`;
        toSelect.appendChild(optionTo);
    });
}

async function compareDiff() {
    const fromVersion = parseInt(document.getElementById('diffFromVersion').value);
    const toVersion = parseInt(document.getElementById('diffToVersion').value);

    if (!fromVersion || !toVersion) {
        showToast('Please select both versions', 'error');
        return;
    }

    if (fromVersion === toVersion) {
        showToast('Versions must be different', 'error');
        return;
    }

    try {
        const response = await fetch(`/api/workflow/diff?from=${fromVersion}&to=${toVersion}`);
        if (response.ok) {
            const diff = await response.json();
            renderDiff(diff);
        } else {
            showToast('Failed to compare versions', 'error');
        }
    } catch (error) {
        console.error('Diff error:', error);
        showToast('Error comparing versions', 'error');
    }
}

function renderDiff(diff) {
    const viewer = document.getElementById('diffViewer');
    viewer.style.display = 'block';
    viewer.innerHTML = '';

    if (diff.nodes_added) {
        const div = document.createElement('div');
        div.innerHTML = '<strong style="color: #333;">Nodes Added:</strong>';
        viewer.appendChild(div);

        diff.nodes_added.forEach(node => {
            const span = document.createElement('span');
            span.className = 'diff-added';
            span.textContent = `+ ${node.agentType} (${node.id})`;
            viewer.appendChild(span);
        });
    }

    if (diff.nodes_removed) {
        const div = document.createElement('div');
        div.innerHTML = '<strong style="color: #333; margin-top: 10px;">Nodes Removed:</strong>';
        viewer.appendChild(div);

        diff.nodes_removed.forEach(node => {
            const span = document.createElement('span');
            span.className = 'diff-removed';
            span.textContent = `- ${node.agentType} (${node.id})`;
            viewer.appendChild(span);
        });
    }

    if (diff.connections_changed) {
        const div = document.createElement('div');
        div.innerHTML = '<strong style="color: #333; margin-top: 10px;">Connections Changed:</strong>';
        viewer.appendChild(div);

        diff.connections_changed.forEach(conn => {
            const span = document.createElement('span');
            span.className = 'diff-unchanged';
            span.textContent = conn;
            viewer.appendChild(span);
        });
    }

    if (!diff.nodes_added && !diff.nodes_removed && !diff.connections_changed) {
        const div = document.createElement('div');
        div.style.textAlign = 'center';
        div.style.color = '#999';
        div.style.padding = '20px';
        div.textContent = 'No differences found';
        viewer.appendChild(div);
    }
}

function updateVersionIndicator() {
    document.getElementById('versionNumber').textContent = `v${currentVersion}`;
    document.getElementById('currentBranch').textContent = currentBranch;
}

// ========== ZOOM CONTROLS ==========
let zoomLevel = 1.0;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 3.0;
const ZOOM_STEP = 0.1;

// ========== PAN CONTROLS (Wave 6) ==========
let panOffsetX = 0;
let panOffsetY = 0;

function zoomIn() {
    if (zoomLevel < MAX_ZOOM) {
        zoomLevel = Math.min(MAX_ZOOM, zoomLevel + ZOOM_STEP);
        applyZoom();
    }
}

function zoomOut() {
    if (zoomLevel > MIN_ZOOM) {
        zoomLevel = Math.max(MIN_ZOOM, zoomLevel - ZOOM_STEP);
        applyZoom();
    }
}

function resetZoom() {
    zoomLevel = 1.0;
    applyZoom();
}

function applyZoom() {
    const canvas = document.getElementById('canvas');
    const connectionsLayer = document.getElementById('connectionsLayer');

    // Apply both pan and zoom transforms (Wave 6: pan support)
    const transform = `translate(${panOffsetX}px, ${panOffsetY}px) scale(${zoomLevel})`;
    canvas.style.transform = transform;
    canvas.style.transformOrigin = '0 0';

    connectionsLayer.style.transform = transform;
    connectionsLayer.style.transformOrigin = '0 0';

    // Update zoom indicator
    document.getElementById('zoomLevel').textContent = `${Math.round(zoomLevel * 100)}%`;

    // Redraw connections to match new scale
    renderConnections();
}

function resetPan() {
    panOffsetX = 0;
    panOffsetY = 0;
    applyZoom();
}

// ========== TOUCH GESTURE SYSTEM (Wave 6) ==========
let touchState = {
    touches: [],
    initialDistance: null,
    initialZoom: null,
    lastPanCenter: null,
    gestureType: null,  // 'drag', 'pan', 'pinch', 'longpress'
    longPressTimer: null,
    longPressTarget: null
};

const LONG_PRESS_MS = 500;
const PINCH_THRESHOLD = 10;  // Minimum distance change to trigger pinch

// Helper: Calculate distance between two touch points
function getTouchDistance(t1, t2) {
    const dx = t1.clientX - t2.clientX;
    const dy = t1.clientY - t2.clientY;
    return Math.sqrt(dx * dx + dy * dy);
}

// Helper: Calculate center point between two touches
function getTouchCenter(t1, t2) {
    return {
        x: (t1.clientX + t2.clientX) / 2,
        y: (t1.clientY + t2.clientY) / 2
    };
}

// Helper: Trigger haptic feedback if available
function triggerHapticFeedback(pattern = 50) {
    if ('vibrate' in navigator) {
        navigator.vibrate(pattern);
    }
}

// Helper: Create touch ripple effect
function createTouchRipple(x, y, parent) {
    const ripple = document.createElement('div');
    ripple.className = 'touch-ripple';
    const rect = parent.getBoundingClientRect();
    ripple.style.left = (x - rect.left) + 'px';
    ripple.style.top = (y - rect.top) + 'px';
    ripple.style.width = ripple.style.height = '20px';
    parent.style.position = 'relative';
    parent.appendChild(ripple);
    setTimeout(() => ripple.remove(), 400);
}

// Canvas touch event handlers for pinch-to-zoom and two-finger pan
function initCanvasTouchGestures() {
    const canvasContainer = document.getElementById('canvas-container');
    if (!canvasContainer) return;

    canvasContainer.addEventListener('touchstart', handleCanvasTouchStart, { passive: false });
    canvasContainer.addEventListener('touchmove', handleCanvasTouchMove, { passive: false });
    canvasContainer.addEventListener('touchend', handleCanvasTouchEnd, { passive: false });
    canvasContainer.addEventListener('touchcancel', handleCanvasTouchEnd, { passive: false });
}

function handleCanvasTouchStart(e) {
    const touches = e.touches;

    // Clear any existing long press timer
    if (touchState.longPressTimer) {
        clearTimeout(touchState.longPressTimer);
        touchState.longPressTimer = null;
    }

    if (touches.length === 2) {
        // Two fingers: prepare for pinch/pan gesture
        e.preventDefault();
        touchState.gestureType = 'pinch';
        touchState.initialDistance = getTouchDistance(touches[0], touches[1]);
        touchState.initialZoom = zoomLevel;
        touchState.lastPanCenter = getTouchCenter(touches[0], touches[1]);
    } else if (touches.length === 1) {
        // Check if touch is on a node (let existing node drag handle it)
        const target = e.target;
        if (target.closest('.workflow-node')) {
            // Let existing node drag handling take over
            return;
        }

        // Single finger on canvas - start long press detection
        touchState.gestureType = 'potential_longpress';
        touchState.longPressTarget = target;
        touchState.longPressTimer = setTimeout(() => {
            if (touchState.gestureType === 'potential_longpress') {
                touchState.gestureType = 'longpress';
                triggerHapticFeedback();
                showMobileContextSheet(e, target);
            }
        }, LONG_PRESS_MS);
    }
}

function handleCanvasTouchMove(e) {
    const touches = e.touches;

    // Cancel long press if finger moves
    if (touchState.longPressTimer) {
        clearTimeout(touchState.longPressTimer);
        touchState.longPressTimer = null;
    }

    if (touches.length === 2 && (touchState.gestureType === 'pinch' || touchState.gestureType === 'pan')) {
        e.preventDefault();

        // Pinch zoom calculation
        const currentDistance = getTouchDistance(touches[0], touches[1]);
        if (touchState.initialDistance && Math.abs(currentDistance - touchState.initialDistance) > PINCH_THRESHOLD) {
            const scale = currentDistance / touchState.initialDistance;
            const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, touchState.initialZoom * scale));

            if (Math.abs(newZoom - zoomLevel) > 0.01) {
                zoomLevel = newZoom;
                applyZoom();
            }
        }

        // Two-finger pan calculation
        const center = getTouchCenter(touches[0], touches[1]);
        if (touchState.lastPanCenter) {
            const dx = center.x - touchState.lastPanCenter.x;
            const dy = center.y - touchState.lastPanCenter.y;

            if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
                panOffsetX += dx;
                panOffsetY += dy;
                applyZoom();
            }
        }
        touchState.lastPanCenter = center;
    } else if (touches.length === 1 && touchState.gestureType === 'potential_longpress') {
        // If moved too much, cancel long press and potentially start pan
        const touch = touches[0];
        touchState.gestureType = 'canvas_drag';
    }
}

function handleCanvasTouchEnd(e) {
    // Clear long press timer
    if (touchState.longPressTimer) {
        clearTimeout(touchState.longPressTimer);
        touchState.longPressTimer = null;
    }

    // Reset touch state when all fingers lifted
    if (e.touches.length === 0) {
        touchState.gestureType = null;
        touchState.initialDistance = null;
        touchState.initialZoom = null;
        touchState.lastPanCenter = null;
        touchState.longPressTarget = null;
    } else if (e.touches.length === 1 && touchState.gestureType === 'pinch') {
        // Transitioned from 2 fingers to 1 - could start new gesture
        touchState.gestureType = null;
        touchState.initialDistance = null;
        touchState.lastPanCenter = null;
    }
}

// Initialize touch gestures when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initCanvasTouchGestures();
});

// ========== MOBILE CONTEXT SHEET (Wave 6) ==========
// Bottom action sheet for touch devices (replaces right-click menu on mobile)

function showMobileContextSheet(e, target) {
    // Close any existing mobile sheet
    closeMobileContextSheet();

    const nodeEl = target.closest('.workflow-node');
    const nodeId = nodeEl ? nodeEl.id : null;
    const node = nodeId ? nodes.find(n => n.id === nodeId) : null;

    // Store target for actions
    if (node) {
        contextMenuTargetNode = node;
    }

    // Build action list based on context
    let actions = [];

    if (node) {
        // Node context actions
        actions = [
            { icon: '⚙️', label: 'Edit Configuration', action: () => { closeMobileContextSheet(); editNodeConfig(nodeId); } },
            { icon: '📋', label: 'Duplicate', action: () => { closeMobileContextSheet(); duplicateNode(node); } },
            { icon: '📄', label: 'Copy', action: () => { closeMobileContextSheet(); contextMenuCopy(); } },
            { icon: '🔴', label: node.breakpoint ? 'Remove Breakpoint' : 'Set Breakpoint', action: () => { closeMobileContextSheet(); contextMenuToggleBreakpoint(); } },
            { icon: '✂️', label: 'Disconnect All', action: () => { closeMobileContextSheet(); contextMenuDisconnectAll(); } },
            { icon: '🗑️', label: 'Delete', action: () => { closeMobileContextSheet(); deleteNode(nodeId); }, danger: true }
        ];
    } else {
        // Canvas context actions
        const touch = e.touches ? e.touches[0] || e.changedTouches[0] : e;
        contextMenuPosition = { x: touch.clientX, y: touch.clientY };

        actions = [
            { icon: '➕', label: 'Add Node Here', action: () => { closeMobileContextSheet(); contextMenuAddNodeHere(); } },
            { icon: '📄', label: 'Paste', action: () => { closeMobileContextSheet(); contextMenuPaste(); }, disabled: copiedNodes.length === 0 },
            { icon: '✓', label: 'Select All', action: () => { closeMobileContextSheet(); selectAllNodes(); } },
            { icon: '📐', label: 'Fit to View', action: () => { closeMobileContextSheet(); fitToView(); } },
            { icon: '🔍', label: 'Reset Zoom', action: () => { closeMobileContextSheet(); resetZoom(); resetPan(); } }
        ];
    }

    // Create the mobile context sheet
    const sheet = document.createElement('div');
    sheet.id = 'mobileContextSheet';
    sheet.className = 'mobile-context-sheet';

    const backdrop = document.createElement('div');
    backdrop.className = 'mobile-sheet-backdrop';
    backdrop.onclick = closeMobileContextSheet;

    sheet.innerHTML = `
        <div class="mobile-sheet-handle" onclick="closeMobileContextSheet()"></div>
        <div class="mobile-sheet-header">
            <h3>${node ? (node.config?.label || node.definition?.name || 'Node') : 'Canvas'}</h3>
            <button class="mobile-sheet-close" onclick="closeMobileContextSheet()">✕</button>
        </div>
        <div class="mobile-sheet-actions">
            ${actions.map(a => `
                <button class="mobile-action-btn ${a.danger ? 'danger' : ''} ${a.disabled ? 'disabled' : ''}"
                        ${a.disabled ? 'disabled' : ''}>
                    <span class="action-icon">${a.icon}</span>
                    <span class="action-label">${a.label}</span>
                </button>
            `).join('')}
        </div>
    `;

    // Add click handlers to action buttons
    document.body.appendChild(backdrop);
    document.body.appendChild(sheet);

    const buttons = sheet.querySelectorAll('.mobile-action-btn:not(.disabled)');
    buttons.forEach((btn, index) => {
        if (actions[index] && !actions[index].disabled) {
            btn.onclick = actions[index].action;
        }
    });

    // Animate in
    requestAnimationFrame(() => {
        backdrop.classList.add('visible');
        sheet.classList.add('open');
    });
}

function closeMobileContextSheet() {
    const sheet = document.getElementById('mobileContextSheet');
    const backdrop = document.querySelector('.mobile-sheet-backdrop');

    if (sheet) {
        sheet.classList.remove('open');
        setTimeout(() => sheet.remove(), 300);
    }
    if (backdrop) {
        backdrop.classList.remove('visible');
        setTimeout(() => backdrop.remove(), 300);
    }
}

// ========== MOBILE PROPERTIES PANEL (Wave 6) ==========
// Slide-up properties panel for editing node config on mobile

function showMobileProperties(nodeId) {
    closeMobileContextSheet();

    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Close any existing mobile properties
    closeMobileProperties();

    const sheet = document.createElement('div');
    sheet.id = 'mobilePropertiesSheet';
    sheet.className = 'mobile-properties-sheet';

    const backdrop = document.createElement('div');
    backdrop.className = 'mobile-sheet-backdrop';
    backdrop.onclick = closeMobileProperties;

    // Build properties HTML (simplified version of properties panel)
    const definition = node.definition || {};
    const config = node.config || {};

    let fieldsHtml = '';
    if (definition.configSchema) {
        definition.configSchema.forEach(field => {
            const value = config[field.name] || field.default || '';
            fieldsHtml += `
                <div class="mobile-property-field">
                    <label>${field.label || field.name}</label>
                    ${field.type === 'select' ?
                        `<select data-field="${field.name}">
                            ${(field.options || []).map(opt =>
                                `<option value="${opt}" ${value === opt ? 'selected' : ''}>${opt}</option>`
                            ).join('')}
                        </select>` :
                    field.type === 'textarea' ?
                        `<textarea data-field="${field.name}" rows="3">${value}</textarea>` :
                        `<input type="${field.type || 'text'}" data-field="${field.name}" value="${value}" />`
                    }
                </div>
            `;
        });
    }

    // Always include label field
    fieldsHtml = `
        <div class="mobile-property-field">
            <label>Label</label>
            <input type="text" data-field="label" value="${config.label || ''}" />
        </div>
    ` + fieldsHtml;

    sheet.innerHTML = `
        <div class="mobile-sheet-handle"></div>
        <div class="mobile-sheet-header">
            <h3>${definition.name || 'Properties'}</h3>
            <button class="mobile-sheet-close" onclick="closeMobileProperties()">✕</button>
        </div>
        <div class="mobile-properties-content">
            <div class="mobile-property-node-type">
                <span class="node-icon">${definition.icon || '📦'}</span>
                <span class="node-type">${definition.category || 'Unknown'} / ${definition.name || 'Unknown'}</span>
            </div>
            ${fieldsHtml}
            <div class="mobile-properties-actions">
                <button class="mobile-btn secondary" onclick="closeMobileProperties()">Cancel</button>
                <button class="mobile-btn primary" onclick="saveMobileProperties('${nodeId}')">Save</button>
            </div>
        </div>
    `;

    document.body.appendChild(backdrop);
    document.body.appendChild(sheet);

    // Enable swipe to close
    let startY = 0;
    let currentY = 0;

    sheet.addEventListener('touchstart', (e) => {
        if (e.target.closest('.mobile-sheet-handle')) {
            startY = e.touches[0].clientY;
        }
    }, { passive: true });

    sheet.addEventListener('touchmove', (e) => {
        if (startY > 0) {
            currentY = e.touches[0].clientY;
            const diff = currentY - startY;
            if (diff > 0) {
                sheet.style.transform = `translateY(${diff}px)`;
            }
        }
    }, { passive: true });

    sheet.addEventListener('touchend', () => {
        const diff = currentY - startY;
        if (diff > 100) {
            closeMobileProperties();
        } else {
            sheet.style.transform = '';
        }
        startY = 0;
        currentY = 0;
    }, { passive: true });

    // Animate in
    requestAnimationFrame(() => {
        backdrop.classList.add('visible');
        sheet.classList.add('open');
    });
}

function closeMobileProperties() {
    const sheet = document.getElementById('mobilePropertiesSheet');
    const backdrop = document.querySelector('.mobile-sheet-backdrop');

    if (sheet) {
        sheet.classList.remove('open');
        setTimeout(() => sheet.remove(), 300);
    }
    if (backdrop) {
        backdrop.classList.remove('visible');
        setTimeout(() => backdrop.remove(), 300);
    }
}

function saveMobileProperties(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    const sheet = document.getElementById('mobilePropertiesSheet');
    if (!sheet) return;

    // Collect all field values
    const fields = sheet.querySelectorAll('[data-field]');
    fields.forEach(field => {
        const fieldName = field.dataset.field;
        const value = field.value;
        node.config[fieldName] = value;
    });

    // Update node display
    updateNodeDisplay(node);
    renderConnections();

    closeMobileProperties();
    showToast('Properties saved', 'success');
}

// Keyboard shortcuts for zoom and search
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        if (e.key === '+' || e.key === '=') {
            e.preventDefault();
            zoomIn();
        } else if (e.key === '-') {
            e.preventDefault();
            zoomOut();
        } else if (e.key === '0') {
            e.preventDefault();
            resetZoom();
        } else if (e.key === 'f' || e.key === 'F') {
            // Ctrl+F: Focus search input (Wave 5.1)
            e.preventDefault();
            focusNodeSearch();
        } else if (e.key === 'p' || e.key === 'P') {
            // Ctrl+P: Open command palette (Wave 5.1)
            e.preventDefault();
            openCommandPalette();
        }
    }
    // Escape closes command palette
    if (e.key === 'Escape') {
        closeCommandPalette();
    }
    // L: Toggle logs panel (Wave 5.6)
    if (e.key === 'l' || e.key === 'L') {
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
            const target = e.target;
            if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
                toggleLogsPanel();
            }
        }
    }
});

// ========== HOTKEY FUNCTIONS ==========
let clipboardNode = null;
let undoStack = [];
let redoStack = [];
const MAX_UNDO_STACK = 50;

// ========== UNDO/REDO HISTORY (Wave 4.1) ==========
let isTransactionActive = false;
let transactionActions = [];

// Action types for history display
const ACTION_ICONS = {
    'add_node': '➕',
    'delete_node': '🗑️',
    'connect': '🔗',
    'disconnect': '✂️',
    'update_config': '⚙️',
    'move_node': '↔️',
    'duplicate': '📋',
    'paste': '📄',
    'clear': '🧹',
    'import': '📥',
    'unknown': '•'
};

function duplicateNode(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    // Create duplicate with offset
    const duplicated = {
        id: `node-${nextNodeId++}`,
        agentType: node.agentType,
        x: node.x + 40,
        y: node.y + 40,
        config: JSON.parse(JSON.stringify(node.config)),
        definition: node.definition
    };

    nodes.push(duplicated);
    renderNode(duplicated);

    // Select the new node
    deselectAll();
    selectNode(duplicated.id);

    showToast(`Duplicated ${node.definition.name}`, 'success');
    saveUndoState('duplicate', `Duplicated ${node.definition.name}`);
}

function copyNode(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    clipboardNode = {
        agentType: node.agentType,
        config: JSON.parse(JSON.stringify(node.config)),
        definition: node.definition
    };

    showToast('Node copied', 'success');
}

function pasteNode() {
    if (!clipboardNode) {
        showToast('Nothing to paste', 'error');
        return;
    }

    // Paste at center of visible canvas
    const viewport = document.getElementById('canvasViewport');
    const x = viewport.scrollLeft + viewport.clientWidth / 2;
    const y = viewport.scrollTop + viewport.clientHeight / 2;

    const pasted = {
        id: `node-${nextNodeId++}`,
        agentType: clipboardNode.agentType,
        x: x,
        y: y,
        config: JSON.parse(JSON.stringify(clipboardNode.config)),
        definition: clipboardNode.definition
    };

    nodes.push(pasted);
    renderNode(pasted);

    // Select the new node
    deselectAll();
    selectNode(pasted.id);

    showToast(`Pasted ${pasted.definition.name}`, 'success');
    saveUndoState('paste', `Pasted ${pasted.definition.name}`);
}

function selectAllNodes() {
    // Currently single-select only, this could be enhanced for multi-select
    if (nodes.length > 0) {
        selectNode(nodes[nodes.length - 1].id);
        showToast(`${nodes.length} nodes on canvas`, 'info');
    }
}

function selectNode(nodeId) {
    deselectAll();
    selectedNode = nodeId;
    const nodeEl = document.getElementById(nodeId);
    if (nodeEl) {
        nodeEl.classList.add('selected');
    }
}

function saveUndoState(actionType = 'unknown', description = '') {
    // If in transaction mode, queue the action
    if (isTransactionActive) {
        transactionActions.push({ actionType, description });
        return;
    }

    const state = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections)),
        // Wave 4.1: Action metadata
        action: actionType,
        description: description || getDefaultDescription(actionType),
        timestamp: Date.now()
    };

    undoStack.push(state);
    if (undoStack.length > MAX_UNDO_STACK) {
        undoStack.shift();
    }

    // Clear redo stack when new action is performed
    redoStack = [];

    // Update history panel if visible
    updateHistoryPanel();
}

// Get default description for action type
function getDefaultDescription(actionType) {
    const descriptions = {
        'add_node': 'Add node',
        'delete_node': 'Delete node',
        'connect': 'Create connection',
        'disconnect': 'Delete connection',
        'update_config': 'Update config',
        'move_node': 'Move node',
        'duplicate': 'Duplicate node',
        'paste': 'Paste node',
        'clear': 'Clear canvas',
        'import': 'Import workflow',
        'unknown': 'Change'
    };
    return descriptions[actionType] || 'Change';
}

// Transaction grouping for multi-action operations
function beginTransaction() {
    isTransactionActive = true;
    transactionActions = [];
}

function commitTransaction(description = 'Multiple changes') {
    if (!isTransactionActive) return;

    isTransactionActive = false;

    // Only save if actions were performed
    if (transactionActions.length > 0) {
        const state = {
            nodes: JSON.parse(JSON.stringify(nodes)),
            connections: JSON.parse(JSON.stringify(connections)),
            action: 'transaction',
            description: description,
            timestamp: Date.now(),
            subActions: transactionActions
        };

        undoStack.push(state);
        if (undoStack.length > MAX_UNDO_STACK) {
            undoStack.shift();
        }
        redoStack = [];
        updateHistoryPanel();
    }

    transactionActions = [];
}

function cancelTransaction() {
    isTransactionActive = false;
    transactionActions = [];
}

function undo() {
    if (undoStack.length === 0) {
        showToast('Nothing to undo', 'error');
        return;
    }

    // Save current state to redo stack with metadata
    const currentState = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections)),
        action: 'undo_restore',
        description: 'Restored state',
        timestamp: Date.now()
    };
    redoStack.push(currentState);

    // Restore previous state
    const prevState = undoStack.pop();
    nodes.length = 0;
    connections.length = 0;
    nodes.push(...prevState.nodes);
    connections.push(...prevState.connections);

    // Re-render
    clearCanvasOnly();
    nodes.forEach(node => renderNode(node));
    renderConnections();
    updateHistoryPanel();

    showToast(`Undo: ${prevState.description || 'Change'}`, 'success');
}

function redo() {
    if (redoStack.length === 0) {
        showToast('Nothing to redo', 'error');
        return;
    }

    // Save current state to undo stack with metadata
    const currentState = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections)),
        action: 'redo_restore',
        description: 'Restored state',
        timestamp: Date.now()
    };
    undoStack.push(currentState);
    if (undoStack.length > MAX_UNDO_STACK) {
        undoStack.shift();
    }

    // Restore redo state
    const nextState = redoStack.pop();
    nodes.length = 0;
    connections.length = 0;
    nodes.push(...nextState.nodes);
    connections.push(...nextState.connections);

    // Re-render
    clearCanvasOnly();
    nodes.forEach(node => renderNode(node));
    renderConnections();
    updateHistoryPanel();

    showToast(`Redo: ${nextState.description || 'Change'}`, 'success');
}

// ========== HISTORY PANEL (Wave 4.1) ==========
let historyPanelVisible = false;

function toggleHistoryPanel() {
    const panel = document.getElementById('historyPanel');
    if (!panel) return;

    historyPanelVisible = !historyPanelVisible;
    panel.classList.toggle('visible', historyPanelVisible);

    if (historyPanelVisible) {
        updateHistoryPanel();
    }
}

function showHistoryPanel() {
    const panel = document.getElementById('historyPanel');
    if (!panel) return;

    historyPanelVisible = true;
    panel.classList.add('visible');
    updateHistoryPanel();
}

function hideHistoryPanel() {
    const panel = document.getElementById('historyPanel');
    if (!panel) return;

    historyPanelVisible = false;
    panel.classList.remove('visible');
}

function updateHistoryPanel() {
    const list = document.getElementById('historyList');
    if (!list) return;

    list.innerHTML = '';

    // Show undo history (newest first)
    const allStates = [...undoStack].reverse();

    if (allStates.length === 0) {
        list.innerHTML = '<div class="history-empty">No history yet</div>';
        return;
    }

    allStates.forEach((state, reverseIndex) => {
        const actualIndex = undoStack.length - 1 - reverseIndex;
        const item = document.createElement('div');
        item.className = 'history-item';
        if (reverseIndex === 0) item.classList.add('current');

        const icon = ACTION_ICONS[state.action] || ACTION_ICONS.unknown;
        const time = state.timestamp ? formatTimeAgo(state.timestamp) : '';

        item.innerHTML = `
            <span class="history-icon">${icon}</span>
            <span class="history-desc">${state.description || 'Unknown action'}</span>
            <span class="history-time">${time}</span>
        `;

        item.addEventListener('click', () => jumpToState(actualIndex));
        item.title = `Click to restore this state (${state.action})`;

        list.appendChild(item);
    });

    // Update history count badge
    const badge = document.getElementById('historyCount');
    if (badge) {
        badge.textContent = undoStack.length;
        badge.style.display = undoStack.length > 0 ? 'inline' : 'none';
    }
}

function jumpToState(targetIndex) {
    if (targetIndex < 0 || targetIndex >= undoStack.length) {
        showToast('Invalid state index', 'error');
        return;
    }

    // How many undos do we need?
    const undosNeeded = undoStack.length - 1 - targetIndex;

    if (undosNeeded === 0) {
        showToast('Already at this state', 'info');
        return;
    }

    // Confirm if jumping more than 5 states
    if (undosNeeded > 5) {
        if (!confirm(`Jump back ${undosNeeded} steps to this state?`)) {
            return;
        }
    }

    // Perform multiple undos
    for (let i = 0; i < undosNeeded; i++) {
        if (undoStack.length === 0) break;

        const currentState = {
            nodes: JSON.parse(JSON.stringify(nodes)),
            connections: JSON.parse(JSON.stringify(connections)),
            action: 'jump_restore',
            description: 'Jumped to state',
            timestamp: Date.now()
        };
        redoStack.push(currentState);

        const prevState = undoStack.pop();
        nodes.length = 0;
        connections.length = 0;
        nodes.push(...prevState.nodes);
        connections.push(...prevState.connections);
    }

    // Re-render
    clearCanvasOnly();
    nodes.forEach(node => renderNode(node));
    renderConnections();
    updateHistoryPanel();

    showToast(`Jumped back ${undosNeeded} step${undosNeeded > 1 ? 's' : ''}`, 'success');
}

function formatTimeAgo(timestamp) {
    const now = Date.now();
    const diff = now - timestamp;

    if (diff < 1000) return 'just now';
    if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
}

function clearHistory() {
    if (!confirm('Clear all undo/redo history?')) return;

    undoStack = [];
    redoStack = [];
    updateHistoryPanel();
    showToast('History cleared', 'success');
}

// ========== PERFORMANCE DASHBOARD (Wave 4.2) ==========
let perfDashboardVisible = false;
let perfRefreshTimer = null;
let perfCharts = {};

// Execution metrics storage
const executionMetrics = {
    totalExecutions: 0,
    successCount: 0,
    failureCount: 0,
    latencyHistory: [],       // [{timestamp, latency_ms, node_count}]
    nodeTypeStats: {},        // {nodeType: {count, totalLatency, successes, failures}}
    executionTimeline: [],    // [{execution_id, nodes: [{name, start, duration}]}]
    lastUpdate: null
};

// Thompson Sampling priors for AI suggestions
const thompsonPriors = {
    // nodeType → {alpha, beta} for success rate estimation
    hololoom: { alpha: 1, beta: 1 },
    search: { alpha: 1, beta: 1 },
    multiquery: { alpha: 1, beta: 1 },
    embedder: { alpha: 1, beta: 1 },
    synthesizer: { alpha: 1, beta: 1 },
    refiner: { alpha: 1, beta: 1 },
    memory_store: { alpha: 1, beta: 1 },
    context_retriever: { alpha: 1, beta: 1 },
    knowledge_fusion: { alpha: 1, beta: 1 },
    thompson: { alpha: 1, beta: 1 },
    convergence: { alpha: 1, beta: 1 },
    safety: { alpha: 1, beta: 1 },
    response: { alpha: 1, beta: 1 },
    format: { alpha: 1, beta: 1 },
    conditional: { alpha: 1, beta: 1 },
    loop: { alpha: 1, beta: 1 },
    parallel: { alpha: 1, beta: 1 }
};

function showPerfDashboard() {
    const dashboard = document.getElementById('perfDashboard');
    const overlay = document.getElementById('perfDashboardOverlay');
    if (!dashboard || !overlay) return;

    perfDashboardVisible = true;
    dashboard.style.display = 'flex';
    overlay.style.display = 'block';

    // Initialize charts if not already done
    initializePerfCharts();

    // Update with current data
    updatePerfDashboard();

    // Start auto-refresh based on current setting
    const interval = parseInt(document.getElementById('perfRefreshInterval')?.value || '10000');
    setPerfRefreshInterval(interval);
}

function hidePerfDashboard() {
    const dashboard = document.getElementById('perfDashboard');
    const overlay = document.getElementById('perfDashboardOverlay');
    if (!dashboard || !overlay) return;

    perfDashboardVisible = false;
    dashboard.style.display = 'none';
    overlay.style.display = 'none';

    // Stop auto-refresh
    if (perfRefreshTimer) {
        clearInterval(perfRefreshTimer);
        perfRefreshTimer = null;
    }
}

function setPerfRefreshInterval(interval) {
    const ms = parseInt(interval);

    // Clear existing timer
    if (perfRefreshTimer) {
        clearInterval(perfRefreshTimer);
        perfRefreshTimer = null;
    }

    // Update live indicator
    const liveDot = document.getElementById('perfLiveDot');
    const liveStatus = document.getElementById('perfLiveStatus');

    if (ms === 0) {
        // Manual mode
        if (liveDot) liveDot.style.background = '#888';
        if (liveStatus) liveStatus.textContent = 'Manual';
    } else {
        // Auto-refresh mode
        if (liveDot) liveDot.style.background = '#4CAF50';
        if (liveStatus) liveStatus.textContent = 'Live';

        perfRefreshTimer = setInterval(() => {
            if (perfDashboardVisible) {
                updatePerfDashboard();
            }
        }, ms);
    }
}

function initializePerfCharts() {
    // Only initialize once
    if (Object.keys(perfCharts).length > 0) return;

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: '#e0e0e0', font: { size: 11 } }
            }
        },
        scales: {
            x: {
                ticks: { color: '#888' },
                grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
                ticks: { color: '#888' },
                grid: { color: 'rgba(255,255,255,0.05)' }
            }
        }
    };

    // 1. Latency Trends (Line Chart)
    const latencyCtx = document.getElementById('latencyChart')?.getContext('2d');
    if (latencyCtx) {
        perfCharts.latency = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency (ms)',
                    data: [],
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.3
                }, {
                    label: 'P95',
                    data: [],
                    borderColor: '#f093fb',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                ...chartOptions,
                plugins: {
                    ...chartOptions.plugins,
                    title: { display: false }
                }
            }
        });
    }

    // 2. Success Rate (Donut Chart)
    const successCtx = document.getElementById('successChart')?.getContext('2d');
    if (successCtx) {
        perfCharts.success = new Chart(successCtx, {
            type: 'doughnut',
            data: {
                labels: ['Success', 'Failure'],
                datasets: [{
                    data: [1, 0],
                    backgroundColor: ['#4CAF50', '#f44336'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#e0e0e0', font: { size: 11 } }
                    }
                },
                cutout: '70%'
            }
        });
    }

    // 3. Node Comparison (Bar Chart)
    const nodeCtx = document.getElementById('nodeComparisonChart')?.getContext('2d');
    if (nodeCtx) {
        perfCharts.nodes = new Chart(nodeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Avg Latency (ms)',
                    data: [],
                    backgroundColor: '#667eea'
                }, {
                    label: 'Success Rate (%)',
                    data: [],
                    backgroundColor: '#4CAF50'
                }]
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // 4. Thompson Priors (Beta Distribution Visualization)
    const thompsonCtx = document.getElementById('thompsonChart')?.getContext('2d');
    if (thompsonCtx) {
        perfCharts.thompson = new Chart(thompsonCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Expected Success Rate',
                    data: [],
                    backgroundColor: [],
                    borderWidth: 1,
                    borderColor: 'rgba(255,255,255,0.2)'
                }]
            },
            options: {
                ...chartOptions,
                indexAxis: 'y',
                scales: {
                    x: {
                        ...chartOptions.scales.x,
                        min: 0,
                        max: 1,
                        ticks: {
                            color: '#888',
                            callback: (v) => `${(v * 100).toFixed(0)}%`
                        }
                    },
                    y: {
                        ...chartOptions.scales.y,
                        ticks: { color: '#e0e0e0', font: { size: 10 } }
                    }
                }
            }
        });
    }

    // 5. Execution Timeline (Horizontal Bar - Gantt style)
    const timelineCtx = document.getElementById('timelineChart')?.getContext('2d');
    if (timelineCtx) {
        perfCharts.timeline = new Chart(timelineCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Duration (ms)',
                    data: [],
                    backgroundColor: '#667eea',
                    barThickness: 16
                }]
            },
            options: {
                ...chartOptions,
                indexAxis: 'y',
                scales: {
                    x: {
                        ...chartOptions.scales.x,
                        stacked: true,
                        title: { display: true, text: 'Time (ms)', color: '#888' }
                    },
                    y: {
                        ...chartOptions.scales.y,
                        stacked: true
                    }
                }
            }
        });
    }

    // 6. Bottleneck Analysis (Pie Chart)
    const bottleneckCtx = document.getElementById('bottleneckChart')?.getContext('2d');
    if (bottleneckCtx) {
        perfCharts.bottleneck = new Chart(bottleneckCtx, {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#667eea', '#f093fb', '#4CAF50', '#FFC107',
                        '#2196F3', '#9C27B0', '#FF5722', '#00BCD4'
                    ],
                    borderWidth: 1,
                    borderColor: '#1a1a2e'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: { color: '#e0e0e0', font: { size: 10 } }
                    }
                }
            }
        });
    }
}

function updatePerfDashboard() {
    executionMetrics.lastUpdate = Date.now();

    // Generate sample data if no real data exists
    if (executionMetrics.latencyHistory.length === 0) {
        generateSampleMetrics();
    }

    // Update charts
    updateLatencyChart();
    updateSuccessChart();
    updateNodeComparisonChart();
    updateThompsonChart();
    updateTimelineChart();
    updateBottleneckChart();

    // Update stats bar
    updateStatsBar();

    // Update badges
    updatePerfBadges();
}

function generateSampleMetrics() {
    // Generate realistic sample data for demonstration
    const now = Date.now();
    const nodeTypes = ['hololoom', 'search', 'synthesizer', 'response', 'safety'];

    // Generate 20 sample executions
    for (let i = 0; i < 20; i++) {
        const latency = 100 + Math.random() * 200 + (i % 5 === 0 ? Math.random() * 100 : 0);
        const success = Math.random() > 0.15;

        executionMetrics.latencyHistory.push({
            timestamp: now - (20 - i) * 60000,
            latency_ms: latency,
            node_count: 3 + Math.floor(Math.random() * 5),
            success: success
        });

        executionMetrics.totalExecutions++;
        if (success) {
            executionMetrics.successCount++;
        } else {
            executionMetrics.failureCount++;
        }

        // Update node stats
        const nodeType = nodeTypes[Math.floor(Math.random() * nodeTypes.length)];
        if (!executionMetrics.nodeTypeStats[nodeType]) {
            executionMetrics.nodeTypeStats[nodeType] = {
                count: 0, totalLatency: 0, successes: 0, failures: 0
            };
        }
        const stats = executionMetrics.nodeTypeStats[nodeType];
        stats.count++;
        stats.totalLatency += latency / 3;
        if (success) stats.successes++;
        else stats.failures++;

        // Update Thompson priors
        if (thompsonPriors[nodeType]) {
            if (success) {
                thompsonPriors[nodeType].alpha += 1;
            } else {
                thompsonPriors[nodeType].beta += 1;
            }
        }
    }

    // Generate sample timeline for last execution
    executionMetrics.executionTimeline = [{
        execution_id: 'exec_001',
        total_ms: 350,
        nodes: [
            { name: 'HoloLoom Query', start: 0, duration: 120 },
            { name: 'Memory Search', start: 120, duration: 80 },
            { name: 'Synthesizer', start: 200, duration: 100 },
            { name: 'Response', start: 300, duration: 50 }
        ]
    }];
}

function updateLatencyChart() {
    if (!perfCharts.latency) return;

    const history = executionMetrics.latencyHistory.slice(-20);
    const labels = history.map((h, i) => `${i + 1}`);
    const data = history.map(h => h.latency_ms);

    // Calculate P95
    const sorted = [...data].sort((a, b) => a - b);
    const p95Index = Math.floor(sorted.length * 0.95);
    const p95 = sorted[p95Index] || sorted[sorted.length - 1] || 0;
    const p95Data = new Array(data.length).fill(p95);

    perfCharts.latency.data.labels = labels;
    perfCharts.latency.data.datasets[0].data = data;
    perfCharts.latency.data.datasets[1].data = p95Data;
    perfCharts.latency.update('none');

    // Update P95 badge
    const badge = document.getElementById('latencyP95Badge');
    if (badge) badge.textContent = `P95: ${p95.toFixed(0)}ms`;
}

function updateSuccessChart() {
    if (!perfCharts.success) return;

    const success = executionMetrics.successCount || 1;
    const failure = executionMetrics.failureCount || 0;

    perfCharts.success.data.datasets[0].data = [success, failure];
    perfCharts.success.update('none');

    // Update success rate badge
    const total = success + failure;
    const rate = total > 0 ? (success / total * 100).toFixed(1) : '100';
    const badge = document.getElementById('successRateBadge');
    if (badge) badge.textContent = `${rate}%`;
}

function updateNodeComparisonChart() {
    if (!perfCharts.nodes) return;

    const stats = executionMetrics.nodeTypeStats;
    const labels = Object.keys(stats).slice(0, 8);
    const avgLatencies = labels.map(k => stats[k].totalLatency / stats[k].count);
    const successRates = labels.map(k => {
        const s = stats[k];
        return s.count > 0 ? (s.successes / s.count * 100) : 0;
    });

    perfCharts.nodes.data.labels = labels.map(l => l.substring(0, 10));
    perfCharts.nodes.data.datasets[0].data = avgLatencies;
    perfCharts.nodes.data.datasets[1].data = successRates;
    perfCharts.nodes.update('none');
}

function updateThompsonChart() {
    if (!perfCharts.thompson) return;

    // Get nodes with data (alpha + beta > 2 means at least one observation)
    const entries = Object.entries(thompsonPriors)
        .filter(([_, v]) => v.alpha + v.beta > 2)
        .sort((a, b) => {
            const expA = a[1].alpha / (a[1].alpha + a[1].beta);
            const expB = b[1].alpha / (b[1].alpha + b[1].beta);
            return expB - expA;
        })
        .slice(0, 10);

    const labels = entries.map(([k]) => k);
    const expectedValues = entries.map(([_, v]) => v.alpha / (v.alpha + v.beta));

    // Color gradient from red (low) to green (high)
    const colors = expectedValues.map(v => {
        const hue = v * 120; // 0 = red, 120 = green
        return `hsl(${hue}, 70%, 50%)`;
    });

    perfCharts.thompson.data.labels = labels;
    perfCharts.thompson.data.datasets[0].data = expectedValues;
    perfCharts.thompson.data.datasets[0].backgroundColor = colors;
    perfCharts.thompson.update('none');
}

function updateTimelineChart() {
    if (!perfCharts.timeline || executionMetrics.executionTimeline.length === 0) return;

    const lastExec = executionMetrics.executionTimeline[executionMetrics.executionTimeline.length - 1];
    const nodes = lastExec.nodes || [];

    const labels = nodes.map(n => n.name.substring(0, 15));
    const durations = nodes.map(n => n.duration);

    perfCharts.timeline.data.labels = labels;
    perfCharts.timeline.data.datasets[0].data = durations;
    perfCharts.timeline.update('none');
}

function updateBottleneckChart() {
    if (!perfCharts.bottleneck) return;

    const stats = executionMetrics.nodeTypeStats;
    const entries = Object.entries(stats)
        .map(([name, data]) => ({ name, totalTime: data.totalLatency }))
        .sort((a, b) => b.totalTime - a.totalTime)
        .slice(0, 8);

    const labels = entries.map(e => e.name);
    const data = entries.map(e => e.totalTime);

    perfCharts.bottleneck.data.labels = labels;
    perfCharts.bottleneck.data.datasets[0].data = data;
    perfCharts.bottleneck.update('none');
}

function updateStatsBar() {
    // Total Executions
    const totalEl = document.getElementById('perfStatTotal');
    if (totalEl) totalEl.textContent = executionMetrics.totalExecutions.toLocaleString();

    // Average Latency
    const avgLatency = executionMetrics.latencyHistory.length > 0
        ? executionMetrics.latencyHistory.reduce((sum, h) => sum + h.latency_ms, 0) / executionMetrics.latencyHistory.length
        : 0;
    const avgEl = document.getElementById('perfStatAvgLatency');
    if (avgEl) avgEl.textContent = `${avgLatency.toFixed(0)}ms`;

    // Success Rate
    const total = executionMetrics.successCount + executionMetrics.failureCount;
    const successRate = total > 0 ? (executionMetrics.successCount / total * 100) : 100;
    const rateEl = document.getElementById('perfStatSuccessRate');
    if (rateEl) rateEl.textContent = `${successRate.toFixed(1)}%`;

    // Top Performer (highest Thompson expected value)
    let topNode = 'N/A';
    let topValue = 0;
    for (const [name, prior] of Object.entries(thompsonPriors)) {
        if (prior.alpha + prior.beta > 2) {
            const expected = prior.alpha / (prior.alpha + prior.beta);
            if (expected > topValue) {
                topValue = expected;
                topNode = name;
            }
        }
    }
    const topEl = document.getElementById('perfStatTopNode');
    if (topEl) topEl.textContent = topNode;
}

function updatePerfBadges() {
    // Individual panel badges are updated in their respective update functions
}

function exportPerfData(format) {
    const data = {
        exportedAt: new Date().toISOString(),
        metrics: executionMetrics,
        thompsonPriors: thompsonPriors
    };

    if (format === 'json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `perf_metrics_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Exported as JSON', 'success');
    } else if (format === 'csv') {
        // Convert latency history to CSV
        const headers = ['timestamp', 'latency_ms', 'node_count', 'success'];
        const rows = executionMetrics.latencyHistory.map(h => [
            new Date(h.timestamp).toISOString(),
            h.latency_ms.toFixed(2),
            h.node_count,
            h.success ? '1' : '0'
        ]);

        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `perf_metrics_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Exported as CSV', 'success');
    }
}

// Record metrics from actual workflow execution
function recordExecutionMetrics(executionResult) {
    const { nodes: executedNodes, duration_ms, success } = executionResult;

    executionMetrics.totalExecutions++;
    if (success) {
        executionMetrics.successCount++;
    } else {
        executionMetrics.failureCount++;
    }

    executionMetrics.latencyHistory.push({
        timestamp: Date.now(),
        latency_ms: duration_ms,
        node_count: executedNodes?.length || 0,
        success: success
    });

    // Keep only last 100 entries
    if (executionMetrics.latencyHistory.length > 100) {
        executionMetrics.latencyHistory.shift();
    }

    // Update node-specific stats
    if (executedNodes) {
        for (const node of executedNodes) {
            const nodeType = node.agentType || 'unknown';
            if (!executionMetrics.nodeTypeStats[nodeType]) {
                executionMetrics.nodeTypeStats[nodeType] = {
                    count: 0, totalLatency: 0, successes: 0, failures: 0
                };
            }
            const stats = executionMetrics.nodeTypeStats[nodeType];
            stats.count++;
            stats.totalLatency += node.duration_ms || (duration_ms / executedNodes.length);
            if (success) stats.successes++;
            else stats.failures++;

            // Update Thompson priors
            if (thompsonPriors[nodeType]) {
                if (success) {
                    thompsonPriors[nodeType].alpha += 0.5;
                } else {
                    thompsonPriors[nodeType].beta += 0.5;
                }
            }
        }
    }

    // If dashboard is open, update it
    if (perfDashboardVisible) {
        updatePerfDashboard();
    }
}

// Keyboard shortcut for Performance Dashboard
function handlePerfDashboardShortcut(e) {
    if ((e.key === 'p' || e.key === 'P') && !e.ctrlKey && !e.metaKey) {
        const activeEl = document.activeElement;
        if (activeEl.tagName !== 'INPUT' && activeEl.tagName !== 'TEXTAREA') {
            e.preventDefault();
            if (perfDashboardVisible) {
                hidePerfDashboard();
            } else {
                showPerfDashboard();
            }
        }
    }
}

// Add keyboard listener for 'P'
document.addEventListener('keydown', handlePerfDashboardShortcut);

// ========== AUTO-LAYOUT SYSTEM (Wave 4.3) ==========

// Layout configuration
const layoutConfig = {
    direction: 'top-down',      // 'top-down', 'left-right', 'bottom-up', 'right-left', 'radial'
    algorithm: 'hierarchical',  // 'hierarchical', 'force-directed'
    nodeSpacingX: 250,          // Horizontal spacing between nodes
    nodeSpacingY: 150,          // Vertical spacing between levels
    animationDuration: 300,     // Animation duration in ms
    centerOnCanvas: true,       // Center result on canvas
    radialRadius: 200,          // Base radius for radial layout
    forceIterations: 100,       // Iterations for force-directed
    forceStrength: 0.1,         // Attraction/repulsion strength
    forceDamping: 0.9           // Velocity damping
};

/**
 * Sugiyama-style hierarchical layout algorithm for DAGs.
 * Steps:
 * 1. Find root nodes (no incoming edges)
 * 2. Assign layers via topological sort
 * 3. Order nodes within layers to minimize edge crossings
 * 4. Assign final positions based on layer and order
 */
function calculateHierarchicalLayout() {
    if (nodes.length === 0) return [];

    // Build adjacency lists
    const outEdges = {};  // node -> [successor nodes]
    const inEdges = {};   // node -> [predecessor nodes]

    nodes.forEach(n => {
        outEdges[n.id] = [];
        inEdges[n.id] = [];
    });

    connections.forEach(c => {
        if (outEdges[c.from]) outEdges[c.from].push(c.to);
        if (inEdges[c.to]) inEdges[c.to].push(c.from);
    });

    // Step 1: Assign layers using longest path from roots
    const layers = {};
    const visited = new Set();

    // Find root nodes (no incoming edges)
    const roots = nodes.filter(n => inEdges[n.id].length === 0);

    // If no roots (cycle), start from first node
    const startNodes = roots.length > 0 ? roots : [nodes[0]];

    // BFS to assign layers
    function assignLayers(startNode, startLayer) {
        const queue = [[startNode.id, startLayer]];

        while (queue.length > 0) {
            const [nodeId, layer] = queue.shift();

            // Take maximum layer if already assigned
            if (layers[nodeId] !== undefined) {
                layers[nodeId] = Math.max(layers[nodeId], layer);
            } else {
                layers[nodeId] = layer;
            }

            // Process successors
            outEdges[nodeId].forEach(successorId => {
                queue.push([successorId, layer + 1]);
            });
        }
    }

    startNodes.forEach(n => assignLayers(n, 0));

    // Assign unvisited nodes to layer 0
    nodes.forEach(n => {
        if (layers[n.id] === undefined) {
            layers[n.id] = 0;
        }
    });

    // Step 2: Group nodes by layer
    const layerGroups = {};
    nodes.forEach(n => {
        const layer = layers[n.id];
        if (!layerGroups[layer]) layerGroups[layer] = [];
        layerGroups[layer].push(n);
    });

    // Step 3: Order nodes within layers (barycenter method for crossing reduction)
    const sortedLayers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);

    sortedLayers.forEach((layer, layerIndex) => {
        if (layerIndex === 0) return; // First layer stays as is

        const currentLayer = layerGroups[layer];
        const prevLayer = layerGroups[sortedLayers[layerIndex - 1]];

        // Calculate barycenter for each node in current layer
        currentLayer.forEach(node => {
            const predecessors = inEdges[node.id];
            if (predecessors.length === 0) {
                node._barycenter = Infinity;
                return;
            }

            let sum = 0;
            predecessors.forEach(predId => {
                const predIndex = prevLayer.findIndex(n => n.id === predId);
                if (predIndex !== -1) sum += predIndex;
            });
            node._barycenter = sum / predecessors.length;
        });

        // Sort by barycenter
        currentLayer.sort((a, b) => a._barycenter - b._barycenter);
    });

    // Step 4: Calculate positions
    const positions = [];
    const maxNodesInLayer = Math.max(...Object.values(layerGroups).map(g => g.length));

    sortedLayers.forEach(layer => {
        const nodesInLayer = layerGroups[layer];
        const layerWidth = nodesInLayer.length * layoutConfig.nodeSpacingX;
        const startX = -layerWidth / 2 + layoutConfig.nodeSpacingX / 2;

        nodesInLayer.forEach((node, index) => {
            let x, y;

            switch (layoutConfig.direction) {
                case 'top-down':
                    x = startX + index * layoutConfig.nodeSpacingX;
                    y = layer * layoutConfig.nodeSpacingY;
                    break;
                case 'bottom-up':
                    x = startX + index * layoutConfig.nodeSpacingX;
                    y = -layer * layoutConfig.nodeSpacingY;
                    break;
                case 'left-right':
                    x = layer * layoutConfig.nodeSpacingX;
                    y = startX + index * layoutConfig.nodeSpacingY;
                    break;
                case 'right-left':
                    x = -layer * layoutConfig.nodeSpacingX;
                    y = startX + index * layoutConfig.nodeSpacingY;
                    break;
                default:
                    x = startX + index * layoutConfig.nodeSpacingX;
                    y = layer * layoutConfig.nodeSpacingY;
            }

            positions.push({ nodeId: node.id, x, y });
        });
    });

    return positions;
}

/**
 * Force-directed layout using simple spring-embedder algorithm.
 * Nodes repel each other, edges act as springs.
 */
function calculateForceDirectedLayout() {
    if (nodes.length === 0) return [];
    if (nodes.length === 1) {
        return [{ nodeId: nodes[0].id, x: 0, y: 0 }];
    }

    // Initialize positions randomly if not already positioned
    const positions = {};
    nodes.forEach(n => {
        positions[n.id] = {
            x: n.x || Math.random() * 500 - 250,
            y: n.y || Math.random() * 500 - 250,
            vx: 0,
            vy: 0
        };
    });

    const repulsionStrength = 5000;
    const attractionStrength = layoutConfig.forceStrength;
    const damping = layoutConfig.forceDamping;
    const iterations = layoutConfig.forceIterations;

    for (let iter = 0; iter < iterations; iter++) {
        // Calculate repulsion forces (all pairs)
        nodes.forEach(n1 => {
            nodes.forEach(n2 => {
                if (n1.id === n2.id) return;

                const dx = positions[n1.id].x - positions[n2.id].x;
                const dy = positions[n1.id].y - positions[n2.id].y;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;

                const force = repulsionStrength / (dist * dist);
                positions[n1.id].vx += (dx / dist) * force;
                positions[n1.id].vy += (dy / dist) * force;
            });
        });

        // Calculate attraction forces (connected nodes)
        connections.forEach(c => {
            const p1 = positions[c.from];
            const p2 = positions[c.to];
            if (!p1 || !p2) return;

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;

            const force = (dist - layoutConfig.nodeSpacingX) * attractionStrength;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            p1.vx += fx;
            p1.vy += fy;
            p2.vx -= fx;
            p2.vy -= fy;
        });

        // Apply velocities and damping
        nodes.forEach(n => {
            const p = positions[n.id];
            p.x += p.vx;
            p.y += p.vy;
            p.vx *= damping;
            p.vy *= damping;
        });
    }

    return nodes.map(n => ({
        nodeId: n.id,
        x: positions[n.id].x,
        y: positions[n.id].y
    }));
}

/**
 * Radial layout - nodes arranged in concentric circles by layer.
 */
function calculateRadialLayout() {
    if (nodes.length === 0) return [];

    // Use hierarchical layers
    const hierarchicalPositions = calculateHierarchicalLayout();

    // Convert to radial
    const layers = {};
    hierarchicalPositions.forEach(p => {
        const node = nodes.find(n => n.id === p.nodeId);
        if (!node) return;

        // Determine layer from Y position
        const layer = Math.round(p.y / layoutConfig.nodeSpacingY);
        if (!layers[layer]) layers[layer] = [];
        layers[layer].push(p);
    });

    const sortedLayers = Object.keys(layers).map(Number).sort((a, b) => a - b);
    const positions = [];

    sortedLayers.forEach((layer, layerIndex) => {
        const nodesInLayer = layers[layer];
        const radius = (layerIndex + 1) * layoutConfig.radialRadius;
        const angleStep = (2 * Math.PI) / nodesInLayer.length;
        const startAngle = -Math.PI / 2; // Start from top

        nodesInLayer.forEach((p, index) => {
            const angle = startAngle + index * angleStep;
            positions.push({
                nodeId: p.nodeId,
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius
            });
        });
    });

    return positions;
}

/**
 * Apply calculated layout positions to nodes with animation.
 */
function applyLayout(positions, animate = true) {
    if (positions.length === 0) return;

    // Calculate center offset if needed
    let offsetX = 0, offsetY = 0;

    if (layoutConfig.centerOnCanvas) {
        const canvas = document.getElementById('canvasViewport');
        const canvasRect = canvas.getBoundingClientRect();

        // Find bounding box of positions
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        positions.forEach(p => {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x + 200);
            maxY = Math.max(maxY, p.y + 100);
        });

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        offsetX = canvasRect.width / 2 - centerX;
        offsetY = canvasRect.height / 2 - centerY + 100; // Add padding from top
    }

    // Apply positions
    positions.forEach(p => {
        const node = nodes.find(n => n.id === p.nodeId);
        if (!node) return;

        const targetX = p.x + offsetX;
        const targetY = p.y + offsetY;

        const element = document.getElementById(p.nodeId);
        if (!element) return;

        if (animate && layoutConfig.animationDuration > 0) {
            // Animate transition
            element.style.transition = `left ${layoutConfig.animationDuration}ms ease-out, top ${layoutConfig.animationDuration}ms ease-out`;
            element.style.left = targetX + 'px';
            element.style.top = targetY + 'px';

            // Update node data after animation
            setTimeout(() => {
                node.x = targetX;
                node.y = targetY;
                element.style.transition = '';
                renderConnections();
                updateMinimap();
            }, layoutConfig.animationDuration);
        } else {
            node.x = targetX;
            node.y = targetY;
            element.style.left = targetX + 'px';
            element.style.top = targetY + 'px';
        }
    });

    // Render connections during animation
    if (animate) {
        const animationSteps = 20;
        const stepDuration = layoutConfig.animationDuration / animationSteps;
        for (let i = 0; i < animationSteps; i++) {
            setTimeout(() => renderConnections(), i * stepDuration);
        }
    } else {
        renderConnections();
        updateMinimap();
    }
}

/**
 * Run auto-layout with current settings.
 */
function runAutoLayout(algorithm = null, direction = null, animate = true) {
    if (nodes.length === 0) {
        showToast('No nodes to layout', 'warning');
        return;
    }

    // Save undo state
    saveUndoState('auto_layout', `Auto-layout (${algorithm || layoutConfig.algorithm})`);

    // Override settings if provided
    if (algorithm) layoutConfig.algorithm = algorithm;
    if (direction) layoutConfig.direction = direction;

    let positions;

    switch (layoutConfig.algorithm) {
        case 'hierarchical':
            positions = calculateHierarchicalLayout();
            break;
        case 'force-directed':
            positions = calculateForceDirectedLayout();
            break;
        case 'radial':
            positions = calculateRadialLayout();
            break;
        default:
            positions = calculateHierarchicalLayout();
    }

    applyLayout(positions, animate);
    showToast(`Applied ${layoutConfig.algorithm} layout (${layoutConfig.direction})`, 'success');
}

/**
 * Show layout options panel.
 */
function showLayoutPanel() {
    const panel = document.getElementById('layoutPanel');
    const overlay = document.getElementById('layoutPanelOverlay');
    if (panel && overlay) {
        panel.style.display = 'block';
        overlay.style.display = 'block';
        updateLayoutPreview();
    }
}

/**
 * Hide layout options panel.
 */
function hideLayoutPanel() {
    const panel = document.getElementById('layoutPanel');
    const overlay = document.getElementById('layoutPanelOverlay');
    if (panel) panel.style.display = 'none';
    if (overlay) overlay.style.display = 'none';
}

/**
 * Update layout preview in the panel.
 */
function updateLayoutPreview() {
    const preview = document.getElementById('layoutPreview');
    if (!preview) return;

    preview.innerHTML = '';

    // Create mini preview of layout
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('viewBox', '-100 -100 200 200');

    // Generate preview positions
    const previewNodes = nodes.length > 0 ? nodes.slice(0, 6) : [
        { id: 'p1' }, { id: 'p2' }, { id: 'p3' }, { id: 'p4' }, { id: 'p5' }
    ];

    let previewPositions = [];
    const direction = layoutConfig.direction;

    if (layoutConfig.algorithm === 'radial' || direction === 'radial') {
        // Radial preview
        const angleStep = (2 * Math.PI) / previewNodes.length;
        previewNodes.forEach((n, i) => {
            const angle = -Math.PI / 2 + i * angleStep;
            const radius = 60;
            previewPositions.push({
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius
            });
        });
    } else if (layoutConfig.algorithm === 'force-directed') {
        // Force-directed preview (rough approximation)
        const angleStep = (2 * Math.PI) / previewNodes.length;
        previewNodes.forEach((n, i) => {
            const angle = i * angleStep;
            const radius = 50 + Math.random() * 20;
            previewPositions.push({
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius
            });
        });
    } else {
        // Hierarchical preview
        const layers = [[0], [1, 2], [3, 4]];
        layers.forEach((layer, layerIdx) => {
            const y = (layerIdx - 1) * 50;
            const xOffset = -((layer.length - 1) * 40) / 2;
            layer.forEach((nodeIdx, i) => {
                if (nodeIdx < previewNodes.length) {
                    let x = xOffset + i * 40;
                    let py = y;

                    if (direction === 'left-right') {
                        [x, py] = [py, x];
                    } else if (direction === 'bottom-up') {
                        py = -y;
                    } else if (direction === 'right-left') {
                        [x, py] = [-py, x];
                    }

                    previewPositions.push({ x, y: py });
                }
            });
        });
    }

    // Draw connections (lines)
    if (previewPositions.length > 1) {
        for (let i = 0; i < previewPositions.length - 1; i++) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', previewPositions[i].x);
            line.setAttribute('y1', previewPositions[i].y);
            line.setAttribute('x2', previewPositions[i + 1].x);
            line.setAttribute('y2', previewPositions[i + 1].y);
            line.setAttribute('stroke', '#4a5568');
            line.setAttribute('stroke-width', '2');
            svg.appendChild(line);
        }
    }

    // Draw nodes
    previewPositions.forEach((pos, i) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pos.x);
        circle.setAttribute('cy', pos.y);
        circle.setAttribute('r', '12');
        circle.setAttribute('fill', `hsl(${i * 50 + 200}, 70%, 60%)`);
        circle.setAttribute('stroke', '#fff');
        circle.setAttribute('stroke-width', '2');
        svg.appendChild(circle);
    });

    preview.appendChild(svg);
}

/**
 * Handle layout option change.
 */
function onLayoutOptionChange() {
    const algorithmSelect = document.getElementById('layoutAlgorithm');
    const directionSelect = document.getElementById('layoutDirection');
    const spacingXInput = document.getElementById('layoutSpacingX');
    const spacingYInput = document.getElementById('layoutSpacingY');
    const animateCheckbox = document.getElementById('layoutAnimate');

    if (algorithmSelect) layoutConfig.algorithm = algorithmSelect.value;
    if (directionSelect) layoutConfig.direction = directionSelect.value;
    if (spacingXInput) layoutConfig.nodeSpacingX = parseInt(spacingXInput.value) || 250;
    if (spacingYInput) layoutConfig.nodeSpacingY = parseInt(spacingYInput.value) || 150;

    updateLayoutPreview();
}

/**
 * Apply layout from panel.
 */
function applyLayoutFromPanel() {
    const animateCheckbox = document.getElementById('layoutAnimate');
    const animate = animateCheckbox ? animateCheckbox.checked : true;

    onLayoutOptionChange();
    runAutoLayout(null, null, animate);
    hideLayoutPanel();
}

/**
 * Select layout algorithm from panel.
 * @param {HTMLElement} element - The clicked option element
 */
function selectLayoutAlgorithm(element) {
    // Remove selected from all options
    const options = document.querySelectorAll('.layout-option');
    options.forEach(opt => opt.classList.remove('selected'));

    // Add selected to clicked option
    element.classList.add('selected');

    // Update config
    const algorithm = element.getAttribute('data-algorithm');
    layoutConfig.algorithm = algorithm;

    // Show/hide direction section based on algorithm
    const directionSection = document.getElementById('layoutDirectionSection');
    if (directionSection) {
        if (algorithm === 'force-directed') {
            directionSection.style.display = 'none';
        } else {
            directionSection.style.display = 'block';
        }
    }

    // Update preview
    updateLayoutPreview();
}

/**
 * Select layout direction from panel.
 * @param {HTMLElement} element - The clicked direction option element
 */
function selectLayoutDirection(element) {
    // Remove selected from all direction options
    const options = document.querySelectorAll('.layout-direction-option');
    options.forEach(opt => opt.classList.remove('selected'));

    // Add selected to clicked option
    element.classList.add('selected');

    // Update config
    const direction = element.getAttribute('data-direction');
    layoutConfig.direction = direction;

    // Update preview
    updateLayoutPreview();
}

/**
 * Update layout slider value display.
 * @param {string} type - 'spacingX' or 'spacingY'
 * @param {string|number} value - The slider value
 */
function updateLayoutSlider(type, value) {
    const numValue = parseInt(value) || 0;

    if (type === 'spacingX') {
        layoutConfig.nodeSpacingX = numValue;
        const valueDisplay = document.getElementById('spacingXValue');
        if (valueDisplay) valueDisplay.textContent = `${numValue}px`;
    } else if (type === 'spacingY') {
        layoutConfig.nodeSpacingY = numValue;
        const valueDisplay = document.getElementById('spacingYValue');
        if (valueDisplay) valueDisplay.textContent = `${numValue}px`;
    }

    // Update preview with new spacing
    updateLayoutPreview();
}

// Keyboard shortcut for layout (L key)
function handleLayoutShortcut(e) {
    if ((e.key === 'l' || e.key === 'L') && !e.ctrlKey && !e.metaKey) {
        const activeEl = document.activeElement;
        if (activeEl.tagName !== 'INPUT' && activeEl.tagName !== 'TEXTAREA') {
            e.preventDefault();
            showLayoutPanel();
        }
    }
}

document.addEventListener('keydown', handleLayoutShortcut);

function showHotkeysModal() {
    const modal = document.getElementById('hotkeysModal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

// ========== MINIMAP ==========
function updateMinimap() {
    const minimapCanvas = document.getElementById('minimapCanvas');
    if (!minimapCanvas) return;

    // Clear minimap
    minimapCanvas.innerHTML = '';

    if (nodes.length === 0) return;

    // Calculate bounding box of all nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    nodes.forEach(node => {
        minX = Math.min(minX, node.x);
        minY = Math.min(minY, node.y);
        maxX = Math.max(maxX, node.x + 200); // Node width
        maxY = Math.max(maxY, node.y + 100); // Node height
    });

    // Add padding
    minX -= 50;
    minY -= 50;
    maxX += 50;
    maxY += 50;

    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;

    // Calculate scale to fit minimap
    const minimap = document.getElementById('minimap');
    const minimapWidth = minimap.clientWidth;
    const minimapHeight = minimap.clientHeight;
    const scale = Math.min(minimapWidth / contentWidth, minimapHeight / contentHeight);

    // Draw nodes
    nodes.forEach(node => {
        const nodeEl = document.createElement('div');
        nodeEl.className = 'minimap-node';
        const x = (node.x - minX) * scale;
        const y = (node.y - minY) * scale;
        nodeEl.style.left = x + 'px';
        nodeEl.style.top = y + 'px';
        nodeEl.style.background = node.definition.color || '#667eea';
        minimapCanvas.appendChild(nodeEl);
    });

    // Draw viewport indicator
    const viewport = document.getElementById('canvasViewport');
    const viewportEl = document.createElement('div');
    viewportEl.className = 'minimap-viewport';
    viewportEl.id = 'minimapViewport';

    const viewportX = (viewport.scrollLeft - minX) * scale;
    const viewportY = (viewport.scrollTop - minY) * scale;
    const viewportWidth = (viewport.clientWidth / zoomLevel) * scale;
    const viewportHeight = (viewport.clientHeight / zoomLevel) * scale;

    viewportEl.style.left = Math.max(0, viewportX) + 'px';
    viewportEl.style.top = Math.max(0, viewportY) + 'px';
    viewportEl.style.width = Math.min(viewportWidth, minimapWidth) + 'px';
    viewportEl.style.height = Math.min(viewportHeight, minimapHeight) + 'px';

    minimapCanvas.appendChild(viewportEl);

    // Store scale and offset for click handling
    minimapCanvas.dataset.scale = scale;
    minimapCanvas.dataset.offsetX = minX;
    minimapCanvas.dataset.offsetY = minY;
}

// Click minimap to navigate
document.addEventListener('DOMContentLoaded', () => {
    const minimap = document.getElementById('minimap');
    if (minimap) {
        minimap.addEventListener('click', (e) => {
            const canvas = document.getElementById('minimapCanvas');
            const viewport = document.getElementById('canvasViewport');
            if (!canvas || !viewport) return;

            const rect = minimap.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const scale = parseFloat(canvas.dataset.scale || 1);
            const offsetX = parseFloat(canvas.dataset.offsetX || 0);
            const offsetY = parseFloat(canvas.dataset.offsetY || 0);

            // Convert minimap coordinates to canvas coordinates
            const canvasX = (x / scale) + offsetX;
            const canvasY = (y / scale) + offsetY;

            // Center viewport on clicked position
            viewport.scrollLeft = canvasX - (viewport.clientWidth / 2);
            viewport.scrollTop = canvasY - (viewport.clientHeight / 2);

            updateMinimap();
        });
    }

    // Update minimap on scroll
    const viewport = document.getElementById('canvasViewport');
    if (viewport) {
        viewport.addEventListener('scroll', updateMinimap);
    }
});

// Call updateMinimap when nodes/connections change
const originalRenderNode = renderNode;
renderNode = function(...args) {
    const result = originalRenderNode.apply(this, args);
    setTimeout(updateMinimap, 0);
    return result;
};

const originalDeleteNode = deleteNode;
deleteNode = function(...args) {
    const result = originalDeleteNode.apply(this, args);
    setTimeout(updateMinimap, 0);
    return result;
};

// ========== WORKFLOW TEMPLATES ==========
let workflowTemplates = [];

// Load built-in templates
async function loadBuiltInTemplates() {
    const builtInTemplates = [
        {
            id: 'crm_lead_scoring',
            name: 'CRM: Simple Lead Scoring',
            category: 'crm',
            description: 'Score leads based on activity and engagement',
            file: 'example_workflows/crm/lead_scoring_simple.json',
            tags: ['crm', 'lead-scoring', 'basic']
        },
        {
            id: 'crm_daily_actions',
            name: 'CRM: Daily Action List',
            category: 'crm',
            description: 'Generate prioritized daily action list',
            file: 'example_workflows/crm/daily_actions.json',
            tags: ['crm', 'automation', 'productivity']
        },
        {
            id: 'crm_multi_factor',
            name: 'CRM: Multi-Factor Lead Scoring',
            category: 'crm',
            description: 'Advanced lead scoring with Thompson Sampling',
            file: 'example_workflows/crm/multi_factor_scoring.json',
            tags: ['crm', 'lead-scoring', 'advanced', 'thompson-sampling']
        },
        {
            id: 'llm_content_creation',
            name: 'LLM: Content Creation Pipeline',
            category: 'llm',
            description: 'Multi-step blog post creation with LLM chains',
            file: 'example_workflows/llm/content_creation.json',
            tags: ['llm', 'content', 'writing', 'prompt-chain']
        },
        {
            id: 'llm_customer_support',
            name: 'LLM: Customer Support Triage',
            category: 'llm',
            description: 'Intelligent ticket routing with consensus',
            file: 'example_workflows/llm/customer_support_triage.json',
            tags: ['llm', 'support', 'triage', 'consensus']
        },
        {
            id: 'research_multi_query',
            name: 'Research: Multi-Query Exploration',
            category: 'research',
            description: 'Break complex research into sub-queries',
            nodes: 3,
            tags: ['research', 'multi-query', 'exploration']
        },
        {
            id: 'automation_pipeline',
            name: 'Automation: Data Processing Pipeline',
            category: 'automation',
            description: 'Sequential data processing with error handling',
            nodes: 5,
            tags: ['automation', 'data', 'pipeline']
        }
    ];

    workflowTemplates = builtInTemplates;

    // Load custom templates from localStorage
    const customTemplates = JSON.parse(localStorage.getItem('customWorkflowTemplates') || '[]');
    workflowTemplates = [...builtInTemplates, ...customTemplates];
}

function showTemplatesModal() {
    loadBuiltInTemplates();
    renderTemplatesList();
    document.getElementById('templatesModal').style.display = 'flex';
}

function renderTemplatesList() {
    const search = document.getElementById('templateSearch').value.toLowerCase();
    const category = document.getElementById('templateCategory').value;

    const filtered = workflowTemplates.filter(t => {
        const matchesSearch = !search ||
            t.name.toLowerCase().includes(search) ||
            t.description.toLowerCase().includes(search) ||
            t.tags.some(tag => tag.includes(search));
        const matchesCategory = category === 'all' || t.category === category;
        return matchesSearch && matchesCategory;
    });

    const list = document.getElementById('templatesList');
    list.innerHTML = '';

    if (filtered.length === 0) {
        list.innerHTML = '<div style="text-align: center; padding: 40px; color: #999;">No templates found</div>';
        return;
    }

    filtered.forEach(template => {
        const item = document.createElement('div');
        item.style.cssText = `
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s;
        `;

        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <div style="font-weight: 600; font-size: 14px; color: #333; margin-bottom: 5px;">
                        ${template.name}
                    </div>
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">
                        ${template.description}
                    </div>
                    <div style="display: flex; gap: 5px; flex-wrap: wrap;">
                        ${template.tags.map(tag => `
                            <span style="background: #f0f0f0; padding: 2px 8px; border-radius: 4px; font-size: 11px; color: #666;">
                                ${tag}
                            </span>
                        `).join('')}
                    </div>
                </div>
                <button class="toolbar-btn" onclick="loadTemplate('${template.id}'); event.stopPropagation();">
                    Load
                </button>
            </div>
        `;

        item.onmouseover = () => item.style.borderColor = '#667eea';
        item.onmouseout = () => item.style.borderColor = '#e0e0e0';
        item.onclick = () => loadTemplate(template.id);

        list.appendChild(item);
    });
}

function filterTemplates() {
    renderTemplatesList();
}

async function loadTemplate(templateId) {
    const template = workflowTemplates.find(t => t.id === templateId);
    if (!template) {
        showToast('Template not found', 'error');
        return;
    }

    try {
        if (template.file) {
            // Load from file
            const response = await fetch(template.file);
            if (!response.ok) {
                showToast(`Template file not found: ${template.file}`, 'error');
                return;
            }
            const workflow = await response.json();

            // Clear canvas and load workflow
            clearCanvas();
            nodes.length = 0;
            connections.length = 0;

            workflow.nodes.forEach(node => {
                nodes.push(node);
                renderNode(node);
            });

            workflow.connections.forEach(conn => {
                connections.push(conn);
            });

            renderConnections();
            document.querySelector('.workflow-title').textContent = template.name;
            showToast(`Loaded template: ${template.name}`, 'success');
        } else if (template.workflow) {
            // Load from stored workflow object (custom templates)
            clearCanvas();
            nodes.length = 0;
            connections.length = 0;

            template.workflow.nodes.forEach(node => {
                nodes.push(node);
                renderNode(node);
            });

            template.workflow.connections.forEach(conn => {
                connections.push(conn);
            });

            renderConnections();
            document.querySelector('.workflow-title').textContent = template.name;
            showToast(`Loaded template: ${template.name}`, 'success');
        }

        closeModal('templatesModal');
    } catch (error) {
        console.error('Failed to load template:', error);
        showToast('Failed to load template', 'error');
    }
}

function saveAsTemplate() {
    if (nodes.length === 0) {
        showToast('Cannot save empty workflow as template', 'error');
        return;
    }

    const name = prompt('Template name:');
    if (!name) return;

    const description = prompt('Template description:');
    const tags = prompt('Tags (comma-separated):')?.split(',').map(t => t.trim()) || [];

    const customTemplate = {
        id: `custom_${Date.now()}`,
        name: name,
        category: 'custom',
        description: description || 'Custom workflow template',
        tags: tags,
        workflow: {
            version: '1.0',
            name: name,
            nodes: JSON.parse(JSON.stringify(nodes)),
            connections: JSON.parse(JSON.stringify(connections))
        }
    };

    // Save to localStorage
    const customTemplates = JSON.parse(localStorage.getItem('customWorkflowTemplates') || '[]');
    customTemplates.push(customTemplate);
    localStorage.setItem('customWorkflowTemplates', JSON.stringify(customTemplates));

    showToast(`Saved template: ${name}`, 'success');
    closeModal('templatesModal');
}

// ========== AI SUGGESTIONS (Wave 3.2) ==========

let suggestionOverlay = null;
let suggestionCache = {};
let suggestionCacheTimeout = 30000; // 30 second cache

/**
 * Show AI suggestions when hovering over a node's output port
 */
async function showSuggestions(nodeId, x, y) {
    // Get current workflow nodes in execution order (topological)
    const currentNodes = getExecutionOrder();
    const nodeTypes = currentNodes.map(n => n.agentType);

    // Find position of this node in the workflow
    const nodeIndex = currentNodes.findIndex(n => n.id === nodeId);
    const relevantTypes = nodeIndex >= 0 ? nodeTypes.slice(0, nodeIndex + 1) : nodeTypes;

    // Check cache
    const cacheKey = relevantTypes.join(',');
    if (suggestionCache[cacheKey] && Date.now() - suggestionCache[cacheKey].timestamp < suggestionCacheTimeout) {
        displaySuggestionOverlay(suggestionCache[cacheKey].data, x, y);
        return;
    }

    try {
        const response = await fetch(`http://localhost:8001/api/suggestions/next-nodes?current=${relevantTypes.join(',')}&k=5`);
        const data = await response.json();

        // Cache result
        suggestionCache[cacheKey] = { data, timestamp: Date.now() };

        displaySuggestionOverlay(data, x, y);
    } catch (error) {
        console.warn('Failed to fetch suggestions:', error);
    }
}

/**
 * Display the suggestion overlay
 */
function displaySuggestionOverlay(data, x, y) {
    hideSuggestions();

    if (!data.suggestions || data.suggestions.length === 0) {
        return;
    }

    suggestionOverlay = document.createElement('div');
    suggestionOverlay.className = 'suggestion-overlay';
    suggestionOverlay.style.cssText = `
        position: absolute;
        left: ${x + 10}px;
        top: ${y - 10}px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #4a4a6a;
        border-radius: 8px;
        padding: 12px;
        min-width: 280px;
        max-width: 350px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        z-index: 10000;
        font-size: 13px;
        animation: suggestionFadeIn 0.2s ease-out;
    `;

    let html = `
        <div style="margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
            <span style="font-weight: 600; color: #e94560;">AI Suggestions</span>
            <span style="font-size: 11px; color: #888;">Thompson Sampling</span>
        </div>
    `;

    data.suggestions.forEach((suggestion, index) => {
        const confidenceColor = suggestion.confidence >= 0.7 ? '#4ade80' :
                              suggestion.confidence >= 0.4 ? '#fbbf24' : '#f87171';
        const agentInfo = agentTypes[suggestion.node_type] || { name: suggestion.node_type };

        html += `
            <div class="suggestion-item"
                 onclick="addSuggestedNode('${suggestion.node_type}')"
                 style="
                    padding: 10px;
                    margin: 6px 0;
                    background: rgba(255,255,255,0.05);
                    border-radius: 6px;
                    cursor: pointer;
                    transition: all 0.15s ease;
                    border-left: 3px solid ${confidenceColor};
                 "
                 onmouseover="this.style.background='rgba(233,69,96,0.15)'"
                 onmouseout="this.style.background='rgba(255,255,255,0.05)'">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500; color: #fff;">${agentInfo.name || suggestion.node_type}</span>
                    <span style="font-size: 11px; color: ${confidenceColor};">
                        ${(suggestion.confidence * 100).toFixed(0)}% confidence
                    </span>
                </div>
                <div style="font-size: 11px; color: #888; margin-top: 4px;">
                    Used ${suggestion.frequency}x | Success: ${(suggestion.success_rate * 100).toFixed(0)}%
                </div>
            </div>
        `;
    });

    // Add learn button if few patterns
    if (data.patterns_available < 10) {
        html += `
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">
                <div style="font-size: 11px; color: #888; margin-bottom: 6px;">
                    ${data.patterns_available} patterns learned
                </div>
                <button onclick="triggerPatternLearning()" style="
                    background: linear-gradient(135deg, #0f3460, #16213e);
                    border: 1px solid #4a4a6a;
                    color: #fff;
                    padding: 6px 12px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 11px;
                    width: 100%;
                ">Learn from History</button>
            </div>
        `;
    }

    suggestionOverlay.innerHTML = html;

    // Add click outside listener
    suggestionOverlay.addEventListener('mouseleave', () => {
        setTimeout(() => {
            if (suggestionOverlay && !suggestionOverlay.matches(':hover')) {
                hideSuggestions();
            }
        }, 500);
    });

    document.body.appendChild(suggestionOverlay);
}

/**
 * Hide suggestion overlay
 */
function hideSuggestions() {
    if (suggestionOverlay) {
        suggestionOverlay.remove();
        suggestionOverlay = null;
    }
}

/**
 * Add a suggested node to the canvas
 */
function addSuggestedNode(nodeType) {
    hideSuggestions();

    // Find the last selected node or use canvas center
    const lastNode = selectedNodes.size > 0 ?
        nodes.find(n => selectedNodes.has(n.id)) :
        nodes[nodes.length - 1];

    let x, y;
    if (lastNode) {
        x = lastNode.x + 300;
        y = lastNode.y;
    } else {
        x = 400;
        y = 300;
    }

    // Create the node
    createNode(nodeType, x, y);

    // Auto-connect to last selected if exists
    if (lastNode) {
        connections.push({
            id: generateId(),
            from: lastNode.id,
            to: nodes[nodes.length - 1].id
        });
        renderConnections();
    }

    showToast(`Added ${nodeType} node`, 'success');
}

/**
 * Trigger pattern learning from history
 */
async function triggerPatternLearning() {
    try {
        const response = await fetch('http://localhost:8001/api/patterns/learn', { method: 'POST' });
        const data = await response.json();

        if (data.status === 'success') {
            showToast(`Learned ${data.patterns_learned.pairs} pair patterns`, 'success');
            // Clear cache
            suggestionCache = {};
        } else {
            showToast('No workflow history to learn from', 'info');
        }
    } catch (error) {
        showToast('Failed to trigger learning', 'error');
    }
}

/**
 * Get nodes in execution order (topological sort)
 */
function getExecutionOrder() {
    const visited = new Set();
    const result = [];

    // Find start nodes
    const nodesWithInputs = new Set(connections.map(c => c.to));
    const startNodes = nodes.filter(n => !nodesWithInputs.has(n.id));

    function visit(node) {
        if (visited.has(node.id)) return;
        visited.add(node.id);

        // Visit dependencies first
        const deps = connections.filter(c => c.to === node.id).map(c => c.from);
        deps.forEach(depId => {
            const depNode = nodes.find(n => n.id === depId);
            if (depNode) visit(depNode);
        });

        result.push(node);
    }

    startNodes.forEach(visit);
    // Visit any remaining unvisited nodes
    nodes.forEach(n => {
        if (!visited.has(n.id)) visit(n);
    });

    return result;
}

/**
 * Show suggestions for a specific port
 */
function initializeSuggestionListeners() {
    // Add hover listener to canvas for output ports
    const canvas = document.getElementById('workflowCanvas');

    canvas.addEventListener('mouseover', (e) => {
        const port = e.target.closest('.output-port, .node-port.output');
        if (port) {
            const nodeEl = port.closest('.workflow-node');
            if (nodeEl) {
                const nodeId = nodeEl.id.replace('node-', '');
                const rect = port.getBoundingClientRect();
                showSuggestions(nodeId, rect.right, rect.top);
            }
        }
    });

    canvas.addEventListener('mouseout', (e) => {
        const port = e.target.closest('.output-port, .node-port.output');
        if (port && suggestionOverlay && !suggestionOverlay.matches(':hover')) {
            setTimeout(() => {
                if (suggestionOverlay && !suggestionOverlay.matches(':hover')) {
                    hideSuggestions();
                }
            }, 300);
        }
    });
}

// Add CSS for suggestion animation
const suggestionStyles = document.createElement('style');
suggestionStyles.textContent = `
    @keyframes suggestionFadeIn {
        from { opacity: 0; transform: translateY(-5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .suggestion-overlay {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .suggestion-item:active {
        transform: scale(0.98);
    }
`;
document.head.appendChild(suggestionStyles);

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSuggestionListeners);
} else {
    initializeSuggestionListeners();
}

// ========== SEARCH & NAVIGATION (Wave 5.1) ==========

// State for search and command palette
let searchResults = [];
let searchHighlightedNodes = [];
let commandPaletteSelectedIndex = 0;
let commandPaletteItems = [];
let recentAgents = [];
let currentGroupPath = []; // Breadcrumb path for group navigation

// 5.1.1: Node Search Functions
function focusNodeSearch() {
    const input = document.getElementById('nodeSearchInput');
    if (input) {
        input.focus();
        input.select();
    }
}

function handleNodeSearch(query) {
    // Clear previous highlights
    clearSearchHighlights();

    if (!query || query.trim().length === 0) {
        hideSearchResults();
        return;
    }

    // Fuzzy search across nodes
    searchResults = fuzzySearchNodes(query.toLowerCase());

    // Show results dropdown
    renderSearchResults(searchResults);

    // Highlight matching nodes on canvas
    searchResults.forEach(result => {
        highlightNode(result.node.id);
    });
}

function fuzzySearchNodes(query) {
    const results = [];

    nodes.forEach(node => {
        const searchFields = [
            node.definition?.name || '',
            node.definition?.category || '',
            node.definition?.description || '',
            node.config?.label || '',
            node.id,
            ...(node.config ? Object.values(node.config).map(v => String(v)) : [])
        ].map(s => s.toLowerCase());

        // Calculate match score
        let score = 0;
        let matchedField = '';

        searchFields.forEach((field, idx) => {
            if (field.includes(query)) {
                // Exact match gets higher score
                const fieldScore = field === query ? 100 :
                                   field.startsWith(query) ? 80 : 50;
                if (fieldScore > score) {
                    score = fieldScore;
                    matchedField = field;
                }
            } else {
                // Fuzzy match (all characters present in order)
                const fuzzyScore = fuzzyMatch(query, field);
                if (fuzzyScore > score) {
                    score = fuzzyScore;
                    matchedField = field;
                }
            }
        });

        if (score > 20) {
            results.push({
                node,
                score,
                matchedField
            });
        }
    });

    // Sort by score descending
    return results.sort((a, b) => b.score - a.score).slice(0, 10);
}

function fuzzyMatch(query, text) {
    let queryIdx = 0;
    let score = 0;
    let consecutive = 0;

    for (let i = 0; i < text.length && queryIdx < query.length; i++) {
        if (text[i] === query[queryIdx]) {
            score += 10 + consecutive * 5;
            consecutive++;
            queryIdx++;
        } else {
            consecutive = 0;
        }
    }

    return queryIdx === query.length ? score : 0;
}

function renderSearchResults(results) {
    const container = document.getElementById('searchResults');
    if (!container) return;

    if (results.length === 0) {
        container.innerHTML = '<div class="search-result-item" style="color: #666;">No nodes found</div>';
        container.style.display = 'block';
        return;
    }

    container.innerHTML = results.map((result, idx) => `
        <div class="search-result-item${idx === 0 ? ' selected' : ''}"
             onclick="jumpToNode('${result.node.id}')"
             data-node-id="${result.node.id}">
            <span class="search-result-icon">${result.node.definition?.icon || '📦'}</span>
            <div class="search-result-info">
                <div class="search-result-name">${result.node.config?.label || result.node.definition?.name || result.node.id}</div>
                <div class="search-result-type">${result.node.definition?.category || 'Node'}</div>
            </div>
        </div>
    `).join('');

    container.style.display = 'block';
}

function showSearchResults() {
    const container = document.getElementById('searchResults');
    const input = document.getElementById('nodeSearchInput');
    if (container && input && input.value.trim()) {
        container.style.display = 'block';
    }
}

function hideSearchResults() {
    const container = document.getElementById('searchResults');
    if (container) {
        container.style.display = 'none';
    }
}

function highlightNode(nodeId) {
    const nodeEl = document.getElementById(nodeId);
    if (nodeEl) {
        nodeEl.classList.add('search-highlight');
        searchHighlightedNodes.push(nodeId);
    }
}

function clearSearchHighlights() {
    searchHighlightedNodes.forEach(nodeId => {
        const nodeEl = document.getElementById(nodeId);
        if (nodeEl) {
            nodeEl.classList.remove('search-highlight');
        }
    });
    searchHighlightedNodes = [];
}

function jumpToNode(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    // If node is in a group, navigate to that group first
    if (node.parentGroup) {
        navigateToGroup(node.parentGroup);
    }

    // Center the canvas on the node
    const canvas = document.getElementById('canvas');
    if (canvas) {
        const canvasRect = canvas.getBoundingClientRect();
        const targetX = canvasRect.width / 2 - node.x;
        const targetY = canvasRect.height / 2 - node.y;

        // Animate pan to node
        canvas.style.transition = 'transform 0.3s ease-out';
        canvas.scrollLeft = node.x - canvasRect.width / 2 + 150;
        canvas.scrollTop = node.y - canvasRect.height / 2 + 75;
    }

    // Select the node
    deselectAll();
    selectNode(nodeId);

    // Clear search
    const input = document.getElementById('nodeSearchInput');
    if (input) {
        input.value = '';
    }
    hideSearchResults();
    clearSearchHighlights();

    showToast(`Jumped to ${node.definition?.name || nodeId}`, 'info');
}

// 5.1.2: Command Palette Functions
function openCommandPalette() {
    const palette = document.getElementById('commandPalette');
    if (!palette) return;

    palette.classList.add('active');

    const input = document.getElementById('commandPaletteInput');
    if (input) {
        input.value = '';
        input.focus();
    }

    // Populate with initial content
    populateCommandPalette('');
}

function closeCommandPalette() {
    const palette = document.getElementById('commandPalette');
    if (palette) {
        palette.classList.remove('active');
    }
}

function populateCommandPalette(query) {
    commandPaletteItems = [];
    commandPaletteSelectedIndex = 0;

    const recentSection = document.getElementById('recentAgentsSection');
    const allSection = document.getElementById('allAgentsSection');
    const actionsSection = document.getElementById('actionsSection');

    if (!recentSection || !allSection || !actionsSection) return;

    // Filter agents based on query
    const matchingAgents = Object.entries(agentDefinitions)
        .filter(([type, def]) => {
            if (!query) return true;
            const searchText = `${def.name} ${def.category} ${def.description}`.toLowerCase();
            return searchText.includes(query.toLowerCase());
        })
        .sort((a, b) => a[1].name.localeCompare(b[1].name));

    // Recent agents (last 5 unique types used)
    const recentHtml = recentAgents.slice(0, 5).map(type => {
        const def = agentDefinitions[type];
        if (!def || (query && !`${def.name} ${def.category}`.toLowerCase().includes(query.toLowerCase()))) return '';
        commandPaletteItems.push({ type: 'agent', agentType: type });
        return `
            <div class="command-item" onclick="addAgentFromPalette('${type}')">
                <span class="command-item-icon">${def.icon}</span>
                <div class="command-item-info">
                    <div class="command-item-name">${def.name}</div>
                    <div class="command-item-desc">${def.description.substring(0, 50)}...</div>
                </div>
                <span class="command-item-kbd">↵</span>
            </div>
        `;
    }).filter(h => h).join('');

    recentSection.innerHTML = recentHtml || '<div style="color: #999; font-size: 12px; padding: 8px;">No recent agents</div>';

    // All agents
    const allHtml = matchingAgents.map(([type, def]) => {
        commandPaletteItems.push({ type: 'agent', agentType: type });
        return `
            <div class="command-item" onclick="addAgentFromPalette('${type}')">
                <span class="command-item-icon">${def.icon}</span>
                <div class="command-item-info">
                    <div class="command-item-name">${def.name}</div>
                    <div class="command-item-desc">${def.category}</div>
                </div>
            </div>
        `;
    }).join('');

    allSection.innerHTML = allHtml || '<div style="color: #999; font-size: 12px; padding: 8px;">No matching agents</div>';

    // Actions
    const actions = [
        { id: 'clear', icon: '🗑️', name: 'Clear Canvas', desc: 'Remove all nodes', action: 'clearCanvas()' },
        { id: 'export', icon: '💾', name: 'Export Workflow', desc: 'Save as JSON', action: 'showExportModal()' },
        { id: 'import', icon: '📁', name: 'Import Workflow', desc: 'Load from file', action: 'importWorkflow()' },
        { id: 'undo', icon: '↩️', name: 'Undo', desc: 'Revert last change', action: 'undo()' },
        { id: 'redo', icon: '↪️', name: 'Redo', desc: 'Restore last undo', action: 'redo()' },
        { id: 'execute', icon: '▶️', name: 'Execute Workflow', desc: 'Run the workflow', action: 'executeWorkflow()' },
        { id: 'validate', icon: '✓', name: 'Validate', desc: 'Check for errors', action: 'validateWorkflow()' },
        { id: 'layout', icon: '📐', name: 'Auto-Layout', desc: 'Arrange nodes', action: 'showLayoutPanel()' },
        { id: 'perf', icon: '📊', name: 'Performance Dashboard', desc: 'View metrics', action: 'showPerfDashboard()' },
    ].filter(a => !query || `${a.name} ${a.desc}`.toLowerCase().includes(query.toLowerCase()));

    const actionsHtml = actions.map(a => {
        commandPaletteItems.push({ type: 'action', action: a.action });
        return `
            <div class="command-item" onclick="${a.action}; closeCommandPalette();">
                <span class="command-item-icon">${a.icon}</span>
                <div class="command-item-info">
                    <div class="command-item-name">${a.name}</div>
                    <div class="command-item-desc">${a.desc}</div>
                </div>
            </div>
        `;
    }).join('');

    actionsSection.innerHTML = actionsHtml || '<div style="color: #999; font-size: 12px; padding: 8px;">No matching actions</div>';

    // Update selection highlight
    updatePaletteSelection();
}

function handleCommandSearch(query) {
    populateCommandPalette(query);
}

function handleCommandKeydown(e) {
    const items = document.querySelectorAll('#commandPalette .command-item');

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        commandPaletteSelectedIndex = Math.min(commandPaletteSelectedIndex + 1, items.length - 1);
        updatePaletteSelection();
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        commandPaletteSelectedIndex = Math.max(commandPaletteSelectedIndex - 1, 0);
        updatePaletteSelection();
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (commandPaletteItems[commandPaletteSelectedIndex]) {
            const item = commandPaletteItems[commandPaletteSelectedIndex];
            if (item.type === 'agent') {
                addAgentFromPalette(item.agentType);
            } else if (item.type === 'action') {
                eval(item.action);
                closeCommandPalette();
            }
        }
    }
}

function updatePaletteSelection() {
    const items = document.querySelectorAll('#commandPalette .command-item');
    items.forEach((item, idx) => {
        item.classList.toggle('selected', idx === commandPaletteSelectedIndex);
        if (idx === commandPaletteSelectedIndex) {
            item.scrollIntoView({ block: 'nearest' });
        }
    });
}

function addAgentFromPalette(agentType) {
    const canvas = document.getElementById('canvas');
    const rect = canvas?.getBoundingClientRect();

    // Add to center of visible canvas
    const x = (rect?.width || 400) / 2 - 100;
    const y = (rect?.height || 300) / 2 - 50;

    addNode(agentType, x, y);

    // Track recent agents
    recentAgents = [agentType, ...recentAgents.filter(t => t !== agentType)].slice(0, 10);

    closeCommandPalette();
    showToast(`Added ${agentDefinitions[agentType]?.name || agentType}`, 'success');
}

// 5.1.3: Breadcrumb Navigation Functions
function navigateToRoot() {
    currentGroupPath = [];
    updateBreadcrumbs();

    // Show all top-level nodes
    nodes.forEach(node => {
        const el = document.getElementById(node.id);
        if (el) {
            el.style.display = node.parentGroup ? 'none' : 'block';
        }
    });

    renderConnections();
    showToast('Navigated to root', 'info');
}

function navigateToGroup(groupId) {
    const groupNode = nodes.find(n => n.id === groupId);
    if (!groupNode || groupNode.agentType !== 'group') return;

    // Build path to this group
    const path = [];
    let current = groupNode;
    while (current) {
        path.unshift({ id: current.id, name: current.config?.label || current.definition?.name || current.id });
        current = current.parentGroup ? nodes.find(n => n.id === current.parentGroup) : null;
    }

    currentGroupPath = path;
    updateBreadcrumbs();

    // Show only nodes in this group
    nodes.forEach(node => {
        const el = document.getElementById(node.id);
        if (el) {
            const isInGroup = node.parentGroup === groupId;
            const isTheGroup = node.id === groupId;
            el.style.display = (isInGroup || isTheGroup) ? 'block' : 'none';
        }
    });

    renderConnections();
    showToast(`Entered group: ${groupNode.config?.label || groupId}`, 'info');
}

function navigateToBreadcrumb(index) {
    if (index < 0) {
        navigateToRoot();
    } else if (index < currentGroupPath.length - 1) {
        const groupId = currentGroupPath[index].id;
        navigateToGroup(groupId);
    }
    // If clicking current (last) breadcrumb, do nothing
}

function updateBreadcrumbs() {
    const nav = document.getElementById('breadcrumbNav');
    const pathContainer = document.getElementById('breadcrumbPath');

    if (!nav || !pathContainer) return;

    if (currentGroupPath.length === 0) {
        nav.style.display = 'none';
        return;
    }

    nav.style.display = 'flex';

    const pathHtml = currentGroupPath.map((item, idx) => `
        <span class="breadcrumb-separator">›</span>
        <span class="breadcrumb-item${idx === currentGroupPath.length - 1 ? ' active' : ''}"
              onclick="navigateToBreadcrumb(${idx})">
            ${item.name}
        </span>
    `).join('');

    pathContainer.innerHTML = pathHtml;
}

// Initialize search on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Search is ready
    });
}

// =============================================================================
// Wave 5.2: Context Menus & UX Polish
// =============================================================================

// Context menu state
let contextMenuTargetNode = null;
let contextMenuTargetConnection = null;
let contextMenuPosition = { x: 0, y: 0 };
let selectedNodes = new Set();
let copiedNodes = [];
let autoSaveInterval = null;
let autoSaveEnabled = true;
let lastSaveTime = Date.now();
const AUTO_SAVE_INTERVAL = 30000; // 30 seconds

// Hide all context menus
function hideAllContextMenus() {
    const menus = ['nodeContextMenu', 'canvasContextMenu', 'connectionContextMenu', 'multiSelectContextMenu'];
    menus.forEach(id => {
        const menu = document.getElementById(id);
        if (menu) menu.style.display = 'none';
    });
}

// Show context menu at position
function showContextMenu(menuId, x, y) {
    hideAllContextMenus();
    const menu = document.getElementById(menuId);
    if (!menu) return;

    // Position menu
    menu.style.left = `${x}px`;
    menu.style.top = `${y}px`;
    menu.style.display = 'block';

    // Ensure menu stays within viewport
    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
        menu.style.left = `${x - rect.width}px`;
    }
    if (rect.bottom > window.innerHeight) {
        menu.style.top = `${y - rect.height}px`;
    }
}

// Handle canvas right-click
function handleCanvasContextMenu(e) {
    e.preventDefault();

    const canvas = document.getElementById('workflowCanvas');
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Check if clicked on a node
    const clickedNode = findNodeAtPosition(clickX, clickY);

    // Check if clicked on a connection
    const clickedConnection = findConnectionAtPosition(clickX, clickY);

    contextMenuPosition = { x: clickX, y: clickY };

    if (clickedNode) {
        contextMenuTargetNode = clickedNode;
        if (selectedNodes.size > 1 && selectedNodes.has(clickedNode.id)) {
            // Multiple nodes selected, show multi-select menu
            showContextMenu('multiSelectContextMenu', e.clientX, e.clientY);
        } else {
            // Single node
            showContextMenu('nodeContextMenu', e.clientX, e.clientY);
        }
    } else if (clickedConnection) {
        contextMenuTargetConnection = clickedConnection;
        showContextMenu('connectionContextMenu', e.clientX, e.clientY);
    } else {
        // Empty canvas area
        showContextMenu('canvasContextMenu', e.clientX, e.clientY);
    }
}

// Find node at canvas position
function findNodeAtPosition(x, y) {
    // Convert from screen coordinates to node coordinates
    const transformedX = (x - canvasTransform.offsetX) / canvasTransform.scale;
    const transformedY = (y - canvasTransform.offsetY) / canvasTransform.scale;

    for (const node of nodes) {
        const nodeWidth = node.type === 'group' ? (node.config?.width || 300) : 220;
        const nodeHeight = node.type === 'group' ? (node.config?.height || 200) : 120;

        if (transformedX >= node.x && transformedX <= node.x + nodeWidth &&
            transformedY >= node.y && transformedY <= node.y + nodeHeight) {
            return node;
        }
    }
    return null;
}

// Find connection at canvas position
function findConnectionAtPosition(x, y) {
    const transformedX = (x - canvasTransform.offsetX) / canvasTransform.scale;
    const transformedY = (y - canvasTransform.offsetY) / canvasTransform.scale;
    const threshold = 10;

    for (const conn of connections) {
        const fromNode = nodes.find(n => n.id === conn.from);
        const toNode = nodes.find(n => n.id === conn.to);
        if (!fromNode || !toNode) continue;

        const fromWidth = fromNode.type === 'group' ? (fromNode.config?.width || 300) : 220;
        const toWidth = toNode.type === 'group' ? (toNode.config?.width || 300) : 220;

        const startX = fromNode.x + fromWidth;
        const startY = fromNode.y + 60;
        const endX = toNode.x;
        const endY = toNode.y + 60;

        // Check distance to line segment
        const dist = pointToLineDistance(transformedX, transformedY, startX, startY, endX, endY);
        if (dist < threshold) {
            return conn;
        }
    }
    return null;
}

// Calculate distance from point to line segment
function pointToLineDistance(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lengthSq = dx * dx + dy * dy;

    if (lengthSq === 0) return Math.hypot(px - x1, py - y1);

    let t = ((px - x1) * dx + (py - y1) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));

    const closestX = x1 + t * dx;
    const closestY = y1 + t * dy;

    return Math.hypot(px - closestX, py - closestY);
}

// Context menu actions - Node
function contextMenuEditConfig() {
    hideAllContextMenus();
    if (contextMenuTargetNode) {
        editNodeConfig(contextMenuTargetNode.id);
    }
}

function contextMenuDuplicate() {
    hideAllContextMenus();
    if (contextMenuTargetNode) {
        duplicateNode(contextMenuTargetNode);
    }
}

function contextMenuCopy() {
    hideAllContextMenus();
    if (selectedNodes.size > 0) {
        copiedNodes = Array.from(selectedNodes).map(id => {
            const node = nodes.find(n => n.id === id);
            return node ? JSON.parse(JSON.stringify(node)) : null;
        }).filter(n => n);
    } else if (contextMenuTargetNode) {
        copiedNodes = [JSON.parse(JSON.stringify(contextMenuTargetNode))];
    }
    showToast(`Copied ${copiedNodes.length} node(s)`, 'info');
}

function contextMenuAddToGroup() {
    hideAllContextMenus();
    if (contextMenuTargetNode) {
        // Find available groups
        const groups = nodes.filter(n => n.type === 'group' && n.id !== contextMenuTargetNode.id);
        if (groups.length === 0) {
            showToast('No groups available. Create a group first.', 'warning');
            return;
        }
        // For now, add to first available group. Could show a picker modal.
        addNodeToGroup(contextMenuTargetNode.id, groups[0].id);
    }
}

function contextMenuToggleBreakpoint() {
    hideAllContextMenus();
    if (contextMenuTargetNode) {
        contextMenuTargetNode.breakpoint = !contextMenuTargetNode.breakpoint;
        renderWorkflow();
        showToast(contextMenuTargetNode.breakpoint ? 'Breakpoint set' : 'Breakpoint removed', 'info');
    }
}

function contextMenuDisconnectAll() {
    hideAllContextMenus();
    if (contextMenuTargetNode) {
        saveUndoState();
        const toRemove = connections.filter(c =>
            c.from === contextMenuTargetNode.id || c.to === contextMenuTargetNode.id
        );
        connections = connections.filter(c =>
            c.from !== contextMenuTargetNode.id && c.to !== contextMenuTargetNode.id
        );
        renderWorkflow();
        showToast(`Removed ${toRemove.length} connection(s)`, 'success');
    }
}

function contextMenuDeleteNode() {
    hideAllContextMenus();
    if (contextMenuTargetNode) {
        deleteNode(contextMenuTargetNode.id);
    }
}

// Context menu actions - Canvas
function contextMenuAddNodeHere() {
    hideAllContextMenus();
    // Open command palette with position context
    openCommandPalette();
    // Store position for when node is added
    window.pendingNodePosition = contextMenuPosition;
}

function contextMenuPaste() {
    hideAllContextMenus();
    if (copiedNodes.length === 0) {
        showToast('Nothing to paste', 'warning');
        return;
    }

    saveUndoState();

    // Calculate offset from original position
    const offsetX = contextMenuPosition.x - (copiedNodes[0].x * canvasTransform.scale + canvasTransform.offsetX);
    const offsetY = contextMenuPosition.y - (copiedNodes[0].y * canvasTransform.scale + canvasTransform.offsetY);

    const idMapping = {};

    copiedNodes.forEach(nodeToCopy => {
        const newId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        idMapping[nodeToCopy.id] = newId;

        const newNode = {
            ...JSON.parse(JSON.stringify(nodeToCopy)),
            id: newId,
            x: (contextMenuPosition.x - canvasTransform.offsetX) / canvasTransform.scale + (nodeToCopy.x - copiedNodes[0].x),
            y: (contextMenuPosition.y - canvasTransform.offsetY) / canvasTransform.scale + (nodeToCopy.y - copiedNodes[0].y)
        };

        nodes.push(newNode);
    });

    renderWorkflow();
    showToast(`Pasted ${copiedNodes.length} node(s)`, 'success');
}

function selectAllNodes() {
    hideAllContextMenus();
    selectedNodes.clear();
    nodes.forEach(node => selectedNodes.add(node.id));
    renderWorkflow();
    showToast(`Selected ${nodes.length} node(s)`, 'info');
}

function contextMenuZoomIn() {
    hideAllContextMenus();
    zoomIn();
}

function contextMenuZoomOut() {
    hideAllContextMenus();
    zoomOut();
}

function fitToView() {
    hideAllContextMenus();
    if (nodes.length === 0) {
        resetZoom();
        return;
    }

    // Calculate bounding box of all nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    nodes.forEach(node => {
        const width = node.type === 'group' ? (node.config?.width || 300) : 220;
        const height = node.type === 'group' ? (node.config?.height || 200) : 120;

        minX = Math.min(minX, node.x);
        minY = Math.min(minY, node.y);
        maxX = Math.max(maxX, node.x + width);
        maxY = Math.max(maxY, node.y + height);
    });

    const canvas = document.getElementById('workflowCanvas');
    const canvasWidth = canvas.clientWidth;
    const canvasHeight = canvas.clientHeight;

    const contentWidth = maxX - minX + 100; // padding
    const contentHeight = maxY - minY + 100;

    const scaleX = canvasWidth / contentWidth;
    const scaleY = canvasHeight / contentHeight;
    const newScale = Math.min(scaleX, scaleY, 1.5); // Cap at 1.5x

    canvasTransform.scale = Math.max(0.25, newScale);
    canvasTransform.offsetX = (canvasWidth - contentWidth * canvasTransform.scale) / 2 - minX * canvasTransform.scale + 50;
    canvasTransform.offsetY = (canvasHeight - contentHeight * canvasTransform.scale) / 2 - minY * canvasTransform.scale + 50;

    renderWorkflow();
    showToast('Fit to view', 'info');
}

function clearCanvas() {
    hideAllContextMenus();
    if (nodes.length === 0 && connections.length === 0) {
        showToast('Canvas is already empty', 'info');
        return;
    }

    if (confirm('Are you sure you want to clear the entire canvas? This action can be undone.')) {
        saveUndoState();
        nodes.length = 0;
        connections.length = 0;
        selectedNodes.clear();
        renderWorkflow();
        showToast('Canvas cleared', 'success');
    }
}

// Context menu actions - Connection
function contextMenuShowDataFlow() {
    hideAllContextMenus();
    if (contextMenuTargetConnection) {
        // Highlight the connection and show data flow info
        showToast(`Data flow: ${contextMenuTargetConnection.from} → ${contextMenuTargetConnection.to}`, 'info');
        // Future: Show data flow inspector panel
    }
}

function contextMenuDeleteConnection() {
    hideAllContextMenus();
    if (contextMenuTargetConnection) {
        saveUndoState();
        connections = connections.filter(c =>
            !(c.from === contextMenuTargetConnection.from && c.to === contextMenuTargetConnection.to)
        );
        renderWorkflow();
        showToast('Connection deleted', 'success');
    }
}

// Context menu actions - Multi-select
function contextMenuCreateGroup() {
    hideAllContextMenus();
    if (selectedNodes.size > 0) {
        createGroupFromSelection();
    }
}

function alignSelectedNodes(alignment) {
    hideAllContextMenus();
    if (selectedNodes.size < 2) {
        showToast('Select at least 2 nodes to align', 'warning');
        return;
    }

    saveUndoState();

    const selectedNodesList = Array.from(selectedNodes).map(id => nodes.find(n => n.id === id)).filter(n => n);

    // Calculate reference point based on alignment type
    let refValue;
    switch (alignment) {
        case 'left':
            refValue = Math.min(...selectedNodesList.map(n => n.x));
            selectedNodesList.forEach(n => n.x = refValue);
            break;
        case 'right':
            refValue = Math.max(...selectedNodesList.map(n => {
                const width = n.type === 'group' ? (n.config?.width || 300) : 220;
                return n.x + width;
            }));
            selectedNodesList.forEach(n => {
                const width = n.type === 'group' ? (n.config?.width || 300) : 220;
                n.x = refValue - width;
            });
            break;
        case 'top':
            refValue = Math.min(...selectedNodesList.map(n => n.y));
            selectedNodesList.forEach(n => n.y = refValue);
            break;
        case 'bottom':
            refValue = Math.max(...selectedNodesList.map(n => {
                const height = n.type === 'group' ? (n.config?.height || 200) : 120;
                return n.y + height;
            }));
            selectedNodesList.forEach(n => {
                const height = n.type === 'group' ? (n.config?.height || 200) : 120;
                n.y = refValue - height;
            });
            break;
        case 'centerH':
            const minX = Math.min(...selectedNodesList.map(n => n.x));
            const maxX = Math.max(...selectedNodesList.map(n => {
                const width = n.type === 'group' ? (n.config?.width || 300) : 220;
                return n.x + width;
            }));
            const centerX = (minX + maxX) / 2;
            selectedNodesList.forEach(n => {
                const width = n.type === 'group' ? (n.config?.width || 300) : 220;
                n.x = centerX - width / 2;
            });
            break;
        case 'centerV':
            const minY = Math.min(...selectedNodesList.map(n => n.y));
            const maxY = Math.max(...selectedNodesList.map(n => {
                const height = n.type === 'group' ? (n.config?.height || 200) : 120;
                return n.y + height;
            }));
            const centerY = (minY + maxY) / 2;
            selectedNodesList.forEach(n => {
                const height = n.type === 'group' ? (n.config?.height || 200) : 120;
                n.y = centerY - height / 2;
            });
            break;
    }

    renderWorkflow();
    showToast(`Aligned ${selectedNodes.size} nodes`, 'success');
}

function distributeSelectedNodes(direction) {
    hideAllContextMenus();
    if (selectedNodes.size < 3) {
        showToast('Select at least 3 nodes to distribute', 'warning');
        return;
    }

    saveUndoState();

    const selectedNodesList = Array.from(selectedNodes).map(id => nodes.find(n => n.id === id)).filter(n => n);

    if (direction === 'horizontal') {
        // Sort by x position
        selectedNodesList.sort((a, b) => a.x - b.x);

        const firstNode = selectedNodesList[0];
        const lastNode = selectedNodesList[selectedNodesList.length - 1];
        const lastWidth = lastNode.type === 'group' ? (lastNode.config?.width || 300) : 220;

        const totalSpan = (lastNode.x + lastWidth) - firstNode.x;
        const totalNodeWidth = selectedNodesList.reduce((sum, n) => {
            return sum + (n.type === 'group' ? (n.config?.width || 300) : 220);
        }, 0);
        const gap = (totalSpan - totalNodeWidth) / (selectedNodesList.length - 1);

        let currentX = firstNode.x;
        selectedNodesList.forEach((node, index) => {
            if (index > 0) {
                node.x = currentX;
            }
            const width = node.type === 'group' ? (node.config?.width || 300) : 220;
            currentX += width + gap;
        });
    } else {
        // Vertical distribution - sort by y position
        selectedNodesList.sort((a, b) => a.y - b.y);

        const firstNode = selectedNodesList[0];
        const lastNode = selectedNodesList[selectedNodesList.length - 1];
        const lastHeight = lastNode.type === 'group' ? (lastNode.config?.height || 200) : 120;

        const totalSpan = (lastNode.y + lastHeight) - firstNode.y;
        const totalNodeHeight = selectedNodesList.reduce((sum, n) => {
            return sum + (n.type === 'group' ? (n.config?.height || 200) : 120);
        }, 0);
        const gap = (totalSpan - totalNodeHeight) / (selectedNodesList.length - 1);

        let currentY = firstNode.y;
        selectedNodesList.forEach((node, index) => {
            if (index > 0) {
                node.y = currentY;
            }
            const height = node.type === 'group' ? (node.config?.height || 200) : 120;
            currentY += height + gap;
        });
    }

    renderWorkflow();
    showToast(`Distributed ${selectedNodes.size} nodes ${direction}ly`, 'success');
}

function deleteSelectedNodes() {
    hideAllContextMenus();
    if (selectedNodes.size === 0) {
        showToast('No nodes selected', 'warning');
        return;
    }

    if (confirm(`Delete ${selectedNodes.size} selected node(s)? This action can be undone.`)) {
        saveUndoState();

        // Remove connections first
        connections = connections.filter(c =>
            !selectedNodes.has(c.from) && !selectedNodes.has(c.to)
        );

        // Remove nodes
        const toDelete = new Set(selectedNodes);
        nodes = nodes.filter(n => !toDelete.has(n.id));

        selectedNodes.clear();
        renderWorkflow();
        showToast(`Deleted ${toDelete.size} node(s)`, 'success');
    }
}

// Duplicate a single node
function duplicateNode(node) {
    saveUndoState();

    const newId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const newNode = {
        ...JSON.parse(JSON.stringify(node)),
        id: newId,
        x: node.x + 30,
        y: node.y + 30
    };

    nodes.push(newNode);
    renderWorkflow();
    showToast('Node duplicated', 'success');
}

// Selection handling
function toggleNodeSelection(nodeId, addToSelection = false) {
    if (addToSelection) {
        if (selectedNodes.has(nodeId)) {
            selectedNodes.delete(nodeId);
        } else {
            selectedNodes.add(nodeId);
        }
    } else {
        selectedNodes.clear();
        selectedNodes.add(nodeId);
    }
    renderWorkflow();
}

function clearSelection() {
    selectedNodes.clear();
    renderWorkflow();
}

// =============================================================================
// Wave 5.2.2: Auto-Save Functionality
// =============================================================================

function initAutoSave() {
    if (autoSaveInterval) {
        clearInterval(autoSaveInterval);
    }

    autoSaveInterval = setInterval(() => {
        if (autoSaveEnabled && nodes.length > 0) {
            autoSaveWorkflow();
        }
    }, AUTO_SAVE_INTERVAL);

    // Save on page unload
    window.addEventListener('beforeunload', (e) => {
        if (nodes.length > 0) {
            autoSaveWorkflow();
        }
    });
}

function autoSaveWorkflow() {
    try {
        const draft = {
            nodes: nodes,
            connections: connections,
            timestamp: Date.now(),
            version: '1.0'
        };

        localStorage.setItem('workflow_draft', JSON.stringify(draft));
        lastSaveTime = Date.now();

        updateAutoSaveIndicator('saved');

        // Brief "Saved" indication
        setTimeout(() => {
            updateAutoSaveIndicator('idle');
        }, 2000);

    } catch (error) {
        console.error('Auto-save failed:', error);
        updateAutoSaveIndicator('error');
    }
}

function updateAutoSaveIndicator(status) {
    const indicator = document.getElementById('autoSaveIndicator');
    if (!indicator) return;

    const icon = indicator.querySelector('.auto-save-icon');
    const text = indicator.querySelector('.auto-save-text');

    switch (status) {
        case 'saving':
            if (icon) icon.textContent = '⏳';
            if (text) text.textContent = 'Saving...';
            indicator.className = 'auto-save-indicator saving';
            break;
        case 'saved':
            if (icon) icon.textContent = '✓';
            if (text) text.textContent = 'Saved';
            indicator.className = 'auto-save-indicator saved';
            break;
        case 'error':
            if (icon) icon.textContent = '⚠';
            if (text) text.textContent = 'Save failed';
            indicator.className = 'auto-save-indicator error';
            break;
        default: // idle
            if (icon) icon.textContent = '💾';
            const timeAgo = Math.round((Date.now() - lastSaveTime) / 1000);
            if (text) text.textContent = timeAgo < 60 ? 'Just saved' : `Saved ${Math.round(timeAgo / 60)}m ago`;
            indicator.className = 'auto-save-indicator';
    }
}

function loadAutoSavedDraft() {
    try {
        const draft = localStorage.getItem('workflow_draft');
        if (draft) {
            const parsed = JSON.parse(draft);
            const timeSince = Date.now() - parsed.timestamp;
            const minutes = Math.round(timeSince / 60000);

            if (minutes < 60 * 24) { // Less than 24 hours old
                if (confirm(`Found auto-saved draft from ${minutes < 60 ? minutes + ' minutes' : Math.round(minutes / 60) + ' hours'} ago. Load it?`)) {
                    nodes.length = 0;
                    connections.length = 0;
                    nodes.push(...parsed.nodes);
                    connections.push(...parsed.connections);
                    renderWorkflow();
                    showToast('Draft restored', 'success');
                    return true;
                }
            }
        }
    } catch (error) {
        console.error('Failed to load auto-saved draft:', error);
    }
    return false;
}

function toggleAutoSave() {
    autoSaveEnabled = !autoSaveEnabled;
    showToast(autoSaveEnabled ? 'Auto-save enabled' : 'Auto-save disabled', 'info');
}

// =============================================================================
// Wave 5.2.3: Additional Keyboard Shortcuts
// =============================================================================

function setupContextMenuListeners() {
    const canvas = document.getElementById('workflowCanvas');
    if (canvas) {
        canvas.addEventListener('contextmenu', handleCanvasContextMenu);
    }

    // Close context menu on click elsewhere
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.context-menu')) {
            hideAllContextMenus();
        }
    });

    // Additional keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Delete selected nodes
        if (e.key === 'Delete' && selectedNodes.size > 0 && !e.target.matches('input, textarea')) {
            e.preventDefault();
            deleteSelectedNodes();
        }

        // Ctrl+A: Select all
        if ((e.ctrlKey || e.metaKey) && e.key === 'a' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            selectAllNodes();
        }

        // Ctrl+C: Copy selected
        if ((e.ctrlKey || e.metaKey) && e.key === 'c' && !e.target.matches('input, textarea')) {
            if (selectedNodes.size > 0) {
                e.preventDefault();
                contextMenuCopy();
            }
        }

        // Ctrl+V: Paste
        if ((e.ctrlKey || e.metaKey) && e.key === 'v' && !e.target.matches('input, textarea')) {
            if (copiedNodes.length > 0) {
                e.preventDefault();
                // Paste at center of canvas
                const canvas = document.getElementById('workflowCanvas');
                contextMenuPosition = {
                    x: canvas.clientWidth / 2,
                    y: canvas.clientHeight / 2
                };
                contextMenuPaste();
            }
        }

        // Ctrl+D: Duplicate selected
        if ((e.ctrlKey || e.metaKey) && e.key === 'd' && !e.target.matches('input, textarea')) {
            if (selectedNodes.size > 0) {
                e.preventDefault();
                const node = nodes.find(n => selectedNodes.has(n.id));
                if (node) duplicateNode(node);
            }
        }

        // Escape: Clear selection
        if (e.key === 'Escape' && selectedNodes.size > 0) {
            clearSelection();
        }
    });
}

// Initialize Wave 5.2 features
function initWave52Features() {
    setupContextMenuListeners();
    initAutoSave();

    // Try to load auto-saved draft on startup
    if (nodes.length === 0) {
        loadAutoSavedDraft();
    }
}

// Call initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave52Features);
} else {
    initWave52Features();
}


// ==========================================
// WAVE 5.3: Dark Mode & Theme System
// ==========================================

// Theme state
let currentTheme = 'light';

/**
 * Get the preferred theme based on system preference or saved preference
 * @returns {string} 'light' or 'dark'
 */
function getPreferredTheme() {
    // Check localStorage first
    const savedTheme = localStorage.getItem('workflow_builder_theme');
    if (savedTheme && ['light', 'dark'].includes(savedTheme)) {
        return savedTheme;
    }

    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    }

    return 'light';
}

/**
 * Apply the specified theme to the document
 * @param {string} theme - 'light' or 'dark'
 */
function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);

    // Update theme toggle button icons
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        const sunIcon = themeToggle.querySelector('.sun-icon');
        const moonIcon = themeToggle.querySelector('.moon-icon');

        if (theme === 'dark') {
            if (sunIcon) sunIcon.style.display = 'none';
            if (moonIcon) moonIcon.style.display = 'inline';
        } else {
            if (sunIcon) sunIcon.style.display = 'inline';
            if (moonIcon) moonIcon.style.display = 'none';
        }
    }

    // Update SVG elements that might need theming
    updateSVGTheme(theme);

    // Update minimap background
    const minimapCanvas = document.getElementById('minimapCanvas');
    if (minimapCanvas) {
        minimapCanvas.style.background = theme === 'dark' ? '#1f1f3d' : '#ffffff';
    }

    // Announce theme change for screen readers
    announceThemeChange(theme);

    // Save preference
    localStorage.setItem('workflow_builder_theme', theme);

    console.log(`[Wave 5.3] Theme changed to: ${theme}`);
}

/**
 * Toggle between light and dark themes
 */
function toggleTheme() {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(newTheme);
}

/**
 * Update SVG elements for theme compatibility
 * @param {string} theme - 'light' or 'dark'
 */
function updateSVGTheme(theme) {
    // Update connection arrows
    const arrows = document.querySelectorAll('.connection-arrow');
    arrows.forEach(arrow => {
        arrow.style.fill = theme === 'dark' ? '#8b92a5' : '#667eea';
    });

    // Update connection paths
    const connections = document.querySelectorAll('.connection-path');
    connections.forEach(conn => {
        if (!conn.classList.contains('connection-active') &&
            !conn.classList.contains('connection-success') &&
            !conn.classList.contains('connection-error')) {
            conn.style.stroke = theme === 'dark' ? '#4a4a6a' : '#667eea';
        }
    });
}

/**
 * Announce theme change for screen readers
 * @param {string} theme - 'light' or 'dark'
 */
function announceThemeChange(theme) {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = `Theme changed to ${theme} mode`;

    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
}

/**
 * Initialize theme system
 */
function initTheme() {
    // Apply saved or system preference theme
    const preferredTheme = getPreferredTheme();
    applyTheme(preferredTheme);

    // Listen for system preference changes
    if (window.matchMedia) {
        const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');

        // Only auto-switch if no saved preference
        darkModeQuery.addEventListener('change', (e) => {
            const savedTheme = localStorage.getItem('workflow_builder_theme');
            if (!savedTheme) {
                applyTheme(e.matches ? 'dark' : 'light');
            }
        });
    }

    // Add keyboard shortcut (T for theme)
    document.addEventListener('keydown', (e) => {
        // T key toggles theme (when not in input)
        if (e.key === 't' || e.key === 'T') {
            if (!e.target.matches('input, textarea, [contenteditable]')) {
                e.preventDefault();
                toggleTheme();
            }
        }
    });

    console.log('[Wave 5.3] Theme system initialized');
}

/**
 * Set a specific theme
 * @param {string} theme - 'light' or 'dark'
 */
function setTheme(theme) {
    if (['light', 'dark'].includes(theme)) {
        applyTheme(theme);
    } else {
        console.warn(`[Wave 5.3] Invalid theme: ${theme}. Use 'light' or 'dark'.`);
    }
}

/**
 * Get current theme
 * @returns {string} Current theme ('light' or 'dark')
 */
function getCurrentTheme() {
    return currentTheme;
}

/**
 * Reset theme to system preference
 */
function resetThemeToSystem() {
    localStorage.removeItem('workflow_builder_theme');

    let systemTheme = 'light';
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        systemTheme = 'dark';
    }

    applyTheme(systemTheme);
    console.log('[Wave 5.3] Theme reset to system preference');
}

// Initialize Wave 5.3 features
function initWave53Features() {
    initTheme();
}

// Call initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave53Features);
} else {
    initWave53Features();
}

// ========== WAVE 5.5: DEBUG SYSTEM ==========

/**
 * Debug mode state
 */
let debugModeActive = false;
let debugState = 'idle'; // idle, running, paused, stopped
let debugCurrentNodeIndex = -1;
let debugExecutionQueue = [];
let debugNodeOutputs = {}; // Stores node input/output for inspection
let debugPausePromiseResolve = null;

/**
 * Toggle debug mode on/off
 */
function toggleDebugMode() {
    debugModeActive = !debugModeActive;
    const toolbar = document.getElementById('debugToolbar');

    if (debugModeActive) {
        toolbar.classList.add('active');
        showToast('Debug mode enabled', 'info');
        console.log('[Wave 5.5] Debug mode enabled');
    } else {
        toolbar.classList.remove('active');
        hideVariableInspector();
        debugStop();
        showToast('Debug mode disabled', 'info');
        console.log('[Wave 5.5] Debug mode disabled');
    }
}

/**
 * Update debug UI state
 */
function updateDebugUI() {
    const statusEl = document.getElementById('debugStatus');
    const runBtn = document.getElementById('debugRunBtn');
    const pauseBtn = document.getElementById('debugPauseBtn');
    const stepOverBtn = document.getElementById('debugStepOverBtn');
    const stepIntoBtn = document.getElementById('debugStepIntoBtn');
    const stopBtn = document.getElementById('debugStopBtn');

    // Update status display
    statusEl.textContent = debugState.charAt(0).toUpperCase() + debugState.slice(1);
    statusEl.className = 'debug-status ' + debugState;

    // Update button states
    switch (debugState) {
        case 'idle':
            runBtn.disabled = false;
            pauseBtn.disabled = true;
            stepOverBtn.disabled = false;
            stepIntoBtn.disabled = false;
            stopBtn.disabled = true;
            break;
        case 'running':
            runBtn.disabled = true;
            pauseBtn.disabled = false;
            stepOverBtn.disabled = true;
            stepIntoBtn.disabled = true;
            stopBtn.disabled = false;
            break;
        case 'paused':
            runBtn.disabled = false;
            pauseBtn.disabled = true;
            stepOverBtn.disabled = false;
            stepIntoBtn.disabled = false;
            stopBtn.disabled = false;
            break;
        case 'stopped':
            runBtn.disabled = false;
            pauseBtn.disabled = true;
            stepOverBtn.disabled = false;
            stepIntoBtn.disabled = false;
            stopBtn.disabled = true;
            break;
    }
}

/**
 * Set execution state on a node (for visualization)
 */
function setNodeExecutionState(nodeId, state, error = null) {
    const node = nodes.find(n => n.id === nodeId);
    if (node) {
        node.executionState = state;
        node.executionError = error;
        renderWorkflow();
    }
}

/**
 * Clear all execution states
 */
function clearExecutionStates() {
    nodes.forEach(node => {
        delete node.executionState;
        delete node.executionError;
    });
    debugNodeOutputs = {};
    renderWorkflow();
}

/**
 * Get execution order for workflow
 */
function getDebugExecutionOrder() {
    // Find root nodes (no incoming connections)
    const hasIncoming = new Set(connections.map(c => c.to));
    const rootNodes = nodes.filter(n => !hasIncoming.has(n.id));

    // BFS to get execution order
    const visited = new Set();
    const queue = [...rootNodes];
    const order = [];

    while (queue.length > 0) {
        const node = queue.shift();
        if (visited.has(node.id)) continue;

        visited.add(node.id);
        order.push(node);

        // Add children
        const outgoing = connections.filter(c => c.from === node.id);
        outgoing.forEach(conn => {
            const child = nodes.find(n => n.id === conn.to);
            if (child && !visited.has(child.id)) {
                queue.push(child);
            }
        });
    }

    return order;
}

/**
 * Debug Run (F5) - Run workflow with breakpoints
 */
async function debugRun() {
    if (!debugModeActive) {
        toggleDebugMode();
    }

    if (debugState === 'paused') {
        // Resume from pause
        debugState = 'running';
        updateDebugUI();
        if (debugPausePromiseResolve) {
            debugPausePromiseResolve();
            debugPausePromiseResolve = null;
        }
        return;
    }

    // Start fresh execution
    clearExecutionStates();
    debugExecutionQueue = getDebugExecutionOrder();
    debugCurrentNodeIndex = 0;
    debugState = 'running';
    updateDebugUI();

    console.log('[Wave 5.5] Debug run started, nodes:', debugExecutionQueue.length);
    showToast('Debug execution started', 'info');

    await executeDebugQueue();
}

/**
 * Execute the debug queue
 */
async function executeDebugQueue() {
    while (debugCurrentNodeIndex < debugExecutionQueue.length && debugState !== 'stopped') {
        const node = debugExecutionQueue[debugCurrentNodeIndex];

        // Check for breakpoint
        if (node.breakpoint && debugState === 'running') {
            debugState = 'paused';
            setNodeExecutionState(node.id, 'breakpoint-hit');
            updateDebugUI();
            updateVariableInspector(node);
            showToast(`Breakpoint hit: ${node.definition.name}`, 'warning');
            console.log('[Wave 5.5] Breakpoint hit at:', node.definition.name);

            // Wait for resume or step
            await new Promise(resolve => {
                debugPausePromiseResolve = resolve;
            });

            if (debugState === 'stopped') break;
        }

        // Mark as running
        setNodeExecutionState(node.id, 'running');

        // Simulate execution
        try {
            // Gather inputs from previous nodes
            const inputs = {};
            const incoming = connections.filter(c => c.to === node.id);
            incoming.forEach(conn => {
                const fromNode = nodes.find(n => n.id === conn.from);
                if (fromNode && debugNodeOutputs[fromNode.id]) {
                    inputs[fromNode.definition.name] = debugNodeOutputs[fromNode.id];
                }
            });

            // Simulate node execution (in real implementation, this would call the backend)
            await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 200));

            // Generate mock output
            const output = {
                status: 'success',
                timestamp: new Date().toISOString(),
                result: `Output from ${node.definition.name}`,
                config: node.config
            };

            debugNodeOutputs[node.id] = { input: inputs, output };
            setNodeExecutionState(node.id, 'success');

            // Update inspector if this node is selected
            if (node.selected) {
                updateVariableInspector(node);
            }

        } catch (error) {
            setNodeExecutionState(node.id, 'error', error.message);
            debugNodeOutputs[node.id] = { error: error.message };

            // Pause on error in debug mode
            debugState = 'paused';
            updateDebugUI();
            showToast(`Error at ${node.definition.name}: ${error.message}`, 'error');

            await new Promise(resolve => {
                debugPausePromiseResolve = resolve;
            });

            if (debugState === 'stopped') break;
        }

        debugCurrentNodeIndex++;
    }

    if (debugState !== 'stopped') {
        debugState = 'idle';
        updateDebugUI();
        showToast('Debug execution completed', 'success');
        console.log('[Wave 5.5] Debug execution completed');
    }
}

/**
 * Debug Pause (F6) - Pause execution
 */
function debugPause() {
    if (debugState === 'running') {
        debugState = 'paused';
        updateDebugUI();
        showToast('Execution paused', 'warning');
        console.log('[Wave 5.5] Execution paused');
    }
}

/**
 * Debug Step Over (F10) - Execute current node and pause at next
 */
async function debugStepOver() {
    if (!debugModeActive) {
        toggleDebugMode();
    }

    if (debugState === 'idle') {
        // Start fresh with step
        clearExecutionStates();
        debugExecutionQueue = getDebugExecutionOrder();
        debugCurrentNodeIndex = 0;
    }

    if (debugCurrentNodeIndex >= debugExecutionQueue.length) {
        showToast('No more nodes to execute', 'info');
        return;
    }

    const node = debugExecutionQueue[debugCurrentNodeIndex];
    debugState = 'running';
    updateDebugUI();

    // Execute single node
    setNodeExecutionState(node.id, 'running');

    try {
        await new Promise(resolve => setTimeout(resolve, 300));

        const inputs = {};
        const incoming = connections.filter(c => c.to === node.id);
        incoming.forEach(conn => {
            const fromNode = nodes.find(n => n.id === conn.from);
            if (fromNode && debugNodeOutputs[fromNode.id]) {
                inputs[fromNode.definition.name] = debugNodeOutputs[fromNode.id];
            }
        });

        debugNodeOutputs[node.id] = {
            input: inputs,
            output: {
                status: 'success',
                result: `Output from ${node.definition.name}`,
                config: node.config
            }
        };

        setNodeExecutionState(node.id, 'success');
        updateVariableInspector(node);
        showToast(`Stepped: ${node.definition.name}`, 'info');

    } catch (error) {
        setNodeExecutionState(node.id, 'error', error.message);
        debugNodeOutputs[node.id] = { error: error.message };
    }

    debugCurrentNodeIndex++;
    debugState = debugCurrentNodeIndex < debugExecutionQueue.length ? 'paused' : 'idle';
    updateDebugUI();
}

/**
 * Debug Step Into (F11) - Step into group nodes
 */
async function debugStepInto() {
    // For now, same as step over (would expand groups in full implementation)
    await debugStepOver();
}

/**
 * Debug Stop (Shift+F5) - Stop execution
 */
function debugStop() {
    debugState = 'stopped';

    // Resolve any pending promises
    if (debugPausePromiseResolve) {
        debugPausePromiseResolve();
        debugPausePromiseResolve = null;
    }

    // Mark remaining nodes as stopped
    for (let i = debugCurrentNodeIndex; i < debugExecutionQueue.length; i++) {
        const node = debugExecutionQueue[i];
        if (node.executionState === 'running' || node.executionState === 'breakpoint-hit') {
            delete node.executionState;
        }
    }

    debugCurrentNodeIndex = -1;
    debugExecutionQueue = [];
    updateDebugUI();
    renderWorkflow();

    showToast('Debug execution stopped', 'error');
    console.log('[Wave 5.5] Debug execution stopped');
}

/**
 * Toggle variable inspector visibility
 */
function toggleVariableInspector() {
    const inspector = document.getElementById('variableInspector');
    if (inspector.classList.contains('visible')) {
        hideVariableInspector();
    } else {
        showVariableInspector();
    }
}

/**
 * Show variable inspector
 */
function showVariableInspector() {
    const inspector = document.getElementById('variableInspector');
    inspector.classList.add('visible');

    // Show selected node data if any
    const selectedNode = nodes.find(n => n.selected);
    if (selectedNode) {
        updateVariableInspector(selectedNode);
    }
}

/**
 * Hide variable inspector
 */
function hideVariableInspector() {
    const inspector = document.getElementById('variableInspector');
    inspector.classList.remove('visible');
}

/**
 * Update variable inspector with node data
 */
function updateVariableInspector(node) {
    const body = document.getElementById('inspectorBody');
    const data = debugNodeOutputs[node.id];

    if (!data) {
        body.innerHTML = `
            <div class="inspector-node-name">${node.definition.name}</div>
            <div class="inspector-empty">
                <div style="font-size: 24px; margin-bottom: 8px;">⏳</div>
                Node has not been executed yet
            </div>
        `;
        return;
    }

    let html = `<div class="inspector-node-name">${node.definition.name}</div>`;

    if (data.input && Object.keys(data.input).length > 0) {
        html += `
            <div class="inspector-section">
                <div class="inspector-section-title">
                    📥 Input
                    <button class="inspector-copy-btn" onclick="copyInspectorData('input', '${node.id}')">Copy</button>
                </div>
                <pre class="inspector-json">${JSON.stringify(data.input, null, 2)}</pre>
            </div>
        `;
    }

    if (data.output) {
        html += `
            <div class="inspector-section">
                <div class="inspector-section-title">
                    📤 Output
                    <button class="inspector-copy-btn" onclick="copyInspectorData('output', '${node.id}')">Copy</button>
                </div>
                <pre class="inspector-json">${JSON.stringify(data.output, null, 2)}</pre>
            </div>
        `;
    }

    if (data.error) {
        html += `
            <div class="inspector-section">
                <div class="inspector-section-title">
                    ❌ Error
                </div>
                <pre class="inspector-json" style="color: #e74c3c;">${data.error}</pre>
            </div>
        `;
    }

    body.innerHTML = html;
}

/**
 * Copy inspector data to clipboard
 */
function copyInspectorData(type, nodeId) {
    const data = debugNodeOutputs[nodeId];
    if (!data) return;

    const content = type === 'input' ? data.input : data.output;
    navigator.clipboard.writeText(JSON.stringify(content, null, 2))
        .then(() => showToast('Copied to clipboard', 'success'))
        .catch(() => showToast('Failed to copy', 'error'));
}

/**
 * Debug keyboard shortcuts
 */
document.addEventListener('keydown', (e) => {
    // Only handle if debug mode is active or toggling it
    if (e.key === 'd' || e.key === 'D') {
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
            // Check if not in input field
            if (document.activeElement.tagName !== 'INPUT' &&
                document.activeElement.tagName !== 'TEXTAREA' &&
                document.activeElement.tagName !== 'SELECT') {
                e.preventDefault();
                toggleDebugMode();
            }
        }
    }

    if (!debugModeActive) return;

    if (e.key === 'F5') {
        e.preventDefault();
        debugRun();
    } else if (e.key === 'F6') {
        e.preventDefault();
        debugPause();
    } else if (e.key === 'F10') {
        e.preventDefault();
        debugStepOver();
    } else if (e.key === 'F11') {
        e.preventDefault();
        debugStepInto();
    } else if (e.key === 'F5' && e.shiftKey) {
        e.preventDefault();
        debugStop();
    } else if (e.key === 'i' || e.key === 'I') {
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
            if (document.activeElement.tagName !== 'INPUT' &&
                document.activeElement.tagName !== 'TEXTAREA' &&
                document.activeElement.tagName !== 'SELECT') {
                e.preventDefault();
                toggleVariableInspector();
            }
        }
    } else if (e.key === 'b' || e.key === 'B') {
        // Ctrl+B: Toggle breakpoint on selected node
        if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            const selectedNode = nodes.find(n => n.selected);
            if (selectedNode) {
                selectedNode.breakpoint = !selectedNode.breakpoint;
                renderWorkflow();
                showToast(selectedNode.breakpoint ? 'Breakpoint set' : 'Breakpoint removed', 'info');
            }
        }
    }
});

// Initialize debug UI on load
function initWave55Features() {
    updateDebugUI();
    console.log('[Wave 5.5] Debug system initialized');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave55Features);
} else {
    initWave55Features();
}

// ========================================
// WAVE 5.6: Backend WebSocket Integration
// ========================================

/**
 * WebSocket connection state
 */
let wsConnection = null;
let wsConnectionState = 'disconnected'; // 'disconnected', 'connecting', 'connected'
let wsAutoReconnect = true;
let wsReconnectInterval = 5000;
let wsReconnectTimer = null;
let wsReconnectAttempts = 0;
const wsMaxReconnectAttempts = 10;

/**
 * Execution logs state
 */
let executionLogs = [];
let logsPanelVisible = false;
let logAutoScroll = true;
let logFilterLevel = 'all';
let logFilterNode = 'all';

/**
 * Node execution timing
 */
const nodeStartTimes = new Map();
const nodeElapsedTimers = new Map();

/**
 * WorkflowWebSocket class for managing WebSocket connection
 */
class WorkflowWebSocket {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.handlers = new Map();
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('[Wave 5.6] Already connected');
            return;
        }

        updateConnectionStatus('connecting');
        console.log('[Wave 5.6] Connecting to:', this.url);

        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('[Wave 5.6] WebSocket connected');
                updateConnectionStatus('connected');
                wsReconnectAttempts = 0;
                addLog('success', null, 'Connected to workflow server');

                // Subscribe to workflow events
                this.send({ type: 'subscribe', workflow_id: 'current' });
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('[Wave 5.6] Failed to parse message:', e);
                }
            };

            this.ws.onclose = (event) => {
                console.log('[Wave 5.6] WebSocket closed:', event.code, event.reason);
                updateConnectionStatus('disconnected');
                addLog('warn', null, `Disconnected from server (code: ${event.code})`);
                this.ws = null;

                // Auto-reconnect
                if (wsAutoReconnect && wsReconnectAttempts < wsMaxReconnectAttempts) {
                    scheduleReconnect();
                }
            };

            this.ws.onerror = (error) => {
                console.error('[Wave 5.6] WebSocket error:', error);
                addLog('error', null, 'WebSocket connection error');
            };
        } catch (e) {
            console.error('[Wave 5.6] Failed to create WebSocket:', e);
            updateConnectionStatus('disconnected');
            addLog('error', null, `Failed to connect: ${e.message}`);
        }
    }

    disconnect() {
        if (wsReconnectTimer) {
            clearTimeout(wsReconnectTimer);
            wsReconnectTimer = null;
        }
        wsAutoReconnect = false;

        if (this.ws) {
            this.ws.close(1000, 'User disconnected');
            this.ws = null;
        }
        updateConnectionStatus('disconnected');
        addLog('info', null, 'Disconnected from server');
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('[Wave 5.6] Cannot send - not connected');
        }
    }

    handleMessage(data) {
        console.log('[Wave 5.6] Received:', data.type, data);

        switch (data.type) {
            case 'node_start':
                handleNodeStart(data);
                break;
            case 'node_complete':
                handleNodeComplete(data);
                break;
            case 'node_error':
                handleNodeError(data);
                break;
            case 'node_status':
                handleNodeStatus(data);
                break;
            case 'log':
                addLog(data.level || 'info', data.node_id, data.message);
                break;
            case 'workflow_start':
                handleWorkflowStart(data);
                break;
            case 'workflow_complete':
                handleWorkflowComplete(data);
                break;
            case 'workflow_error':
                handleWorkflowError(data);
                break;
            case 'progress':
                handleProgressUpdate(data);
                break;
            default:
                console.log('[Wave 5.6] Unknown message type:', data.type);
        }
    }
}

/**
 * Schedule reconnect attempt
 */
function scheduleReconnect() {
    if (wsReconnectTimer) return;

    wsReconnectAttempts++;
    const delay = Math.min(wsReconnectInterval * Math.pow(1.5, wsReconnectAttempts - 1), 30000);

    console.log(`[Wave 5.6] Reconnecting in ${delay/1000}s (attempt ${wsReconnectAttempts}/${wsMaxReconnectAttempts})`);
    addLog('info', null, `Reconnecting in ${Math.round(delay/1000)}s...`);

    wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        if (wsConnection) {
            wsConnection.connect();
        }
    }, delay);
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(status) {
    wsConnectionState = status;
    const statusEl = document.getElementById('connectionStatus');
    const textEl = statusEl?.querySelector('.connection-status-text');
    const connectBtn = document.getElementById('disconnectBtn');
    const connectPrimaryBtn = document.querySelector('.connection-modal-actions .primary');

    if (statusEl) {
        statusEl.className = `connection-status ${status}`;
    }
    if (textEl) {
        const labels = {
            'disconnected': 'Disconnected',
            'connecting': 'Connecting...',
            'connected': 'Connected'
        };
        textEl.textContent = labels[status] || status;
    }
    if (connectBtn && connectPrimaryBtn) {
        if (status === 'connected') {
            connectBtn.style.display = 'inline-block';
            connectPrimaryBtn.style.display = 'none';
        } else {
            connectBtn.style.display = 'none';
            connectPrimaryBtn.style.display = 'inline-block';
        }
    }
}

/**
 * Handle node execution start
 */
function handleNodeStart(data) {
    const nodeId = data.node_id;
    const node = nodes.find(n => n.id === nodeId);

    if (node) {
        const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
        if (nodeEl) {
            // Clear previous state
            nodeEl.classList.remove('ws-complete', 'ws-error');
            nodeEl.classList.add('ws-running');

            // Start timing
            nodeStartTimes.set(nodeId, Date.now());
            startElapsedTimer(nodeId, nodeEl);
        }

        addLog('info', nodeId, `Starting execution: ${node.definition?.name || nodeId}`);
    }
}

/**
 * Handle node execution complete
 */
function handleNodeComplete(data) {
    const nodeId = data.node_id;
    const node = nodes.find(n => n.id === nodeId);

    if (node) {
        const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
        if (nodeEl) {
            nodeEl.classList.remove('ws-running');
            nodeEl.classList.add('ws-complete');
            stopElapsedTimer(nodeId);
        }

        const elapsed = nodeStartTimes.get(nodeId) ? Date.now() - nodeStartTimes.get(nodeId) : 0;
        addLog('success', nodeId, `Completed: ${node.definition?.name || nodeId} (${elapsed}ms)`);

        // Store output for inspector
        if (data.output) {
            debugNodeOutputs[nodeId] = data.output;
        }
    }
}

/**
 * Handle node execution error
 */
function handleNodeError(data) {
    const nodeId = data.node_id;
    const node = nodes.find(n => n.id === nodeId);

    if (node) {
        const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
        if (nodeEl) {
            nodeEl.classList.remove('ws-running');
            nodeEl.classList.add('ws-error');
            stopElapsedTimer(nodeId);
        }

        addLog('error', nodeId, `Error: ${data.error || 'Unknown error'}`);
    }
}

/**
 * Handle node status update
 */
function handleNodeStatus(data) {
    const nodeId = data.node_id;
    const status = data.status;

    switch (status) {
        case 'running':
            handleNodeStart(data);
            break;
        case 'success':
        case 'complete':
            handleNodeComplete(data);
            break;
        case 'error':
        case 'failed':
            handleNodeError(data);
            break;
        default:
            console.log('[Wave 5.6] Unknown node status:', status);
    }
}

/**
 * Handle workflow start
 */
function handleWorkflowStart(data) {
    // Clear all node states
    clearWsNodeStates();
    showWorkflowProgress(true);
    updateWorkflowProgress(0);
    addLog('info', null, `Workflow started: ${data.workflow_id || 'current'}`);
}

/**
 * Handle workflow complete
 */
function handleWorkflowComplete(data) {
    showWorkflowProgress(false);
    addLog('success', null, `Workflow completed (${data.duration_ms || 0}ms)`);
    showToast('Workflow execution completed', 'success');
}

/**
 * Handle workflow error
 */
function handleWorkflowError(data) {
    showWorkflowProgress(false);
    addLog('error', null, `Workflow failed: ${data.error || 'Unknown error'}`);
    showToast('Workflow execution failed', 'error');
}

/**
 * Handle progress update
 */
function handleProgressUpdate(data) {
    const progress = data.progress || 0;
    updateWorkflowProgress(progress);
}

/**
 * Start elapsed time display for node
 */
function startElapsedTimer(nodeId, nodeEl) {
    stopElapsedTimer(nodeId);

    // Create elapsed element if needed
    let elapsedEl = nodeEl.querySelector('.node-elapsed');
    if (!elapsedEl) {
        elapsedEl = document.createElement('div');
        elapsedEl.className = 'node-elapsed';
        nodeEl.appendChild(elapsedEl);
    }

    const startTime = Date.now();
    const updateElapsed = () => {
        const elapsed = Date.now() - startTime;
        elapsedEl.textContent = formatElapsed(elapsed);
    };

    updateElapsed();
    const timer = setInterval(updateElapsed, 100);
    nodeElapsedTimers.set(nodeId, { timer, el: elapsedEl });
}

/**
 * Stop elapsed timer for node
 */
function stopElapsedTimer(nodeId) {
    const timerData = nodeElapsedTimers.get(nodeId);
    if (timerData) {
        clearInterval(timerData.timer);
        // Keep the final time displayed
        nodeElapsedTimers.delete(nodeId);
    }
}

/**
 * Format elapsed time
 */
function formatElapsed(ms) {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    const mins = Math.floor(ms / 60000);
    const secs = Math.floor((ms % 60000) / 1000);
    return `${mins}m ${secs}s`;
}

/**
 * Clear all WebSocket-set node states
 */
function clearWsNodeStates() {
    document.querySelectorAll('.node.ws-running, .node.ws-complete, .node.ws-error').forEach(el => {
        el.classList.remove('ws-running', 'ws-complete', 'ws-error');
    });

    // Clear all timers
    nodeElapsedTimers.forEach((data, nodeId) => {
        clearInterval(data.timer);
        if (data.el && data.el.parentNode) {
            data.el.parentNode.removeChild(data.el);
        }
    });
    nodeElapsedTimers.clear();
    nodeStartTimes.clear();
}

/**
 * Show/hide workflow progress bar
 */
function showWorkflowProgress(show) {
    const progressEl = document.getElementById('workflowProgress');
    if (progressEl) {
        progressEl.classList.toggle('visible', show);
        if (!show) {
            updateWorkflowProgress(0);
        }
    }
}

/**
 * Update workflow progress bar
 */
function updateWorkflowProgress(percent) {
    const barEl = document.getElementById('workflowProgressBar');
    if (barEl) {
        barEl.style.width = `${Math.min(100, Math.max(0, percent))}%`;
    }
}

// ===== Logs Panel Functions =====

/**
 * Add a log entry
 */
function addLog(level, nodeId, message) {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
    const entry = {
        timestamp,
        level: level || 'info',
        nodeId,
        nodeName: nodeId ? (nodes.find(n => n.id === nodeId)?.definition?.name || nodeId) : null,
        message
    };

    executionLogs.push(entry);

    // Update node filter dropdown
    updateLogNodeFilter();

    // Render if panel is visible
    if (logsPanelVisible) {
        renderLogs();
    }

    // Update count
    updateLogCount();
}

/**
 * Render logs to panel
 */
function renderLogs() {
    const container = document.getElementById('logsContent');
    if (!container) return;

    // Filter logs
    const filtered = executionLogs.filter(log => {
        if (logFilterLevel !== 'all' && log.level !== logFilterLevel) return false;
        if (logFilterNode !== 'all' && log.nodeId !== logFilterNode) return false;
        return true;
    });

    if (filtered.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 20px; color: var(--text-muted, #999);">
                No logs match the current filters.
            </div>
        `;
        return;
    }

    container.innerHTML = filtered.map(log => `
        <div class="log-entry">
            <span class="log-timestamp">${log.timestamp}</span>
            <span class="log-level ${log.level}">${log.level}</span>
            ${log.nodeName ? `<span class="log-node" title="${log.nodeId}">${log.nodeName}</span>` : '<span class="log-node">—</span>'}
            <span class="log-message">${escapeHtml(log.message)}</span>
        </div>
    `).join('');

    // Auto-scroll to bottom
    if (logAutoScroll) {
        container.scrollTop = container.scrollHeight;
    }
}

/**
 * Escape HTML for safe display
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Update log count badge
 */
function updateLogCount() {
    const countEl = document.getElementById('logCount');
    if (countEl) {
        countEl.textContent = executionLogs.length;
    }
}

/**
 * Update node filter dropdown
 */
function updateLogNodeFilter() {
    const select = document.getElementById('logNodeFilter');
    if (!select) return;

    const nodeIds = [...new Set(executionLogs.filter(l => l.nodeId).map(l => l.nodeId))];
    const currentValue = select.value;

    select.innerHTML = '<option value="all">All Nodes</option>';
    nodeIds.forEach(nodeId => {
        const node = nodes.find(n => n.id === nodeId);
        const name = node?.definition?.name || nodeId;
        select.innerHTML += `<option value="${nodeId}">${name}</option>`;
    });

    // Restore selection if still valid
    if (nodeIds.includes(currentValue) || currentValue === 'all') {
        select.value = currentValue;
    }
}

/**
 * Toggle logs panel visibility
 */
function toggleLogsPanel() {
    logsPanelVisible = !logsPanelVisible;
    const panel = document.getElementById('logsPanel');
    if (panel) {
        panel.classList.toggle('visible', logsPanelVisible);
        if (logsPanelVisible) {
            renderLogs();
        }
    }
}

/**
 * Show logs panel
 */
function showLogsPanel() {
    logsPanelVisible = true;
    const panel = document.getElementById('logsPanel');
    if (panel) {
        panel.classList.add('visible');
        renderLogs();
    }
}

/**
 * Hide logs panel
 */
function hideLogsPanel() {
    logsPanelVisible = false;
    const panel = document.getElementById('logsPanel');
    if (panel) {
        panel.classList.remove('visible');
    }
}

/**
 * Filter logs by level
 */
function filterLogs() {
    logFilterLevel = document.getElementById('logLevelFilter')?.value || 'all';
    logFilterNode = document.getElementById('logNodeFilter')?.value || 'all';
    renderLogs();
}

/**
 * Toggle auto-scroll for logs
 */
function toggleAutoScroll() {
    logAutoScroll = !logAutoScroll;
    const btn = document.getElementById('logAutoScrollBtn');
    if (btn) {
        btn.classList.toggle('active', logAutoScroll);
    }
    if (logAutoScroll) {
        const container = document.getElementById('logsContent');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }
}

/**
 * Clear all logs
 */
function clearLogs() {
    executionLogs = [];
    updateLogCount();
    renderLogs();
    showToast('Logs cleared', 'info');
}

/**
 * Export logs as JSON
 */
function exportLogs() {
    const data = {
        timestamp: new Date().toISOString(),
        logs: executionLogs
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow-logs-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Logs exported', 'success');
}

// ===== Connection Modal Functions =====

/**
 * Show connection settings modal
 */
function showConnectionModal() {
    const overlay = document.getElementById('connectionModalOverlay');
    const modal = document.getElementById('connectionModal');
    if (overlay) overlay.style.display = 'block';
    if (modal) modal.classList.add('visible');

    // Update checkbox state
    const autoReconnect = document.getElementById('wsAutoReconnect');
    if (autoReconnect) autoReconnect.checked = wsAutoReconnect;

    const interval = document.getElementById('wsReconnectInterval');
    if (interval) interval.value = wsReconnectInterval / 1000;
}

/**
 * Hide connection settings modal
 */
function hideConnectionModal() {
    const overlay = document.getElementById('connectionModalOverlay');
    const modal = document.getElementById('connectionModal');
    if (overlay) overlay.style.display = 'none';
    if (modal) modal.classList.remove('visible');
}

/**
 * Connect to WebSocket server
 */
function connectWebSocket() {
    const urlInput = document.getElementById('wsServerUrl');
    const autoReconnectInput = document.getElementById('wsAutoReconnect');
    const intervalInput = document.getElementById('wsReconnectInterval');

    const url = urlInput?.value || 'ws://localhost:8001/ws';
    wsAutoReconnect = autoReconnectInput?.checked ?? true;
    wsReconnectInterval = (parseInt(intervalInput?.value) || 5) * 1000;
    wsReconnectAttempts = 0;

    if (wsConnection) {
        wsConnection.disconnect();
    }

    wsConnection = new WorkflowWebSocket(url);
    wsConnection.connect();
    hideConnectionModal();
}

/**
 * Disconnect from WebSocket server
 */
function disconnectWebSocket() {
    if (wsConnection) {
        wsConnection.disconnect();
    }
    hideConnectionModal();
}

/**
 * Get WebSocket connection state
 */
function getWsConnectionState() {
    return wsConnectionState;
}

/**
 * Send message via WebSocket
 */
function wsSend(data) {
    if (wsConnection) {
        wsConnection.send(data);
    }
}

/**
 * Execute workflow via WebSocket
 */
function wsExecuteWorkflow() {
    if (wsConnectionState !== 'connected') {
        showToast('Not connected to server. Click connection status to connect.', 'warning');
        return;
    }

    const workflow = exportWorkflow();
    wsConnection.send({
        type: 'execute_workflow',
        workflow: workflow,
        input_data: {}
    });

    showToast('Workflow execution started via WebSocket', 'info');
}

/**
 * Initialize Wave 5.6 features
 */
function initWave56Features() {
    console.log('[Wave 5.6] Initializing Backend WebSocket Integration');

    // Add keyboard shortcut for logs panel
    document.addEventListener('keydown', (e) => {
        if (e.key === 'l' && !e.ctrlKey && !e.metaKey && !e.altKey) {
            const active = document.activeElement;
            if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
                return;
            }
            e.preventDefault();
            toggleLogsPanel();
        }
    });

    // Initialize auto-scroll button state
    const autoScrollBtn = document.getElementById('logAutoScrollBtn');
    if (autoScrollBtn) {
        autoScrollBtn.classList.toggle('active', logAutoScroll);
    }

    // Try to auto-connect if enabled
    const savedUrl = localStorage.getItem('wsServerUrl');
    const autoConnect = localStorage.getItem('wsAutoConnect') === 'true';

    if (autoConnect && savedUrl) {
        const urlInput = document.getElementById('wsServerUrl');
        if (urlInput) urlInput.value = savedUrl;
        setTimeout(() => connectWebSocket(), 1000);
    }

    console.log('[Wave 5.6] Backend WebSocket Integration initialized');
}

// Initialize on DOM ready (add to existing init if available)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave56Features);
} else {
    initWave56Features();
}

// ========================================
// WAVE 7.1: Real-Time Collaboration
// ========================================

/**
 * Collaboration state
 */
let collaborationSessionId = null;
let collaborationParticipants = [];
let remoteCursors = {};  // { odlisId: { userId, x, y, color, name, element } }
let nodeLocks = {};      // { nodeId: { lockedBy, lockedByName, lockedAt } }
let localUserId = null;
let localUserName = 'Anonymous';
let localUserColor = '#667eea';

// Generate random user color
function generateUserColor() {
    const colors = [
        '#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a',
        '#fee140', '#30cfd0', '#a8edea', '#fed6e3', '#ffecd2'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// Generate random user ID
function generateUserId() {
    return 'user_' + Math.random().toString(36).substr(2, 9);
}

// Initialize local user
localUserId = localStorage.getItem('collaborationUserId') || generateUserId();
localUserName = localStorage.getItem('collaborationUserName') || 'Anonymous';
localUserColor = localStorage.getItem('collaborationUserColor') || generateUserColor();
localStorage.setItem('collaborationUserId', localUserId);
localStorage.setItem('collaborationUserColor', localUserColor);

/**
 * Remote Cursors - Show other users' cursor positions
 */
let lastCursorSend = 0;
const CURSOR_THROTTLE_MS = 50;

function updateRemoteCursor(userId, x, y, name, color) {
    if (userId === localUserId) return; // Don't show own cursor

    let cursorEl = document.getElementById(`cursor-${userId}`);
    if (!cursorEl) {
        cursorEl = document.createElement('div');
        cursorEl.id = `cursor-${userId}`;
        cursorEl.className = 'remote-cursor';
        cursorEl.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="${color}">
                <path d="M5.5 3.21V20.8l5.5-5.5h9.5L5.5 3.21z"/>
            </svg>
            <span class="cursor-label" style="background:${color}">${name}</span>
        `;
        document.getElementById('canvas').appendChild(cursorEl);
    }

    // Transform cursor position accounting for pan/zoom
    const canvasX = (x - panOffsetX) / zoomLevel;
    const canvasY = (y - panOffsetY) / zoomLevel;
    cursorEl.style.transform = `translate(${x}px, ${y}px)`;
    cursorEl.style.opacity = '1';

    // Store cursor data
    remoteCursors[userId] = { userId, x, y, name, color, element: cursorEl };

    // Fade out after inactivity
    clearTimeout(remoteCursors[userId].fadeTimer);
    remoteCursors[userId].fadeTimer = setTimeout(() => {
        if (cursorEl) cursorEl.style.opacity = '0.3';
    }, 3000);
}

function removeRemoteCursor(userId) {
    const cursorEl = document.getElementById(`cursor-${userId}`);
    if (cursorEl) cursorEl.remove();
    delete remoteCursors[userId];
}

function sendCursorPosition(e) {
    const now = Date.now();
    if (now - lastCursorSend < CURSOR_THROTTLE_MS || !collaborationSessionId) return;
    lastCursorSend = now;

    if (wsConnection && wsConnection.ws && wsConnection.ws.readyState === WebSocket.OPEN) {
        wsConnection.send({
            type: 'cursor_move',
            session_id: collaborationSessionId,
            user_id: localUserId,
            x: e.clientX,
            y: e.clientY,
            name: localUserName,
            color: localUserColor
        });
    }
}

// Track cursor movement on canvas
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.addEventListener('mousemove', sendCursorPosition);
    }
});

/**
 * Toggle Presence Panel visibility
 */
function togglePresencePanel() {
    const panel = document.getElementById('presencePanel');
    if (!panel) return;

    if (panel.style.display === 'none' || !panel.style.display) {
        panel.style.display = 'block';
        renderPresencePanel();
    } else {
        panel.style.display = 'none';
    }
}

/**
 * Presence Panel - Show who's in the session
 */
function renderPresencePanel() {
    let panel = document.getElementById('presencePanel');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'presencePanel';
        panel.className = 'presence-panel';
        document.body.appendChild(panel);
    }

    if (!collaborationSessionId) {
        panel.innerHTML = `
            <div class="presence-header">
                <span class="presence-title">👥 Collaboration</span>
                <button class="presence-join-btn" onclick="showJoinSessionModal()">Join Session</button>
            </div>
        `;
        panel.classList.remove('active');
        return;
    }

    const participantHTML = collaborationParticipants.map(p => `
        <div class="participant" style="border-color: ${p.color}">
            <div class="avatar" style="background: ${p.color}">${(p.name || 'A')[0].toUpperCase()}</div>
            <span class="name">${p.name || 'Anonymous'}${p.user_id === localUserId ? ' (you)' : ''}</span>
            <span class="status ${p.editing ? 'editing' : ''}">${p.editing || 'idle'}</span>
        </div>
    `).join('');

    panel.innerHTML = `
        <div class="presence-header">
            <span class="presence-title">👥 ${collaborationParticipants.length} Online</span>
            <button class="presence-leave-btn" onclick="leaveCollaborationSession()">Leave</button>
        </div>
        <div class="participants-list">
            ${participantHTML}
        </div>
        <div class="session-info">
            <span class="session-id">Session: ${collaborationSessionId.slice(0, 8)}...</span>
            <button class="copy-session-btn" onclick="copySessionLink()">📋 Copy Link</button>
        </div>
    `;
    panel.classList.add('active');
}

function updatePresence(participants) {
    collaborationParticipants = participants;
    renderPresencePanel();
}

/**
 * Node Lock Indicators - Show which nodes are being edited
 */
function showNodeLock(nodeId, lockedBy, lockedByName, color) {
    const nodeEl = document.getElementById(nodeId);
    if (!nodeEl || lockedBy === localUserId) return;

    nodeEl.classList.add('locked');
    nodeEl.dataset.lockedBy = lockedBy;

    // Remove existing lock badge
    nodeEl.querySelector('.lock-badge')?.remove();

    const lockBadge = document.createElement('div');
    lockBadge.className = 'lock-badge';
    lockBadge.style.borderColor = color || '#667eea';
    lockBadge.innerHTML = `🔒 ${lockedByName || 'Someone'}`;
    nodeEl.appendChild(lockBadge);

    nodeLocks[nodeId] = { lockedBy, lockedByName, lockedAt: Date.now() };
}

function releaseNodeLock(nodeId) {
    const nodeEl = document.getElementById(nodeId);
    if (!nodeEl) return;

    nodeEl.classList.remove('locked');
    delete nodeEl.dataset.lockedBy;
    nodeEl.querySelector('.lock-badge')?.remove();
    delete nodeLocks[nodeId];
}

function requestNodeLock(nodeId) {
    if (!collaborationSessionId || !wsConnection) return true; // No session, always allow

    // Check if already locked by someone else
    if (nodeLocks[nodeId] && nodeLocks[nodeId].lockedBy !== localUserId) {
        showToast(`Node is being edited by ${nodeLocks[nodeId].lockedByName}`, 'warning');
        return false;
    }

    // Request lock from server
    wsConnection.send({
        type: 'node_lock',
        session_id: collaborationSessionId,
        node_id: nodeId,
        user_id: localUserId,
        user_name: localUserName
    });

    return true;
}

function releaseNodeLockRequest(nodeId) {
    if (!collaborationSessionId || !wsConnection) return;

    wsConnection.send({
        type: 'node_unlock',
        session_id: collaborationSessionId,
        node_id: nodeId,
        user_id: localUserId
    });
}

/**
 * Collaboration Session Management
 */
function showJoinSessionModal() {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay collaboration-modal';
    modal.id = 'joinSessionModal';
    modal.innerHTML = `
        <div class="modal-content" style="width: 400px;">
            <h3>Join Collaboration Session</h3>
            <div class="form-group">
                <label for="userName">Your Name</label>
                <input type="text" id="userName" value="${localUserName}" placeholder="Enter your name">
            </div>
            <div class="form-group">
                <label for="sessionId">Session ID (leave empty to create new)</label>
                <input type="text" id="sessionId" placeholder="Enter session ID or leave empty">
            </div>
            <div class="modal-actions">
                <button class="btn-secondary" onclick="closeJoinSessionModal()">Cancel</button>
                <button class="btn-primary" onclick="joinOrCreateSession()">Join/Create</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    setTimeout(() => modal.classList.add('visible'), 10);
}

function closeJoinSessionModal() {
    const modal = document.getElementById('joinSessionModal');
    if (modal) {
        modal.classList.remove('visible');
        setTimeout(() => modal.remove(), 300);
    }
}

async function joinOrCreateSession() {
    const nameInput = document.getElementById('userName');
    const sessionInput = document.getElementById('sessionId');

    localUserName = nameInput?.value || 'Anonymous';
    localStorage.setItem('collaborationUserName', localUserName);

    const sessionId = sessionInput?.value?.trim();

    closeJoinSessionModal();

    try {
        if (sessionId) {
            // Join existing session
            await joinCollaborationSession(sessionId);
        } else {
            // Create new session
            await createCollaborationSession();
        }
    } catch (e) {
        showToast(`Failed to join session: ${e.message}`, 'error');
    }
}

async function createCollaborationSession() {
    const response = await fetch('http://localhost:8001/api/collaboration/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            workflow: exportWorkflow(),
            user_id: localUserId,
            user_name: localUserName
        })
    });

    if (!response.ok) throw new Error('Failed to create session');

    const data = await response.json();
    collaborationSessionId = data.session_id;

    // Connect WebSocket and subscribe to session
    if (!wsConnection || wsConnection.ws?.readyState !== WebSocket.OPEN) {
        await connectWebSocket();
    }

    wsConnection.send({
        type: 'join_session',
        session_id: collaborationSessionId,
        user_id: localUserId,
        user_name: localUserName,
        color: localUserColor
    });

    showToast(`Created session: ${collaborationSessionId.slice(0, 8)}...`, 'success');
    renderPresencePanel();
}

async function joinCollaborationSession(sessionId) {
    const response = await fetch(`http://localhost:8001/api/collaboration/${sessionId}/join`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: localUserId,
            user_name: localUserName
        })
    });

    if (!response.ok) throw new Error('Session not found');

    const data = await response.json();
    collaborationSessionId = sessionId;

    // Load the workflow from session
    if (data.workflow) {
        importWorkflow(JSON.stringify(data.workflow));
    }

    // Connect WebSocket
    if (!wsConnection || wsConnection.ws?.readyState !== WebSocket.OPEN) {
        await connectWebSocket();
    }

    wsConnection.send({
        type: 'join_session',
        session_id: collaborationSessionId,
        user_id: localUserId,
        user_name: localUserName,
        color: localUserColor
    });

    showToast(`Joined session: ${sessionId.slice(0, 8)}...`, 'success');
    renderPresencePanel();
}

async function leaveCollaborationSession() {
    if (!collaborationSessionId) return;

    try {
        await fetch(`http://localhost:8001/api/collaboration/${collaborationSessionId}/leave`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: localUserId })
        });
    } catch (e) {
        console.warn('Failed to leave session:', e);
    }

    // Clean up
    Object.keys(remoteCursors).forEach(removeRemoteCursor);
    Object.keys(nodeLocks).forEach(releaseNodeLock);

    collaborationSessionId = null;
    collaborationParticipants = [];

    showToast('Left collaboration session', 'info');
    renderPresencePanel();
}

function copySessionLink() {
    if (!collaborationSessionId) return;

    const link = `${window.location.origin}${window.location.pathname}?session=${collaborationSessionId}`;
    navigator.clipboard.writeText(link).then(() => {
        showToast('Session link copied!', 'success');
    });
}

/**
 * Handle remote workflow operations (CRDT)
 */
function handleRemoteOperation(op) {
    console.log('[Wave 7.1] Remote operation:', op);

    switch (op.operation_type) {
        case 'add_node':
            // Add node without triggering local broadcast
            addNodeFromRemote(op.data);
            break;
        case 'update_node':
            updateNodeFromRemote(op.node_id, op.data);
            break;
        case 'delete_node':
            deleteNodeFromRemote(op.node_id);
            break;
        case 'add_connection':
            addConnectionFromRemote(op.data);
            break;
        case 'delete_connection':
            deleteConnectionFromRemote(op.connection_id);
            break;
        case 'move_node':
            moveNodeFromRemote(op.node_id, op.x, op.y);
            break;
    }
}

function addNodeFromRemote(nodeData) {
    // Check if node already exists
    if (nodes.find(n => n.id === nodeData.id)) return;

    nodes.push(nodeData);
    renderNode(nodeData);
    saveUndoState();
}

function updateNodeFromRemote(nodeId, data) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    Object.assign(node, data);
    updateNodeVisual(nodeId);
    saveUndoState();
}

function deleteNodeFromRemote(nodeId) {
    const idx = nodes.findIndex(n => n.id === nodeId);
    if (idx === -1) return;

    // Remove connections
    connections = connections.filter(c => c.source !== nodeId && c.target !== nodeId);

    // Remove node
    nodes.splice(idx, 1);
    document.getElementById(nodeId)?.remove();
    renderConnections();
    saveUndoState();
}

function addConnectionFromRemote(connData) {
    // Check if connection already exists
    if (connections.find(c => c.id === connData.id)) return;

    connections.push(connData);
    renderConnections();
    saveUndoState();
}

function deleteConnectionFromRemote(connectionId) {
    const idx = connections.findIndex(c => c.id === connectionId);
    if (idx === -1) return;

    connections.splice(idx, 1);
    renderConnections();
    saveUndoState();
}

function moveNodeFromRemote(nodeId, x, y) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    node.x = x;
    node.y = y;

    const el = document.getElementById(nodeId);
    if (el) {
        el.style.left = x + 'px';
        el.style.top = y + 'px';
    }

    renderConnections();
}

/**
 * Broadcast local operations to collaboration session
 */
function broadcastOperation(operationType, data) {
    if (!collaborationSessionId || !wsConnection) return;

    wsConnection.send({
        type: 'workflow_operation',
        session_id: collaborationSessionId,
        user_id: localUserId,
        operation: {
            operation_type: operationType,
            ...data,
            timestamp: Date.now()
        }
    });
}

/**
 * Extend WebSocket message handler for collaboration
 */
const originalHandleMessage = WorkflowWebSocket.prototype.handleMessage;
WorkflowWebSocket.prototype.handleMessage = function(data) {
    // Handle collaboration messages
    switch (data.type) {
        case 'cursor_move':
            updateRemoteCursor(data.user_id, data.x, data.y, data.name, data.color);
            return;
        case 'presence_update':
            updatePresence(data.participants);
            return;
        case 'node_lock':
            showNodeLock(data.node_id, data.user_id, data.user_name, data.color);
            return;
        case 'node_unlock':
            releaseNodeLock(data.node_id);
            return;
        case 'remote_operation':
            handleRemoteOperation(data.operation);
            return;
        case 'session_joined':
            collaborationSessionId = data.session_id;
            updatePresence(data.participants);
            showToast(`${data.user_name} joined the session`, 'info');
            return;
        case 'session_left':
            removeRemoteCursor(data.user_id);
            updatePresence(data.participants);
            showToast(`${data.user_name} left the session`, 'info');
            return;
    }

    // Call original handler for non-collaboration messages
    originalHandleMessage.call(this, data);
};

/**
 * Initialize Wave 7.1 Features
 */
function initWave71Features() {
    // Check for session ID in URL
    const urlParams = new URLSearchParams(window.location.search);
    const sessionId = urlParams.get('session');

    if (sessionId) {
        // Auto-join session from URL
        setTimeout(() => joinCollaborationSession(sessionId), 1000);
    }

    // Render presence panel
    renderPresencePanel();

    // Keyboard shortcut: C for Collaboration panel
    document.addEventListener('keydown', (e) => {
        if (e.key === 'c' || e.key === 'C') {
            if (!e.ctrlKey && !e.metaKey && !e.altKey) {
                // Check if not in input field
                if (document.activeElement.tagName !== 'INPUT' &&
                    document.activeElement.tagName !== 'TEXTAREA' &&
                    document.activeElement.tagName !== 'SELECT') {
                    e.preventDefault();
                    togglePresencePanel();
                }
            }
        }
    });

    console.log('[Wave 7.1] Collaboration features initialized');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave71Features);
} else {
    initWave71Features();
}

// ========== WAVE 7.2: Advanced Workflows (Nested Workflows, Marketplace) ==========

/**
 * Navigation stack for composite workflow drill-down
 * Each entry stores the parent workflow state
 */
let compositeNavigationStack = [];

/**
 * Saved workflows registry for composite node references
 * { workflow_id: { name, nodes, connections, metadata } }
 */
let savedWorkflowsRegistry = {};

/**
 * Load saved workflows from localStorage
 */
function loadSavedWorkflowsRegistry() {
    try {
        const saved = localStorage.getItem('hololoom_saved_workflows');
        if (saved) {
            savedWorkflowsRegistry = JSON.parse(saved);
        }
    } catch (e) {
        console.error('[Wave 7.2] Error loading saved workflows:', e);
    }
    return savedWorkflowsRegistry;
}

/**
 * Save workflow to registry (for use in composite nodes)
 */
function saveWorkflowToRegistry(workflowId, name, workflowData) {
    savedWorkflowsRegistry[workflowId] = {
        id: workflowId,
        name: name,
        nodes: workflowData.nodes || [],
        connections: workflowData.connections || [],
        metadata: {
            created: new Date().toISOString(),
            updated: new Date().toISOString(),
            nodeCount: (workflowData.nodes || []).length
        }
    };

    try {
        localStorage.setItem('hololoom_saved_workflows', JSON.stringify(savedWorkflowsRegistry));
    } catch (e) {
        console.error('[Wave 7.2] Error saving workflow registry:', e);
    }

    showToast(`Saved workflow: ${name}`, 'success');
    return workflowId;
}

/**
 * Get list of saved workflows for workflow_select config type
 */
function getSavedWorkflowsList() {
    loadSavedWorkflowsRegistry();
    return Object.entries(savedWorkflowsRegistry).map(([id, wf]) => ({
        id: id,
        name: wf.name,
        nodeCount: wf.metadata?.nodeCount || 0,
        updated: wf.metadata?.updated
    }));
}

/**
 * Drill into a composite node (double-click handler)
 * Saves current state to navigation stack and loads nested workflow
 */
function drillIntoComposite(compositeNodeId) {
    const node = nodes.find(n => n.id === compositeNodeId);
    if (!node || !node.definition?.isComposite) {
        console.warn('[Wave 7.2] Not a composite node:', compositeNodeId);
        return;
    }

    const workflowId = node.config?.workflow_id;
    if (!workflowId) {
        showToast('No workflow selected for this composite node', 'warning');
        return;
    }

    const nestedWorkflow = savedWorkflowsRegistry[workflowId];
    if (!nestedWorkflow) {
        showToast(`Workflow not found: ${workflowId}`, 'error');
        return;
    }

    // Save current state to navigation stack
    compositeNavigationStack.push({
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections)),
        compositeNodeId: compositeNodeId,
        compositeNodeLabel: node.config?.label || 'Composite Workflow',
        panOffsetX: panOffsetX,
        panOffsetY: panOffsetY,
        zoomLevel: zoomLevel
    });

    // Load nested workflow
    nodes.length = 0;
    connections.length = 0;

    nestedWorkflow.nodes.forEach(n => {
        nodes.push({
            ...n,
            definition: agentDefinitions[n.agentType]
        });
    });

    nestedWorkflow.connections.forEach(c => connections.push({...c}));

    // Reset view
    panOffsetX = 0;
    panOffsetY = 0;
    applyPan();

    // Re-render
    renderAllNodes();
    renderConnections();
    updateBreadcrumbNavigation();

    showToast(`Entered: ${nestedWorkflow.name}`, 'info');
    console.log('[Wave 7.2] Drilled into composite:', workflowId);
}

/**
 * Drill up from nested workflow back to parent
 */
function drillUp() {
    if (compositeNavigationStack.length === 0) {
        showToast('Already at root level', 'info');
        return;
    }

    const parentState = compositeNavigationStack.pop();

    // Restore parent state
    nodes.length = 0;
    connections.length = 0;

    parentState.nodes.forEach(n => {
        nodes.push({
            ...n,
            definition: agentDefinitions[n.agentType]
        });
    });

    parentState.connections.forEach(c => connections.push({...c}));

    // Restore view
    panOffsetX = parentState.panOffsetX || 0;
    panOffsetY = parentState.panOffsetY || 0;
    zoomLevel = parentState.zoomLevel || 1;
    applyPan();
    applyZoom();

    // Re-render
    renderAllNodes();
    renderConnections();
    updateBreadcrumbNavigation();

    showToast('Returned to parent workflow', 'info');
    console.log('[Wave 7.2] Drilled up to parent');
}

/**
 * Update breadcrumb navigation UI
 */
function updateBreadcrumbNavigation() {
    let breadcrumbContainer = document.getElementById('compositeBreadcrumb');

    if (!breadcrumbContainer) {
        // Create breadcrumb container if it doesn't exist
        breadcrumbContainer = document.createElement('div');
        breadcrumbContainer.id = 'compositeBreadcrumb';
        breadcrumbContainer.className = 'composite-breadcrumb';

        const toolbar = document.querySelector('.workflow-toolbar');
        if (toolbar) {
            toolbar.appendChild(breadcrumbContainer);
        }
    }

    // Build breadcrumb path
    const path = ['Root'];
    compositeNavigationStack.forEach(entry => {
        path.push(entry.compositeNodeLabel);
    });

    // Render breadcrumb items
    breadcrumbContainer.innerHTML = path.map((name, index) => {
        const isLast = index === path.length - 1;
        const clickHandler = isLast ? '' : `onclick="navigateToBreadcrumbLevel(${index})"`;
        const className = isLast ? 'breadcrumb-item active' : 'breadcrumb-item clickable';
        return `<span class="${className}" ${clickHandler}>${name}</span>`;
    }).join('<span class="breadcrumb-separator">›</span>');

    // Show/hide based on depth
    breadcrumbContainer.style.display = compositeNavigationStack.length > 0 ? 'flex' : 'none';
}

/**
 * Navigate to specific breadcrumb level
 */
function navigateToBreadcrumbLevel(targetLevel) {
    // Pop states until we reach target level
    while (compositeNavigationStack.length > targetLevel) {
        drillUp();
    }
}

/**
 * Render a mini preview of a nested workflow inside composite node
 */
function renderMiniWorkflow(workflowId) {
    const workflow = savedWorkflowsRegistry[workflowId];
    if (!workflow) {
        return '<div class="mini-workflow-placeholder">No workflow selected</div>';
    }

    const nodeCount = workflow.nodes?.length || 0;
    const connectionCount = workflow.connections?.length || 0;

    // Create mini node representations
    const miniNodes = (workflow.nodes || []).slice(0, 5).map(n => {
        const def = agentDefinitions[n.agentType];
        const icon = getAgentIcon(def?.type || 'control');
        return `<div class="mini-node" style="background: ${def?.color || '#6c757d'}" title="${def?.name || n.agentType}">${icon}</div>`;
    }).join('');

    const moreNodes = nodeCount > 5 ? `<div class="mini-node more">+${nodeCount - 5}</div>` : '';

    return `
        <div class="mini-workflow-preview">
            <div class="mini-nodes">${miniNodes}${moreNodes}</div>
            <div class="mini-stats">${nodeCount} nodes, ${connectionCount} connections</div>
        </div>
    `;
}

/**
 * Special rendering for composite nodes
 */
function renderCompositeNodeContent(node) {
    const workflowId = node.config?.workflow_id;
    const showPreview = node.config?.show_preview !== false;

    let previewHtml = '';
    if (showPreview && workflowId) {
        previewHtml = renderMiniWorkflow(workflowId);
    } else if (!workflowId) {
        previewHtml = '<div class="mini-workflow-placeholder">Double-click to select workflow</div>';
    }

    return `
        <div class="composite-node-content">
            ${previewHtml}
            <div class="composite-hint">Double-click to enter</div>
        </div>
    `;
}

/**
 * Handle double-click on nodes (for composite drill-down)
 */
function handleNodeDoubleClick(nodeId) {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    if (node.definition?.isComposite) {
        drillIntoComposite(nodeId);
    } else if (node.agentType === 'group' || node.agentType === 'subworkflow') {
        // For group nodes, toggle collapse
        toggleGroupCollapse(nodeId);
    }
}

// ========== MARKETPLACE UI (Wave 7.2.2) ==========

/**
 * Marketplace state
 */
let marketplaceTemplates = [];
let marketplaceFilter = { category: '', search: '', sort: 'downloads' };

/**
 * Load marketplace templates (mock data for now)
 */
function loadMarketplaceTemplates() {
    // In production, this would fetch from backend API
    marketplaceTemplates = [
        {
            id: 'rag-basic',
            name: 'Basic RAG Pipeline',
            description: 'Query → Retrieve → Generate workflow',
            category: 'RAG',
            tags: ['rag', 'retrieval', 'basic'],
            author: { name: 'HoloLoom Team' },
            stats: { downloads: 1523, rating: 4.8, reviews: 45 },
            preview_image: null,
            created_at: '2025-11-01',
            updated_at: '2025-12-01'
        },
        {
            id: 'agentic-research',
            name: 'Research Agent',
            description: 'Multi-query research with verification',
            category: 'Agentic',
            tags: ['agentic', 'research', 'verification'],
            author: { name: 'HoloLoom Team' },
            stats: { downloads: 892, rating: 4.6, reviews: 28 },
            preview_image: null,
            created_at: '2025-11-15',
            updated_at: '2025-12-10'
        },
        {
            id: 'memory-fusion',
            name: 'Memory Fusion Pipeline',
            description: 'Multi-hop knowledge graph traversal',
            category: 'Memory',
            tags: ['memory', 'knowledge-graph', 'fusion'],
            author: { name: 'HoloLoom Team' },
            stats: { downloads: 654, rating: 4.5, reviews: 19 },
            preview_image: null,
            created_at: '2025-10-20',
            updated_at: '2025-12-05'
        },
        {
            id: 'safety-gated',
            name: 'Safety-Gated Workflow',
            description: 'Workflow with risk-based safety guardrails',
            category: 'Control',
            tags: ['safety', 'guardrails', 'risk'],
            author: { name: 'HoloLoom Team' },
            stats: { downloads: 421, rating: 4.9, reviews: 12 },
            preview_image: null,
            created_at: '2025-11-10',
            updated_at: '2025-12-08'
        }
    ];

    return marketplaceTemplates;
}

/**
 * Show marketplace modal
 */
function showMarketplaceModal() {
    loadMarketplaceTemplates();

    let modal = document.getElementById('marketplaceModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'marketplaceModal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content marketplace-modal">
                <div class="modal-header">
                    <h2>📦 Workflow Marketplace</h2>
                    <button class="close-btn" onclick="closeMarketplaceModal()">✕</button>
                </div>
                <div class="marketplace-header">
                    <input type="search" id="marketplaceSearch" placeholder="Search workflows..."
                           oninput="filterMarketplace()">
                    <select id="marketplaceCategoryFilter" onchange="filterMarketplace()">
                        <option value="">All Categories</option>
                        <option value="RAG">RAG Pipelines</option>
                        <option value="Agentic">Agentic Workflows</option>
                        <option value="Memory">Memory Operations</option>
                        <option value="Control">Control Flow</option>
                    </select>
                    <select id="marketplaceSortBy" onchange="filterMarketplace()">
                        <option value="downloads">Most Downloaded</option>
                        <option value="rating">Highest Rated</option>
                        <option value="recent">Most Recent</option>
                    </select>
                </div>
                <div class="marketplace-grid" id="marketplaceGrid">
                    <!-- Populated dynamically -->
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    renderMarketplaceGrid();
    modal.style.display = 'flex';
}

/**
 * Close marketplace modal
 */
function closeMarketplaceModal() {
    const modal = document.getElementById('marketplaceModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Filter marketplace based on search, category, sort
 */
function filterMarketplace() {
    const search = document.getElementById('marketplaceSearch')?.value.toLowerCase() || '';
    const category = document.getElementById('marketplaceCategoryFilter')?.value || '';
    const sortBy = document.getElementById('marketplaceSortBy')?.value || 'downloads';

    marketplaceFilter = { search, category, sort: sortBy };
    renderMarketplaceGrid();
}

/**
 * Render marketplace grid
 */
function renderMarketplaceGrid() {
    const grid = document.getElementById('marketplaceGrid');
    if (!grid) return;

    let filtered = [...marketplaceTemplates];

    // Apply filters
    if (marketplaceFilter.category) {
        filtered = filtered.filter(t => t.category === marketplaceFilter.category);
    }
    if (marketplaceFilter.search) {
        filtered = filtered.filter(t =>
            t.name.toLowerCase().includes(marketplaceFilter.search) ||
            t.description.toLowerCase().includes(marketplaceFilter.search) ||
            t.tags.some(tag => tag.toLowerCase().includes(marketplaceFilter.search))
        );
    }

    // Apply sort
    const sortFns = {
        downloads: (a, b) => b.stats.downloads - a.stats.downloads,
        rating: (a, b) => b.stats.rating - a.stats.rating,
        recent: (a, b) => new Date(b.updated_at) - new Date(a.updated_at)
    };
    filtered.sort(sortFns[marketplaceFilter.sort] || sortFns.downloads);

    // Render cards
    grid.innerHTML = filtered.map(template => `
        <div class="marketplace-card" data-id="${template.id}">
            <div class="marketplace-card-header">
                <span class="marketplace-category">${template.category}</span>
                <span class="marketplace-rating">⭐ ${template.stats.rating}</span>
            </div>
            <h3 class="marketplace-card-title">${template.name}</h3>
            <p class="marketplace-card-desc">${template.description}</p>
            <div class="marketplace-card-tags">
                ${template.tags.map(t => `<span class="marketplace-tag">${t}</span>`).join('')}
            </div>
            <div class="marketplace-card-meta">
                <span class="downloads">⬇️ ${template.stats.downloads}</span>
                <span class="author">by ${template.author.name}</span>
            </div>
            <button class="btn-primary marketplace-install-btn" onclick="installMarketplaceTemplate('${template.id}')">
                Install
            </button>
        </div>
    `).join('') || '<div class="marketplace-empty">No workflows found matching your filters.</div>';
}

/**
 * Install a marketplace template
 */
function installMarketplaceTemplate(templateId) {
    const template = marketplaceTemplates.find(t => t.id === templateId);
    if (!template) {
        showToast('Template not found', 'error');
        return;
    }

    // In production, this would fetch the full template from backend
    // For now, we'll use a mock workflow structure
    const mockWorkflow = getMockWorkflowForTemplate(templateId);

    // Save to registry for use as composite workflows
    const workflowId = `marketplace_${templateId}_${Date.now()}`;
    saveWorkflowToRegistry(workflowId, template.name, mockWorkflow);

    // Optionally load directly into canvas
    if (confirm(`Install "${template.name}"?\n\nThis will add it to your saved workflows and can be used in composite nodes.`)) {
        // Could also offer to load into current canvas
        showToast(`Installed: ${template.name}`, 'success');
        closeMarketplaceModal();
    }
}

/**
 * Get mock workflow data for marketplace templates
 */
function getMockWorkflowForTemplate(templateId) {
    const mockWorkflows = {
        'rag-basic': {
            nodes: [
                { id: 'n1', agentType: 'hololoom', x: 100, y: 200, config: { pattern: 'fast' } },
                { id: 'n2', agentType: 'response', x: 400, y: 200, config: { format: 'text' } }
            ],
            connections: [
                { id: 'c1', source: 'n1', target: 'n2', sourcePort: 'spacetime', targetPort: 'data' }
            ]
        },
        'agentic-research': {
            nodes: [
                { id: 'n1', agentType: 'multiquery', x: 100, y: 200, config: { mode: 'research' } },
                { id: 'n2', agentType: 'hololoom', x: 350, y: 150, config: { pattern: 'fused' } },
                { id: 'n3', agentType: 'refiner', x: 600, y: 200, config: { strategy: 'verify' } },
                { id: 'n4', agentType: 'response', x: 850, y: 200, config: { format: 'markdown' } }
            ],
            connections: [
                { id: 'c1', source: 'n1', target: 'n2', sourcePort: 'subqueries', targetPort: 'query' },
                { id: 'c2', source: 'n2', target: 'n3', sourcePort: 'spacetime', targetPort: 'spacetime' },
                { id: 'c3', source: 'n3', target: 'n4', sourcePort: 'refined', targetPort: 'data' }
            ]
        },
        'memory-fusion': {
            nodes: [
                { id: 'n1', agentType: 'search', x: 100, y: 200, config: { max_results: 20 } },
                { id: 'n2', agentType: 'fusion', x: 350, y: 200, config: { max_depth: 3 } },
                { id: 'n3', agentType: 'synthesizer', x: 600, y: 200, config: {} },
                { id: 'n4', agentType: 'response', x: 850, y: 200, config: { format: 'json' } }
            ],
            connections: [
                { id: 'c1', source: 'n1', target: 'n2', sourcePort: 'memories', targetPort: 'query' },
                { id: 'c2', source: 'n2', target: 'n3', sourcePort: 'expanded', targetPort: 'text' },
                { id: 'c3', source: 'n3', target: 'n4', sourcePort: 'synthesis', targetPort: 'data' }
            ]
        },
        'safety-gated': {
            nodes: [
                { id: 'n1', agentType: 'hololoom', x: 100, y: 200, config: { pattern: 'fast' } },
                { id: 'n2', agentType: 'safety', x: 350, y: 200, config: { risk_threshold: 'MEDIUM' } },
                { id: 'n3', agentType: 'conditional', x: 600, y: 200, config: { condition_type: 'confidence' } },
                { id: 'n4', agentType: 'response', x: 850, y: 100, config: { format: 'text' } },
                { id: 'n5', agentType: 'refiner', x: 850, y: 300, config: { strategy: 'verify' } }
            ],
            connections: [
                { id: 'c1', source: 'n1', target: 'n2', sourcePort: 'spacetime', targetPort: 'action' },
                { id: 'c2', source: 'n2', target: 'n3', sourcePort: 'gated', targetPort: 'condition' },
                { id: 'c3', source: 'n3', target: 'n4', sourcePort: 'true', targetPort: 'data' },
                { id: 'c4', source: 'n3', target: 'n5', sourcePort: 'false', targetPort: 'spacetime' }
            ]
        }
    };

    return mockWorkflows[templateId] || { nodes: [], connections: [] };
}

/**
 * Initialize Wave 7.2 features
 */
function initWave72Features() {
    // Load saved workflows registry
    loadSavedWorkflowsRegistry();

    // Add double-click handler for composite nodes
    const canvas = document.getElementById('canvas');
    if (canvas) {
        canvas.addEventListener('dblclick', (e) => {
            const nodeEl = e.target.closest('.workflow-node');
            if (nodeEl) {
                handleNodeDoubleClick(nodeEl.id);
            }
        });
    }

    // Keyboard shortcut: Backspace/Delete to drill up when in nested workflow
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace' && compositeNavigationStack.length > 0) {
            // Only if not in input field
            if (document.activeElement.tagName !== 'INPUT' &&
                document.activeElement.tagName !== 'TEXTAREA') {
                e.preventDefault();
                drillUp();
            }
        }
    });

    // Keyboard shortcut: M for Marketplace
    document.addEventListener('keydown', (e) => {
        if ((e.key === 'm' || e.key === 'M') && !e.ctrlKey && !e.metaKey && !e.altKey) {
            if (document.activeElement.tagName !== 'INPUT' &&
                document.activeElement.tagName !== 'TEXTAREA' &&
                document.activeElement.tagName !== 'SELECT') {
                e.preventDefault();
                showMarketplaceModal();
            }
        }
    });

    console.log('[Wave 7.2] Advanced Workflows features initialized');
}

// Initialize Wave 7.2 on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave72Features);
} else {
    initWave72Features();
}

// ========== WAVE 7.3: PERFORMANCE OPTIMIZATION ==========
// Virtual scrolling, connection culling, lazy loading, FPS monitoring
// For handling 100+ node workflows at 60fps

/**
 * VirtualCanvas - Only render nodes visible in viewport
 * Implements virtual scrolling for massive workflows
 */
class VirtualCanvas {
    constructor(canvasElement, viewportPadding = 200) {
        this.canvas = canvasElement;
        this.viewportPadding = viewportPadding;
        this.renderedNodes = new Set();
        this.nodeElementCache = new Map();
        this.lastViewport = null;
        this.updateScheduled = false;
    }

    /**
     * Get current viewport bounds in canvas coordinates
     * Accounts for pan offset and zoom level
     */
    getViewport() {
        const padding = this.viewportPadding;
        return {
            left: (-panOffsetX / zoomLevel) - padding,
            top: (-panOffsetY / zoomLevel) - padding,
            right: ((-panOffsetX + window.innerWidth) / zoomLevel) + padding,
            bottom: ((-panOffsetY + window.innerHeight) / zoomLevel) + padding
        };
    }

    /**
     * Check if a node is within the visible viewport
     * Uses node position and estimated size (200x100)
     */
    isNodeVisible(node) {
        const vp = this.getViewport();
        const nodeWidth = 200;
        const nodeHeight = 100;

        return node.x + nodeWidth >= vp.left &&
               node.x <= vp.right &&
               node.y + nodeHeight >= vp.top &&
               node.y <= vp.bottom;
    }

    /**
     * Update which nodes are rendered based on viewport
     * Called on pan, zoom, or node position changes
     */
    updateVisibleNodes() {
        if (!this.canvas) return;

        const viewport = this.getViewport();
        const visibleNodeIds = new Set();

        // Determine which nodes should be visible
        nodes.forEach(node => {
            if (this.isNodeVisible(node)) {
                visibleNodeIds.add(node.id);
            }
        });

        // Remove nodes that are no longer visible
        for (const nodeId of this.renderedNodes) {
            if (!visibleNodeIds.has(nodeId)) {
                this.removeNodeElement(nodeId);
            }
        }

        // Add nodes that have become visible
        for (const nodeId of visibleNodeIds) {
            if (!this.renderedNodes.has(nodeId)) {
                const node = nodes.find(n => n.id === nodeId);
                if (node) {
                    this.renderNodeElement(node);
                }
            }
        }

        this.renderedNodes = visibleNodeIds;
        this.lastViewport = viewport;

        // Update performance monitor
        if (performanceMonitor) {
            performanceMonitor.recordNodeCount(visibleNodeIds.size, nodes.length);
        }
    }

    /**
     * Render a single node element to the canvas
     */
    renderNodeElement(node) {
        // Check if element already exists
        let nodeEl = document.getElementById(node.id);
        if (nodeEl) {
            this.renderedNodes.add(node.id);
            return;
        }

        // Use the existing createNodeElement function
        if (typeof createNodeElement === 'function') {
            nodeEl = createNodeElement(node);
            this.canvas.appendChild(nodeEl);
            this.renderedNodes.add(node.id);
            this.nodeElementCache.set(node.id, nodeEl);
        }
    }

    /**
     * Remove a node element from the canvas (keep node data)
     */
    removeNodeElement(nodeId) {
        const nodeEl = document.getElementById(nodeId);
        if (nodeEl) {
            nodeEl.remove();
        }
        this.renderedNodes.delete(nodeId);
        this.nodeElementCache.delete(nodeId);
    }

    /**
     * Schedule viewport update (debounced)
     */
    scheduleUpdate() {
        if (this.updateScheduled) return;

        this.updateScheduled = true;
        requestAnimationFrame(() => {
            this.updateVisibleNodes();
            this.updateScheduled = false;
        });
    }

    /**
     * Force immediate update (use sparingly)
     */
    forceUpdate() {
        this.updateScheduled = false;
        this.updateVisibleNodes();
    }

    /**
     * Get statistics about current rendering state
     */
    getStats() {
        return {
            renderedNodes: this.renderedNodes.size,
            totalNodes: nodes.length,
            renderPercentage: nodes.length > 0
                ? ((this.renderedNodes.size / nodes.length) * 100).toFixed(1)
                : 0,
            viewport: this.getViewport()
        };
    }

    /**
     * Clear all rendered nodes (for full re-render)
     */
    clear() {
        for (const nodeId of this.renderedNodes) {
            this.removeNodeElement(nodeId);
        }
        this.renderedNodes.clear();
        this.nodeElementCache.clear();
    }
}

/**
 * ConnectionCuller - Only render connections where at least one endpoint is visible
 * Reduces SVG path complexity for large workflows
 */
class ConnectionCuller {
    constructor(svgLayer) {
        this.svgLayer = svgLayer;
        this.renderedConnections = new Set();
        this.connectionPathCache = new Map();
    }

    /**
     * Check if a connection should be rendered
     * Connection is visible if either source or target node is visible
     */
    isConnectionVisible(conn, virtualCanvas) {
        const sourceNode = nodes.find(n => n.id === conn.source);
        const targetNode = nodes.find(n => n.id === conn.target);

        if (!sourceNode || !targetNode) return false;

        // Connection visible if either endpoint is in viewport
        return virtualCanvas.isNodeVisible(sourceNode) ||
               virtualCanvas.isNodeVisible(targetNode);
    }

    /**
     * Update visible connections based on viewport
     */
    updateVisibleConnections(virtualCanvas) {
        if (!this.svgLayer || !virtualCanvas) return;

        const visibleConnectionIds = new Set();

        // Determine which connections should be visible
        connections.forEach(conn => {
            const connId = `conn-${conn.source}-${conn.target}`;
            if (this.isConnectionVisible(conn, virtualCanvas)) {
                visibleConnectionIds.add(connId);
            }
        });

        // Remove connections that are no longer visible
        for (const connId of this.renderedConnections) {
            if (!visibleConnectionIds.has(connId)) {
                this.removeConnectionPath(connId);
            }
        }

        // Add connections that have become visible
        connections.forEach(conn => {
            const connId = `conn-${conn.source}-${conn.target}`;
            if (visibleConnectionIds.has(connId) && !this.renderedConnections.has(connId)) {
                this.renderConnectionPath(conn);
            }
        });

        this.renderedConnections = visibleConnectionIds;
    }

    /**
     * Render a single connection path
     */
    renderConnectionPath(conn) {
        const connId = `conn-${conn.source}-${conn.target}`;

        // Check if already exists
        let pathEl = document.getElementById(connId);
        if (pathEl) {
            this.renderedConnections.add(connId);
            return;
        }

        // Create new connection path using existing render function
        // Connection rendering is handled by renderConnections(),
        // so we just mark it as needing render
        this.renderedConnections.add(connId);
    }

    /**
     * Remove a connection path from SVG layer
     */
    removeConnectionPath(connId) {
        const pathEl = document.getElementById(connId);
        if (pathEl) {
            pathEl.remove();
        }
        this.renderedConnections.delete(connId);
        this.connectionPathCache.delete(connId);
    }

    /**
     * Get statistics about current connection rendering
     */
    getStats() {
        return {
            renderedConnections: this.renderedConnections.size,
            totalConnections: connections.length,
            renderPercentage: connections.length > 0
                ? ((this.renderedConnections.size / connections.length) * 100).toFixed(1)
                : 0
        };
    }

    /**
     * Clear all rendered connections
     */
    clear() {
        for (const connId of this.renderedConnections) {
            this.removeConnectionPath(connId);
        }
        this.renderedConnections.clear();
        this.connectionPathCache.clear();
    }
}

/**
 * PerformanceMonitor - Track frame times and FPS
 * Provides real-time performance metrics for debugging
 */
class PerformanceMonitor {
    constructor() {
        this.frameTimeSamples = [];
        this.maxSamples = 60;
        this.lastFrameTime = performance.now();
        this.nodeStats = { rendered: 0, total: 0 };
        this.connectionStats = { rendered: 0, total: 0 };
        this.warningThreshold = 16.67; // 60fps target
        this.enabled = true;
        this.debugPanelVisible = false;
    }

    /**
     * Record start of a new frame
     */
    startFrame() {
        this.frameStart = performance.now();
    }

    /**
     * Record end of frame and calculate duration
     */
    endFrame() {
        if (!this.enabled) return;

        const now = performance.now();
        const frameTime = now - (this.frameStart || this.lastFrameTime);

        this.frameTimeSamples.push(frameTime);
        if (this.frameTimeSamples.length > this.maxSamples) {
            this.frameTimeSamples.shift();
        }

        // Warn if frame time exceeds 60fps target
        if (frameTime > this.warningThreshold && this.debugPanelVisible) {
            console.warn(`[Perf] Slow frame: ${frameTime.toFixed(1)}ms`);
        }

        this.lastFrameTime = now;
    }

    /**
     * Record node rendering statistics
     */
    recordNodeCount(rendered, total) {
        this.nodeStats = { rendered, total };
    }

    /**
     * Record connection rendering statistics
     */
    recordConnectionCount(rendered, total) {
        this.connectionStats = { rendered, total };
    }

    /**
     * Calculate average FPS from recent samples
     */
    getAverageFPS() {
        if (this.frameTimeSamples.length === 0) return 60;

        const avgFrameTime = this.frameTimeSamples.reduce((a, b) => a + b, 0) /
                            this.frameTimeSamples.length;
        return 1000 / avgFrameTime;
    }

    /**
     * Calculate average frame time in milliseconds
     */
    getAverageFrameTime() {
        if (this.frameTimeSamples.length === 0) return 0;

        return this.frameTimeSamples.reduce((a, b) => a + b, 0) /
               this.frameTimeSamples.length;
    }

    /**
     * Get comprehensive performance statistics
     */
    getStats() {
        const fps = this.getAverageFPS();
        const avgFrameTime = this.getAverageFrameTime();

        return {
            fps: fps.toFixed(1),
            avgFrameTime: avgFrameTime.toFixed(2),
            nodes: this.nodeStats,
            connections: this.connectionStats,
            samplesCollected: this.frameTimeSamples.length,
            status: fps >= 55 ? 'good' : fps >= 30 ? 'warning' : 'poor'
        };
    }

    /**
     * Reset all statistics
     */
    reset() {
        this.frameTimeSamples = [];
        this.nodeStats = { rendered: 0, total: 0 };
        this.connectionStats = { rendered: 0, total: 0 };
    }

    /**
     * Toggle debug panel visibility
     */
    toggleDebugPanel() {
        this.debugPanelVisible = !this.debugPanelVisible;
        const panel = document.getElementById('performanceDebugPanel');
        if (panel) {
            panel.style.display = this.debugPanelVisible ? 'block' : 'none';
        }
        return this.debugPanelVisible;
    }

    /**
     * Update debug panel with current stats
     */
    updateDebugPanel() {
        if (!this.debugPanelVisible) return;

        const panel = document.getElementById('performanceDebugPanel');
        if (!panel) return;

        const stats = this.getStats();
        const statusColor = stats.status === 'good' ? '#27ae60' :
                           stats.status === 'warning' ? '#f39c12' : '#e74c3c';

        panel.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <span>⚡ Performance</span>
                <span style="color: ${statusColor}; font-size: 12px;">${stats.fps} FPS</span>
            </div>
            <div style="font-size: 11px; line-height: 1.6;">
                <div>Frame: ${stats.avgFrameTime}ms</div>
                <div>Nodes: ${stats.nodes.rendered}/${stats.nodes.total}
                    (${stats.nodes.total > 0 ? ((stats.nodes.rendered / stats.nodes.total) * 100).toFixed(0) : 0}%)</div>
                <div>Connections: ${stats.connections.rendered}/${stats.connections.total}
                    (${stats.connections.total > 0 ? ((stats.connections.rendered / stats.connections.total) * 100).toFixed(0) : 0}%)</div>
            </div>
        `;
    }
}

/**
 * Lazy Property Loading - Only load node config when selected
 * Reduces initial load time for large workflows
 */
const lazyPropertyLoader = {
    loadedConfigs: new Map(),
    pendingLoads: new Map(),

    /**
     * Check if a node's config has been fully loaded
     */
    isConfigLoaded(nodeId) {
        return this.loadedConfigs.has(nodeId);
    },

    /**
     * Load a node's configuration (async for large configs)
     */
    async loadNodeConfig(node) {
        // Already loaded
        if (this.loadedConfigs.has(node.id)) {
            return this.loadedConfigs.get(node.id);
        }

        // Already loading
        if (this.pendingLoads.has(node.id)) {
            return this.pendingLoads.get(node.id);
        }

        // Start loading
        const loadPromise = new Promise((resolve) => {
            // For now, configs are stored in-memory, but this could be async
            // from backend for very large workflows
            requestAnimationFrame(() => {
                const config = node.config || {};
                this.loadedConfigs.set(node.id, config);
                this.pendingLoads.delete(node.id);
                resolve(config);
            });
        });

        this.pendingLoads.set(node.id, loadPromise);
        return loadPromise;
    },

    /**
     * Clear cached config for a node
     */
    clearConfig(nodeId) {
        this.loadedConfigs.delete(nodeId);
    },

    /**
     * Clear all cached configs
     */
    clearAll() {
        this.loadedConfigs.clear();
        this.pendingLoads.clear();
    },

    /**
     * Get stats about config caching
     */
    getStats() {
        return {
            loadedConfigs: this.loadedConfigs.size,
            pendingLoads: this.pendingLoads.size,
            totalNodes: nodes.length
        };
    }
};

// Wave 7.3 singleton instances
let virtualCanvas = null;
let connectionCuller = null;
let performanceMonitor = null;
let wave73Enabled = false;

/**
 * Initialize Wave 7.3 performance features
 */
function initWave73Performance() {
    const canvas = document.getElementById('canvas');
    const svgLayer = document.getElementById('connectionsLayer');

    if (!canvas) {
        console.warn('[Wave 7.3] Canvas element not found');
        return;
    }

    // Create performance instances
    virtualCanvas = new VirtualCanvas(canvas, 200);
    connectionCuller = new ConnectionCuller(svgLayer);
    performanceMonitor = new PerformanceMonitor();

    // Create debug panel
    createPerformanceDebugPanel();

    // Hook into pan/zoom events
    hookPerformanceUpdates();

    wave73Enabled = true;
    console.log('[Wave 7.3] Performance optimization initialized');
}

/**
 * Create the performance debug panel element
 */
function createPerformanceDebugPanel() {
    // Check if panel already exists
    if (document.getElementById('performanceDebugPanel')) return;

    const panel = document.createElement('div');
    panel.id = 'performanceDebugPanel';
    panel.className = 'performance-debug-panel';
    panel.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        width: 180px;
        background: rgba(0, 0, 0, 0.85);
        color: #fff;
        padding: 12px;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        z-index: 10000;
        display: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    `;

    document.body.appendChild(panel);
}

/**
 * Hook performance updates into pan/zoom events
 */
function hookPerformanceUpdates() {
    // Hook into applyPan
    const originalApplyPan = window.applyPan || applyPan;
    if (typeof originalApplyPan === 'function') {
        window._originalApplyPan = originalApplyPan;
    }

    // Hook into applyZoom
    const originalApplyZoom = window.applyZoom || applyZoom;
    if (typeof originalApplyZoom === 'function') {
        window._originalApplyZoom = originalApplyZoom;
    }

    // Schedule viewport updates on animation frames
    let updatePending = false;

    window.schedulePerformanceUpdate = function() {
        if (updatePending || !wave73Enabled) return;

        updatePending = true;
        requestAnimationFrame(() => {
            if (performanceMonitor) performanceMonitor.startFrame();

            if (virtualCanvas) virtualCanvas.updateVisibleNodes();
            if (connectionCuller && virtualCanvas) {
                connectionCuller.updateVisibleConnections(virtualCanvas);
                performanceMonitor?.recordConnectionCount(
                    connectionCuller.renderedConnections.size,
                    connections.length
                );
            }

            if (performanceMonitor) {
                performanceMonitor.endFrame();
                performanceMonitor.updateDebugPanel();
            }

            updatePending = false;
        });
    };
}

/**
 * Toggle performance debug panel
 */
function togglePerformanceDebugPanel() {
    if (performanceMonitor) {
        return performanceMonitor.toggleDebugPanel();
    }
    return false;
}

/**
 * Get performance statistics
 */
function getPerformanceStats() {
    return {
        virtualCanvas: virtualCanvas?.getStats() || null,
        connectionCuller: connectionCuller?.getStats() || null,
        performanceMonitor: performanceMonitor?.getStats() || null,
        lazyLoader: lazyPropertyLoader.getStats()
    };
}

/**
 * Enable/disable Wave 7.3 performance features
 */
function setPerformanceOptimizationEnabled(enabled) {
    wave73Enabled = enabled;

    if (enabled && !virtualCanvas) {
        initWave73Performance();
    }

    console.log(`[Wave 7.3] Performance optimization ${enabled ? 'enabled' : 'disabled'}`);
}

/**
 * Force a full render refresh (use when adding many nodes)
 */
function forcePerformanceRefresh() {
    if (virtualCanvas) {
        virtualCanvas.clear();
        virtualCanvas.forceUpdate();
    }
    if (connectionCuller) {
        connectionCuller.clear();
        if (virtualCanvas) {
            connectionCuller.updateVisibleConnections(virtualCanvas);
        }
    }
}

// Keyboard shortcut for debug panel (P key)
document.addEventListener('keydown', (e) => {
    if (e.key === 'p' || e.key === 'P') {
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
            if (document.activeElement.tagName !== 'INPUT' &&
                document.activeElement.tagName !== 'TEXTAREA' &&
                document.activeElement.tagName !== 'SELECT') {
                e.preventDefault();
                togglePerformanceDebugPanel();
            }
        }
    }
});

// Initialize Wave 7.3 on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWave73Performance);
} else {
    initWave73Performance();
}

console.log('[Wave 7.3] Performance optimization module loaded');

// Export for console debugging
window.workflowBuilder = {
    nodes,
    connections,
    executionState,
    versionHistory,
    currentVersion,
    currentBranch,
    exportWorkflow,
    importWorkflow,
    executeWorkflow,
    saveWorkflowVersion,
    createBranch,
    loadVersion,
    showVersionHistoryModal,
    zoomIn,
    zoomOut,
    resetZoom,
    showTemplatesModal,
    loadTemplate,
    saveAsTemplate,
    // AI Suggestions (Wave 3.2)
    showSuggestions,
    hideSuggestions,
    addSuggestedNode,
    triggerPatternLearning,
    getExecutionOrder,
    // Performance Dashboard (Wave 4.2)
    showPerfDashboard,
    hidePerfDashboard,
    updatePerfDashboard,
    exportPerfData,
    recordExecutionMetrics,
    executionMetrics,
    thompsonPriors,
    // Auto-Layout (Wave 4.3)
    showLayoutPanel,
    hideLayoutPanel,
    runAutoLayout,
    layoutConfig,
    applyLayoutFromPanel,
    calculateHierarchicalLayout,
    calculateForceDirectedLayout,
    calculateRadialLayout,
    // Node Grouping (Wave 4.4)
    createGroupFromSelection,
    renderGroupNode,
    toggleGroupCollapse,
    cycleGroupColor,
    deleteGroupNode,
    addNodeToGroup,
    removeNodeFromGroup,
    // Search & Navigation (Wave 5.1)
    focusNodeSearch,
    handleNodeSearch,
    fuzzySearchNodes,
    jumpToNode,
    openCommandPalette,
    closeCommandPalette,
    populateCommandPalette,
    addAgentFromPalette,
    navigateToRoot,
    navigateToGroup,
    navigateToBreadcrumb,
    updateBreadcrumbs,
    searchResults,
    recentAgents,
    currentGroupPath,
    // Context Menus & UX (Wave 5.2)
    hideAllContextMenus,
    showContextMenu,
    handleCanvasContextMenu,
    findNodeAtPosition,
    findConnectionAtPosition,
    contextMenuEditConfig,
    contextMenuDuplicate,
    contextMenuCopy,
    contextMenuAddToGroup,
    contextMenuToggleBreakpoint,
    contextMenuDisconnectAll,
    contextMenuDeleteNode,
    contextMenuAddNodeHere,
    contextMenuPaste,
    selectAllNodes,
    fitToView,
    clearCanvas,
    contextMenuShowDataFlow,
    contextMenuDeleteConnection,
    contextMenuCreateGroup,
    alignSelectedNodes,
    distributeSelectedNodes,
    deleteSelectedNodes,
    duplicateNode,
    toggleNodeSelection,
    clearSelection,
    selectedNodes,
    copiedNodes,
    // Auto-Save (Wave 5.2)
    initAutoSave,
    autoSaveWorkflow,
    updateAutoSaveIndicator,
    loadAutoSavedDraft,
    toggleAutoSave,
    autoSaveEnabled,
    // Theme System (Wave 5.3)
    toggleTheme,
    setTheme,
    getCurrentTheme,
    getPreferredTheme,
    resetThemeToSystem,
    initTheme,
    currentTheme,
    // Debug System (Wave 5.5)
    toggleDebugMode,
    debugModeActive,
    debugRun,
    debugPause,
    debugStepOver,
    debugStepInto,
    debugStop,
    toggleVariableInspector,
    showVariableInspector,
    hideVariableInspector,
    updateVariableInspector,
    copyInspectorData,
    debugState,
    debugNodeOutputs,
    setNodeExecutionState,
    clearExecutionStates,
    getDebugExecutionOrder,
    // WebSocket Integration (Wave 5.6)
    connectWebSocket,
    disconnectWebSocket,
    wsExecuteWorkflow,
    wsSend,
    getWsConnectionState,
    showConnectionModal,
    hideConnectionModal,
    toggleLogsPanel,
    showLogsPanel,
    hideLogsPanel,
    addLog,
    clearLogs,
    exportLogs,
    filterLogs,
    toggleAutoScroll,
    updateConnectionStatus,
    handleNodeStart,
    handleNodeComplete,
    handleNodeError,
    showWorkflowProgress,
    updateWorkflowProgress,
    executionLogs,
    wsConnectionState,
    // Mobile/Touch Support (Wave 6)
    showMobileContextSheet,
    closeMobileContextSheet,
    showMobileProperties,
    closeMobileProperties,
    saveMobileProperties,
    resetPan,
    triggerHapticFeedback,
    initCanvasTouchGestures,
    createTouchRipple,
    panOffsetX,
    panOffsetY,
    touchState,
    // Real-Time Collaboration (Wave 7.1)
    showJoinSessionModal,
    closeJoinSessionModal,
    joinOrCreateSession,
    createCollaborationSession,
    joinCollaborationSession,
    leaveCollaborationSession,
    copySessionLink,
    renderPresencePanel,
    togglePresencePanel,
    updatePresence,
    updateRemoteCursor,
    removeRemoteCursor,
    showNodeLock,
    releaseNodeLock,
    requestNodeLock,
    releaseNodeLockRequest,
    broadcastOperation,
    handleRemoteOperation,
    collaborationSessionId,
    collaborationParticipants,
    remoteCursors,
    nodeLocks,
    localUserId,
    localUserName,
    localUserColor,
    // Advanced Workflows (Wave 7.2)
    drillIntoComposite,
    drillUp,
    updateBreadcrumbNavigation,
    navigateToBreadcrumbLevel,
    renderMiniWorkflow,
    renderCompositeNodeContent,
    handleNodeDoubleClick,
    loadSavedWorkflowsRegistry,
    saveWorkflowToRegistry,
    getSavedWorkflowsList,
    showMarketplaceModal,
    closeMarketplaceModal,
    loadMarketplaceTemplates,
    filterMarketplace,
    renderMarketplaceGrid,
    installMarketplaceTemplate,
    initWave72Features,
    compositeNavigationStack,
    savedWorkflowsRegistry,
    marketplaceTemplates,
    // Performance Optimization (Wave 7.3)
    VirtualCanvas,
    ConnectionCuller,
    PerformanceMonitor,
    lazyPropertyLoader,
    virtualCanvas,
    connectionCuller,
    performanceMonitor,
    wave73Enabled,
    initWave73Performance,
    togglePerformanceDebugPanel,
    getPerformanceStats,
    setPerformanceOptimizationEnabled,
    forcePerformanceRefresh
};
