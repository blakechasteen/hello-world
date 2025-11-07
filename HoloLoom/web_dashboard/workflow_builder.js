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
    }
};

// Initialize drag-and-drop
document.addEventListener('DOMContentLoaded', () => {
    initializeDragAndDrop();
    initializeCanvas();
    setupEventListeners();
});

function initializeDragAndDrop() {
    const templates = document.querySelectorAll('.agent-template');

    templates.forEach(template => {
        template.addEventListener('dragstart', (e) => {
            const agentType = template.dataset.agent;
            e.dataTransfer.setData('agentType', agentType);
            e.dataTransfer.effectAllowed = 'copy';
        });
    });
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
    showToast(`Added ${definition.name}`, 'success');
}

function renderNode(node) {
    const canvas = document.getElementById('canvas');
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

    // Make draggable
    makeNodeDraggable(nodeEl, node);

    // Click to select
    nodeEl.addEventListener('click', (e) => {
        e.stopPropagation();
        selectNode(node.id);
    });

    canvas.appendChild(nodeEl);
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

    element.addEventListener('mousedown', (e) => {
        if (e.target.closest('.node-delete') || e.target.closest('.node-port') ||
            e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
            return;
        }

        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        initialX = node.x;
        initialY = node.y;
        element.classList.add('dragging');

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;

        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        node.x = initialX + dx;
        node.y = initialY + dy;

        element.style.left = node.x + 'px';
        element.style.top = node.y + 'px';

        updateConnections();
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
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
    saveUndoState();
}

function deleteNode(nodeId) {
    if (!confirm('Delete this node?')) return;

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
    connections = connections.filter(c => c.id !== connId);
    updateConnections();
    showToast('Connection deleted', 'success');
}

function clearCanvas() {
    if (!confirm('Clear all nodes and connections?')) return;

    nodes = [];
    connections = [];
    selectedNode = null;

    document.getElementById('canvas').innerHTML = '';
    updateConnections();
    deselectAll();
    showToast('Canvas cleared', 'success');
}

function exportWorkflow() {
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

    const json = JSON.stringify(workflow, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow-${Date.now()}.json`;
    a.click();

    showToast('Workflow exported', 'success');
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
    clearCanvas();

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
        control: '🔀'
    };
    return icons[type] || '📦';
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

    canvas.style.transform = `scale(${zoomLevel})`;
    canvas.style.transformOrigin = '0 0';

    connectionsLayer.style.transform = `scale(${zoomLevel})`;
    connectionsLayer.style.transformOrigin = '0 0';

    // Update zoom indicator
    document.getElementById('zoomLevel').textContent = `${Math.round(zoomLevel * 100)}%`;

    // Redraw connections to match new scale
    renderConnections();
}

// Keyboard shortcuts for zoom
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
        }
    }
});

// ========== HOTKEY FUNCTIONS ==========
let clipboardNode = null;
let undoStack = [];
let redoStack = [];
const MAX_UNDO_STACK = 50;

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
    saveUndoState();
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
    saveUndoState();
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

function saveUndoState() {
    const state = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections))
    };

    undoStack.push(state);
    if (undoStack.length > MAX_UNDO_STACK) {
        undoStack.shift();
    }

    // Clear redo stack when new action is performed
    redoStack = [];
}

function undo() {
    if (undoStack.length === 0) {
        showToast('Nothing to undo', 'error');
        return;
    }

    // Save current state to redo stack
    const currentState = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        connections: JSON.parse(JSON.stringify(connections))
    };
    redoStack.push(currentState);

    // Restore previous state
    const prevState = undoStack.pop();
    nodes.length = 0;
    connections.length = 0;
    nodes.push(...prevState.nodes);
    connections.push(...prevState.connections);

    // Re-render
    clearCanvas();
    nodes.forEach(node => renderNode(node));
    renderConnections();

    showToast('Undo', 'success');
}

function redo() {
    if (redoStack.length === 0) {
        showToast('Nothing to redo', 'error');
        return;
    }

    // Save current state to undo stack
    saveUndoState();

    // Restore redo state
    const nextState = redoStack.pop();
    nodes.length = 0;
    connections.length = 0;
    nodes.push(...nextState.nodes);
    connections.push(...nextState.connections);

    // Re-render
    clearCanvas();
    nodes.forEach(node => renderNode(node));
    renderConnections();

    showToast('Redo', 'success');
}

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
    saveAsTemplate
};
