// Logic Builder Component

function LogicBuilder({ toolDef, onChange }) {
    const [expandedBlocks, setExpandedBlocks] = React.useState(new Set());

    function addLogicBlock(type) {
        const newBlock = {
            id: `block_${Date.now()}`,
            type: type,
        };

        onChange({
            ...toolDef,
            logic_blocks: [...toolDef.logic_blocks, newBlock]
        });
    }

    function updateBlock(index, updates) {
        const newBlocks = [...toolDef.logic_blocks];
        newBlocks[index] = { ...newBlocks[index], ...updates };
        onChange({ ...toolDef, logic_blocks: newBlocks });
    }

    function deleteBlock(index) {
        if (!confirm('Delete this logic block?')) return;

        onChange({
            ...toolDef,
            logic_blocks: toolDef.logic_blocks.filter((_, i) => i !== index)
        });
    }

    function toggleExpand(blockId) {
        const newExpanded = new Set(expandedBlocks);
        if (newExpanded.has(blockId)) {
            newExpanded.delete(blockId);
        } else {
            newExpanded.add(blockId);
        }
        setExpandedBlocks(newExpanded);
    }

    return (
        <div className="logic-builder">
            <div className="section-header">
                <h2>Logic Flow</h2>
                <div className="block-menu">
                    <button className="btn-add-block" onClick={() => addLogicBlock('if_else')}>
                        <i className="fas fa-code-branch"></i> If/Else
                    </button>
                    <button className="btn-add-block" onClick={() => addLogicBlock('for_loop')}>
                        <i className="fas fa-sync"></i> For Loop
                    </button>
                    <button className="btn-add-block" onClick={() => addLogicBlock('variable')}>
                        <i className="fas fa-equals"></i> Variable
                    </button>
                    <button className="btn-add-block" onClick={() => addLogicBlock('return')}>
                        <i className="fas fa-reply"></i> Return
                    </button>
                </div>
            </div>

            {toolDef.logic_blocks.length === 0 && (
                <div className="empty-state">
                    <i className="fas fa-project-diagram fa-3x"></i>
                    <p>No logic blocks defined yet</p>
                    <p className="hint">Add logic blocks to define your tool's behavior</p>
                </div>
            )}

            <div className="logic-flow">
                {toolDef.logic_blocks.map((block, index) => (
                    <div key={block.id} className="logic-block">
                        <div className="block-header">
                            <div className="block-info">
                                <i className={`fas fa-${getBlockIcon(block.type)}`}></i>
                                <strong>{formatBlockType(block.type)}</strong>
                                <span className="block-id">{block.id}</span>
                            </div>
                            <div className="block-actions">
                                <button
                                    className="btn-icon"
                                    onClick={() => toggleExpand(block.id)}
                                >
                                    <i className={`fas fa-chevron-${expandedBlocks.has(block.id) ? 'up' : 'down'}`}></i>
                                </button>
                                <button
                                    className="btn-icon danger"
                                    onClick={() => deleteBlock(index)}
                                >
                                    <i className="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>

                        {expandedBlocks.has(block.id) && (
                            <div className="block-editor">
                                {block.type === 'if_else' && (
                                    <IfElseEditor block={block} onChange={(updates) => updateBlock(index, updates)} />
                                )}
                                {block.type === 'for_loop' && (
                                    <ForLoopEditor block={block} onChange={(updates) => updateBlock(index, updates)} />
                                )}
                                {block.type === 'variable' && (
                                    <VariableEditor block={block} onChange={(updates) => updateBlock(index, updates)} />
                                )}
                                {block.type === 'return' && (
                                    <ReturnEditor block={block} onChange={(updates) => updateBlock(index, updates)} />
                                )}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <style jsx>{`
                .logic-builder {
                    max-width: 900px;
                }

                .section-header {
                    margin-bottom: 2rem;
                }

                .section-header h2 {
                    color: var(--text-dark);
                    margin-bottom: 1rem;
                }

                .block-menu {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.5rem;
                }

                .btn-add-block {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 0.875rem;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .btn-add-block:hover {
                    background: var(--primary-hover);
                    transform: translateY(-2px);
                }

                .logic-flow {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }

                .logic-block {
                    background: white;
                    border: 2px solid #e2e8f0;
                    border-radius: 0.5rem;
                    overflow: hidden;
                }

                .block-header {
                    padding: 1rem;
                    background: #f8fafc;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .block-info {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }

                .block-info i {
                    color: var(--primary-color);
                }

                .block-id {
                    font-size: 0.75rem;
                    color: #94a3b8;
                    font-family: monospace;
                }

                .block-editor {
                    padding: 1rem;
                    border-top: 1px solid #e2e8f0;
                }
            `}</style>
        </div>
    );
}

// Helper components for different block types

function IfElseEditor({ block, onChange }) {
    return (
        <div className="if-else-editor">
            <div className="form-group">
                <label>Condition</label>
                <div className="condition-builder">
                    <input
                        type="text"
                        placeholder="Left value (e.g., ${variable})"
                        value={block.condition?.left || ''}
                        onChange={(e) => onChange({
                            condition: { ...(block.condition || {}), left: e.target.value }
                        })}
                    />
                    <select
                        value={block.condition?.operator || '=='}
                        onChange={(e) => onChange({
                            condition: { ...(block.condition || {}), operator: e.target.value }
                        })}
                    >
                        <option value="==">==</option>
                        <option value="!=">!=</option>
                        <option value="<">&lt;</option>
                        <option value=">">&gt;</option>
                        <option value="<=">≤</option>
                        <option value=">=">≥</option>
                        <option value="in">in</option>
                        <option value="not in">not in</option>
                    </select>
                    <input
                        type="text"
                        placeholder="Right value"
                        value={block.condition?.right || ''}
                        onChange={(e) => onChange({
                            condition: { ...(block.condition || {}), right: e.target.value }
                        })}
                    />
                </div>
            </div>
            <p className="hint">Note: Nested blocks are not yet supported in the UI. Edit JSON directly for complex logic.</p>
        </div>
    );
}

function ForLoopEditor({ block, onChange }) {
    return (
        <div className="for-loop-editor">
            <div className="form-row">
                <div className="form-group">
                    <label>Iterator Variable</label>
                    <input
                        type="text"
                        placeholder="item"
                        value={block.iterator || ''}
                        onChange={(e) => onChange({ iterator: e.target.value })}
                    />
                </div>
                <div className="form-group">
                    <label>Iterate Over</label>
                    <input
                        type="text"
                        placeholder="${array_variable}"
                        value={block.iterable || ''}
                        onChange={(e) => onChange({ iterable: e.target.value })}
                    />
                </div>
            </div>
            <p className="hint">Loop body blocks not yet supported in UI. Edit JSON for complex loops.</p>
        </div>
    );
}

function VariableEditor({ block, onChange }) {
    return (
        <div className="variable-editor">
            <div className="form-row">
                <div className="form-group">
                    <label>Variable Name</label>
                    <input
                        type="text"
                        placeholder="my_variable"
                        value={block.variable_name || ''}
                        onChange={(e) => onChange({ variable_name: e.target.value })}
                    />
                </div>
                <div className="form-group">
                    <label>Value</label>
                    <input
                        type="text"
                        placeholder="${param} or 'value'"
                        value={block.variable_value || ''}
                        onChange={(e) => onChange({ variable_value: e.target.value })}
                    />
                </div>
            </div>
        </div>
    );
}

function ReturnEditor({ block, onChange }) {
    return (
        <div className="return-editor">
            <div className="form-group">
                <label>Return Value</label>
                <input
                    type="text"
                    placeholder="${result}"
                    value={block.return_value || ''}
                    onChange={(e) => onChange({ return_value: e.target.value })}
                />
                <small>Use ${'{'}variable{'}'} syntax to reference variables</small>
            </div>
        </div>
    );
}

// Helper functions
function getBlockIcon(type) {
    const icons = {
        'if_else': 'code-branch',
        'for_loop': 'sync',
        'while_loop': 'circle-notch',
        'transform': 'magic',
        'variable': 'equals',
        'return': 'reply',
    };
    return icons[type] || 'cube';
}

function formatBlockType(type) {
    return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}
