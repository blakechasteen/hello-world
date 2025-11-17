// Parameter Configuration Component

function ParameterConfig({ toolDef, onChange }) {
    const [editingParam, setEditingParam] = React.useState(null);

    function addParameter() {
        const newParam = {
            name: `param_${toolDef.parameters.length + 1}`,
            type: 'text',
            description: '',
            required: true,
            default: null,
            options: null,
            validation: null
        };

        onChange({
            ...toolDef,
            parameters: [...toolDef.parameters, newParam]
        });

        setEditingParam(toolDef.parameters.length);
    }

    function updateParameter(index, updates) {
        const newParams = [...toolDef.parameters];
        newParams[index] = { ...newParams[index], ...updates };
        onChange({ ...toolDef, parameters: newParams });
    }

    function deleteParameter(index) {
        if (!confirm('Delete this parameter?')) return;

        onChange({
            ...toolDef,
            parameters: toolDef.parameters.filter((_, i) => i !== index)
        });

        if (editingParam === index) {
            setEditingParam(null);
        }
    }

    function moveParameter(index, direction) {
        const newParams = [...toolDef.parameters];
        const newIndex = index + direction;

        if (newIndex < 0 || newIndex >= newParams.length) return;

        [newParams[index], newParams[newIndex]] = [newParams[newIndex], newParams[index]];
        onChange({ ...toolDef, parameters: newParams });
    }

    return (
        <div className="parameter-config">
            <div className="section-header">
                <h2>Input Parameters</h2>
                <button className="btn-add" onClick={addParameter}>
                    <i className="fas fa-plus"></i> Add Parameter
                </button>
            </div>

            {toolDef.parameters.length === 0 && (
                <div className="empty-state">
                    <i className="fas fa-inbox fa-3x"></i>
                    <p>No parameters defined yet</p>
                    <p className="hint">Click "Add Parameter" to get started</p>
                </div>
            )}

            <div className="parameter-list">
                {toolDef.parameters.map((param, index) => (
                    <div key={index} className="parameter-item">
                        <div className="parameter-header">
                            <div className="parameter-info">
                                <strong>{param.name}</strong>
                                <span className="param-type">{param.type}</span>
                                {param.required && <span className="required-badge">Required</span>}
                            </div>
                            <div className="parameter-actions">
                                <button
                                    className="btn-icon"
                                    onClick={() => moveParameter(index, -1)}
                                    disabled={index === 0}
                                    title="Move up"
                                >
                                    <i className="fas fa-arrow-up"></i>
                                </button>
                                <button
                                    className="btn-icon"
                                    onClick={() => moveParameter(index, 1)}
                                    disabled={index === toolDef.parameters.length - 1}
                                    title="Move down"
                                >
                                    <i className="fas fa-arrow-down"></i>
                                </button>
                                <button
                                    className="btn-icon"
                                    onClick={() => setEditingParam(editingParam === index ? null : index)}
                                    title={editingParam === index ? "Collapse" : "Expand"}
                                >
                                    <i className={`fas fa-chevron-${editingParam === index ? 'up' : 'down'}`}></i>
                                </button>
                                <button
                                    className="btn-icon danger"
                                    onClick={() => deleteParameter(index)}
                                    title="Delete"
                                >
                                    <i className="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>

                        {editingParam === index && (
                            <div className="parameter-editor">
                                <div className="form-row">
                                    <div className="form-group">
                                        <label>Parameter Name *</label>
                                        <input
                                            type="text"
                                            value={param.name}
                                            onChange={(e) => updateParameter(index, { name: e.target.value })}
                                            placeholder="parameter_name"
                                        />
                                    </div>

                                    <div className="form-group">
                                        <label>Type *</label>
                                        <select
                                            value={param.type}
                                            onChange={(e) => updateParameter(index, { type: e.target.value })}
                                        >
                                            <option value="text">Text</option>
                                            <option value="number">Number</option>
                                            <option value="file">File</option>
                                            <option value="dropdown">Dropdown</option>
                                            <option value="checkbox">Checkbox</option>
                                            <option value="date">Date</option>
                                            <option value="json">JSON</option>
                                            <option value="array">Array</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="form-group">
                                    <label>Description *</label>
                                    <textarea
                                        value={param.description}
                                        onChange={(e) => updateParameter(index, { description: e.target.value })}
                                        placeholder="Describe this parameter..."
                                        rows="2"
                                    />
                                </div>

                                <div className="form-row">
                                    <div className="form-group">
                                        <label>
                                            <input
                                                type="checkbox"
                                                checked={param.required}
                                                onChange={(e) => updateParameter(index, { required: e.target.checked })}
                                            />
                                            {' '}Required
                                        </label>
                                    </div>

                                    {!param.required && (
                                        <div className="form-group">
                                            <label>Default Value</label>
                                            <input
                                                type="text"
                                                value={param.default || ''}
                                                onChange={(e) => updateParameter(index, { default: e.target.value })}
                                                placeholder="Default value"
                                            />
                                        </div>
                                    )}
                                </div>

                                {param.type === 'dropdown' && (
                                    <div className="form-group">
                                        <label>Options (comma-separated)</label>
                                        <input
                                            type="text"
                                            value={(param.options || []).join(', ')}
                                            onChange={(e) => updateParameter(index, {
                                                options: e.target.value.split(',').map(o => o.trim()).filter(Boolean)
                                            })}
                                            placeholder="option1, option2, option3"
                                        />
                                    </div>
                                )}

                                {(param.type === 'number' || param.type === 'text') && (
                                    <div className="validation-section">
                                        <h4>Validation Rules</h4>

                                        {param.type === 'number' && (
                                            <div className="form-row">
                                                <div className="form-group">
                                                    <label>Minimum</label>
                                                    <input
                                                        type="number"
                                                        value={param.validation?.min || ''}
                                                        onChange={(e) => updateParameter(index, {
                                                            validation: {
                                                                ...(param.validation || {}),
                                                                min: parseFloat(e.target.value)
                                                            }
                                                        })}
                                                    />
                                                </div>

                                                <div className="form-group">
                                                    <label>Maximum</label>
                                                    <input
                                                        type="number"
                                                        value={param.validation?.max || ''}
                                                        onChange={(e) => updateParameter(index, {
                                                            validation: {
                                                                ...(param.validation || {}),
                                                                max: parseFloat(e.target.value)
                                                            }
                                                        })}
                                                    />
                                                </div>
                                            </div>
                                        )}

                                        {param.type === 'text' && (
                                            <>
                                                <div className="form-row">
                                                    <div className="form-group">
                                                        <label>Min Length</label>
                                                        <input
                                                            type="number"
                                                            value={param.validation?.min_length || ''}
                                                            onChange={(e) => updateParameter(index, {
                                                                validation: {
                                                                    ...(param.validation || {}),
                                                                    min_length: parseInt(e.target.value)
                                                                }
                                                            })}
                                                        />
                                                    </div>

                                                    <div className="form-group">
                                                        <label>Max Length</label>
                                                        <input
                                                            type="number"
                                                            value={param.validation?.max_length || ''}
                                                            onChange={(e) => updateParameter(index, {
                                                                validation: {
                                                                    ...(param.validation || {}),
                                                                    max_length: parseInt(e.target.value)
                                                                }
                                                            })}
                                                        />
                                                    </div>
                                                </div>

                                                <div className="form-group">
                                                    <label>Pattern (Regex)</label>
                                                    <input
                                                        type="text"
                                                        value={param.validation?.pattern || ''}
                                                        onChange={(e) => updateParameter(index, {
                                                            validation: {
                                                                ...(param.validation || {}),
                                                                pattern: e.target.value
                                                            }
                                                        })}
                                                        placeholder="^[a-z]+$"
                                                    />
                                                </div>
                                            </>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <style jsx>{`
                .parameter-config {
                    max-width: 900px;
                }

                .section-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 2rem;
                }

                .section-header h2 {
                    color: var(--text-dark);
                }

                .btn-add {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 1rem;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .btn-add:hover {
                    background: var(--primary-hover);
                    transform: translateY(-2px);
                }

                .empty-state {
                    text-align: center;
                    padding: 3rem;
                    color: #94a3b8;
                }

                .empty-state i {
                    color: #cbd5e1;
                    margin-bottom: 1rem;
                }

                .empty-state .hint {
                    font-size: 0.875rem;
                    margin-top: 0.5rem;
                }

                .parameter-list {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }

                .parameter-item {
                    background: white;
                    border: 2px solid #e2e8f0;
                    border-radius: 0.5rem;
                    overflow: hidden;
                }

                .parameter-header {
                    padding: 1rem;
                    background: #f8fafc;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .parameter-info {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }

                .param-type {
                    background: var(--secondary-color);
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 1rem;
                    font-size: 0.75rem;
                }

                .required-badge {
                    background: var(--danger-color);
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 1rem;
                    font-size: 0.75rem;
                }

                .parameter-actions {
                    display: flex;
                    gap: 0.25rem;
                }

                .btn-icon {
                    background: transparent;
                    border: none;
                    color: #64748b;
                    padding: 0.5rem;
                    cursor: pointer;
                    border-radius: 0.375rem;
                    transition: all 0.2s;
                }

                .btn-icon:hover:not(:disabled) {
                    background: var(--primary-color);
                    color: white;
                }

                .btn-icon.danger:hover {
                    background: var(--danger-color);
                    color: white;
                }

                .btn-icon:disabled {
                    opacity: 0.3;
                    cursor: not-allowed;
                }

                .parameter-editor {
                    padding: 1rem;
                    border-top: 1px solid #e2e8f0;
                }

                .validation-section {
                    margin-top: 1.5rem;
                    padding: 1rem;
                    background: #f8fafc;
                    border-radius: 0.5rem;
                }

                .validation-section h4 {
                    color: var(--text-dark);
                    margin-bottom: 1rem;
                    font-size: 1rem;
                }
            `}</style>
        </div>
    );
}
