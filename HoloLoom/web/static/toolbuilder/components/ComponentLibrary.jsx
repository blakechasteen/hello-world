// Component Library - Drag and Drop Components Reference

function ComponentLibrary() {
    const components = {
        inputs: [
            { id: 'text', icon: 'font', name: 'Text Input', desc: 'Single line text' },
            { id: 'number', icon: 'hashtag', name: 'Number Input', desc: 'Numeric value' },
            { id: 'file', icon: 'file', name: 'File Upload', desc: 'File selection' },
            { id: 'dropdown', icon: 'caret-down', name: 'Dropdown', desc: 'Select from options' },
            { id: 'checkbox', icon: 'check-square', name: 'Checkbox', desc: 'Boolean value' },
            { id: 'date', icon: 'calendar', name: 'Date Picker', desc: 'Date selection' },
        ],
        logic: [
            { id: 'if_else', icon: 'code-branch', name: 'If/Else', desc: 'Conditional logic' },
            { id: 'for_loop', icon: 'sync', name: 'For Loop', desc: 'Iterate over items' },
            { id: 'while_loop', icon: 'circle-notch', name: 'While Loop', desc: 'Conditional loop' },
            { id: 'transform', icon: 'magic', name: 'Transform', desc: 'Map/filter/reduce' },
            { id: 'variable', icon: 'equals', name: 'Variable', desc: 'Store a value' },
        ],
        actions: [
            { id: 'http', icon: 'globe', name: 'HTTP Request', desc: 'Call an API' },
            { id: 'database', icon: 'database', name: 'Database Query', desc: 'Query database' },
            { id: 'file_op', icon: 'file-alt', name: 'File Operation', desc: 'Read/write files' },
            { id: 'compute', icon: 'calculator', name: 'Compute', desc: 'Math or string ops' },
            { id: 'call_tool', icon: 'wrench', name: 'Call Tool', desc: 'Execute another tool' },
        ],
        outputs: [
            { id: 'return', icon: 'reply', name: 'Return', desc: 'Return a value' },
            { id: 'memory', icon: 'brain', name: 'To Memory', desc: 'Store in memory' },
        ]
    };

    return (
        <div className="component-library">
            <h3>Component Library</h3>
            <p className="description">Available building blocks for your tool</p>

            {Object.entries(components).map(([category, items]) => (
                <div key={category} className="component-category">
                    <h4>{category.charAt(0).toUpperCase() + category.slice(1)}</h4>
                    <div className="component-grid">
                        {items.map(item => (
                            <div key={item.id} className="component-card" title={item.desc}>
                                <i className={`fas fa-${item.icon}`}></i>
                                <span>{item.name}</span>
                            </div>
                        ))}
                    </div>
                </div>
            ))}

            <style jsx>{`
                .component-library {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    margin-bottom: 1.5rem;
                }

                .component-library h3 {
                    color: var(--text-dark);
                    margin-bottom: 0.5rem;
                }

                .description {
                    color: #64748b;
                    margin-bottom: 1.5rem;
                }

                .component-category {
                    margin-bottom: 1.5rem;
                }

                .component-category h4 {
                    color: var(--text-dark);
                    margin-bottom: 0.75rem;
                    font-size: 1rem;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }

                .component-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                    gap: 0.75rem;
                }

                .component-card {
                    background: #f8fafc;
                    border: 2px dashed var(--border-color);
                    padding: 1rem;
                    border-radius: 0.5rem;
                    text-align: center;
                    cursor: grab;
                    transition: all 0.2s;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 0.5rem;
                }

                .component-card:hover {
                    background: var(--primary-color);
                    color: white;
                    border-color: var(--primary-color);
                    transform: translateY(-2px);
                }

                .component-card i {
                    font-size: 1.5rem;
                }

                .component-card span {
                    font-size: 0.875rem;
                    font-weight: 500;
                }
            `}</style>
        </div>
    );
}
