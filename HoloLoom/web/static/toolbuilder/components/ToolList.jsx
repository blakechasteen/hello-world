// Tool List Component

function ToolList({ tools, onEdit, onRefresh }) {
    const [filter, setFilter] = React.useState('');
    const [categoryFilter, setCategoryFilter] = React.useState('all');

    async function deleteTool(toolId) {
        if (!confirm('Are you sure you want to delete this tool?')) {
            return;
        }

        try {
            const response = await fetch(`/api/toolbuilder/tools/${toolId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                alert('Tool deleted successfully');
                onRefresh();
            } else {
                const data = await response.json();
                alert('Failed to delete tool: ' + (data.detail || 'Unknown error'));
            }

        } catch (err) {
            alert('Error deleting tool: ' + err.message);
        }
    }

    async function toggleEnabled(toolId, currentState) {
        try {
            const endpoint = currentState ? 'disable' : 'enable';
            const response = await fetch(`/api/toolbuilder/tools/${toolId}/${endpoint}`, {
                method: 'POST'
            });

            if (response.ok) {
                onRefresh();
            } else {
                alert('Failed to update tool status');
            }

        } catch (err) {
            alert('Error updating tool: ' + err.message);
        }
    }

    // Filter tools
    const filteredTools = tools.filter(tool => {
        const matchesSearch = !filter ||
            tool.name.toLowerCase().includes(filter.toLowerCase()) ||
            tool.description.toLowerCase().includes(filter.toLowerCase());

        const matchesCategory = categoryFilter === 'all' || tool.category === categoryFilter;

        return matchesSearch && matchesCategory;
    });

    // Get unique categories
    const categories = ['all', ...new Set(tools.map(t => t.category))];

    return (
        <div className="tool-list">
            <div className="tool-list-header">
                <h2>My Custom Tools</h2>
                <button className="btn-refresh" onClick={onRefresh}>
                    <i className="fas fa-sync"></i> Refresh
                </button>
            </div>

            <div className="filter-section">
                <div className="search-box">
                    <i className="fas fa-search"></i>
                    <input
                        type="text"
                        placeholder="Search tools..."
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                    />
                </div>

                <div className="category-filter">
                    <label>Category:</label>
                    <select value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
                        {categories.map(cat => (
                            <option key={cat} value={cat}>
                                {cat === 'all' ? 'All Categories' : cat}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {filteredTools.length === 0 && (
                <div className="empty-state">
                    <i className="fas fa-tools fa-3x"></i>
                    <p>No tools found</p>
                    {filter && <p className="hint">Try adjusting your search or filters</p>}
                </div>
            )}

            {filteredTools.length > 0 && (
                <div className="tool-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Description</th>
                                <th>Category</th>
                                <th>Version</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredTools.map(tool => (
                                <tr key={tool.id}>
                                    <td>
                                        <strong>{tool.name}</strong>
                                        <br />
                                        <small className="tool-id">{tool.id}</small>
                                    </td>
                                    <td>{tool.description}</td>
                                    <td>
                                        <span className="category-badge">{tool.category}</span>
                                    </td>
                                    <td>{tool.version}</td>
                                    <td>
                                        <span className={`status-badge ${tool.enabled ? 'enabled' : 'disabled'}`}>
                                            {tool.enabled ? 'Enabled' : 'Disabled'}
                                        </span>
                                    </td>
                                    <td>
                                        <div className="tool-actions">
                                            <button
                                                className="btn-action"
                                                onClick={() => onEdit(tool)}
                                                title="Edit"
                                            >
                                                <i className="fas fa-edit"></i>
                                            </button>
                                            <button
                                                className="btn-action"
                                                onClick={() => toggleEnabled(tool.id, tool.enabled)}
                                                title={tool.enabled ? 'Disable' : 'Enable'}
                                            >
                                                <i className={`fas fa-${tool.enabled ? 'toggle-on' : 'toggle-off'}`}></i>
                                            </button>
                                            <button
                                                className="btn-action danger"
                                                onClick={() => deleteTool(tool.id)}
                                                title="Delete"
                                            >
                                                <i className="fas fa-trash"></i>
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            <div className="tool-stats">
                <div className="stat">
                    <strong>{tools.length}</strong>
                    <span>Total Tools</span>
                </div>
                <div className="stat">
                    <strong>{tools.filter(t => t.enabled).length}</strong>
                    <span>Enabled</span>
                </div>
                <div className="stat">
                    <strong>{tools.filter(t => !t.enabled).length}</strong>
                    <span>Disabled</span>
                </div>
            </div>

            <style jsx>{`
                .tool-list {
                    max-width: 1200px;
                    margin: 0 auto;
                }

                .filter-section {
                    display: flex;
                    gap: 1rem;
                    margin-bottom: 1.5rem;
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                }

                .search-box {
                    flex: 1;
                    position: relative;
                }

                .search-box i {
                    position: absolute;
                    left: 1rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #94a3b8;
                }

                .search-box input {
                    width: 100%;
                    padding: 0.75rem 1rem 0.75rem 2.75rem;
                    border: 2px solid var(--border-color);
                    border-radius: 0.5rem;
                    font-size: 1rem;
                }

                .search-box input:focus {
                    outline: none;
                    border-color: var(--primary-color);
                }

                .category-filter {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .category-filter label {
                    color: var(--text-dark);
                    font-weight: 600;
                }

                .category-filter select {
                    padding: 0.75rem;
                    border: 2px solid var(--border-color);
                    border-radius: 0.5rem;
                    font-size: 1rem;
                }

                .tool-id {
                    color: #94a3b8;
                    font-family: monospace;
                    font-size: 0.75rem;
                }

                .category-badge {
                    background: var(--secondary-color);
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 1rem;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                }

                .tool-stats {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 1rem;
                    margin-top: 1.5rem;
                }

                .stat {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    text-align: center;
                }

                .stat strong {
                    display: block;
                    font-size: 2rem;
                    color: var(--primary-color);
                    margin-bottom: 0.5rem;
                }

                .stat span {
                    color: #64748b;
                    font-size: 0.875rem;
                }

                @media (max-width: 768px) {
                    .filter-section {
                        flex-direction: column;
                    }

                    .tool-table {
                        overflow-x: auto;
                    }

                    .tool-stats {
                        grid-template-columns: 1fr;
                    }
                }
            `}</style>
        </div>
    );
}
