// Code Preview Component

function CodePreview({ code, toolId }) {
    const [displayCode, setDisplayCode] = React.useState(code);
    const [loading, setLoading] = React.useState(false);
    const [copied, setCopied] = React.useState(false);

    React.useEffect(() => {
        setDisplayCode(code);
    }, [code]);

    async function regenerateCode() {
        setLoading(true);

        try {
            const response = await fetch(`/api/toolbuilder/tools/${toolId}/regenerate`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                setDisplayCode(data.code);
            } else {
                alert('Failed to regenerate code: ' + (data.detail || 'Unknown error'));
            }

        } catch (err) {
            alert('Error regenerating code: ' + err.message);
        } finally {
            setLoading(false);
        }
    }

    function copyToClipboard() {
        navigator.clipboard.writeText(displayCode).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        });
    }

    function downloadCode() {
        const blob = new Blob([displayCode], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${toolId}.py`;
        a.click();
        URL.revokeObjectURL(url);
    }

    return (
        <div className="code-preview">
            <div className="code-header">
                <h2>Generated Python Code</h2>
                <div className="code-actions">
                    <button
                        className="btn-action"
                        onClick={regenerateCode}
                        disabled={loading}
                    >
                        <i className="fas fa-sync"></i>
                        {loading ? 'Regenerating...' : 'Regenerate'}
                    </button>
                    <button className="btn-action" onClick={copyToClipboard}>
                        <i className={`fas fa-${copied ? 'check' : 'copy'}`}></i>
                        {copied ? 'Copied!' : 'Copy'}
                    </button>
                    <button className="btn-action" onClick={downloadCode}>
                        <i className="fas fa-download"></i>
                        Download
                    </button>
                </div>
            </div>

            {!displayCode && (
                <div className="no-code">
                    <i className="fas fa-file-code fa-3x"></i>
                    <p>No code generated yet</p>
                    <p className="hint">Save the tool to generate Python code</p>
                </div>
            )}

            {displayCode && (
                <div className="code-container">
                    <pre className="code-block">
                        <code>{displayCode}</code>
                    </pre>
                </div>
            )}

            <div className="code-info">
                <h3>About Generated Code</h3>
                <ul>
                    <li>Generated code is MCP-compatible and ready to use</li>
                    <li>Code includes type hints and docstrings</li>
                    <li>All logic blocks are converted to Python functions</li>
                    <li>Security restrictions are enforced during execution</li>
                </ul>
            </div>

            <style jsx>{`
                .code-preview {
                    max-width: 1100px;
                }

                .code-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1.5rem;
                }

                .code-header h2 {
                    color: var(--text-dark);
                }

                .code-actions {
                    display: flex;
                    gap: 0.5rem;
                }

                .btn-action {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.75rem 1.25rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 0.875rem;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .btn-action:hover:not(:disabled) {
                    background: var(--primary-hover);
                    transform: translateY(-2px);
                }

                .btn-action:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }

                .no-code {
                    text-align: center;
                    padding: 3rem;
                    background: white;
                    border-radius: 0.5rem;
                    color: #94a3b8;
                }

                .no-code i {
                    color: #cbd5e1;
                    margin-bottom: 1rem;
                }

                .no-code .hint {
                    font-size: 0.875rem;
                    margin-top: 0.5rem;
                }

                .code-container {
                    background: #1e293b;
                    border-radius: 0.5rem;
                    overflow: hidden;
                    margin-bottom: 1.5rem;
                }

                .code-block {
                    margin: 0;
                    padding: 1.5rem;
                    overflow-x: auto;
                    color: #e2e8f0;
                    font-family: 'Courier New', monospace;
                    font-size: 0.875rem;
                    line-height: 1.6;
                }

                .code-block code {
                    color: #e2e8f0;
                }

                .code-info {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid var(--primary-color);
                }

                .code-info h3 {
                    color: var(--text-dark);
                    margin-bottom: 1rem;
                    font-size: 1.125rem;
                }

                .code-info ul {
                    list-style: none;
                    padding: 0;
                }

                .code-info li {
                    color: #64748b;
                    padding: 0.5rem 0;
                    padding-left: 1.5rem;
                    position: relative;
                }

                .code-info li:before {
                    content: '✓';
                    position: absolute;
                    left: 0;
                    color: var(--success-color);
                    font-weight: bold;
                }
            `}</style>
        </div>
    );
}
