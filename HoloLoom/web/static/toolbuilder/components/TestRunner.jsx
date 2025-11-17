// Test Runner Component

function TestRunner({ toolDef }) {
    const [testParams, setTestParams] = React.useState({});
    const [testResult, setTestResult] = React.useState(null);
    const [testing, setTesting] = React.useState(false);
    const [error, setError] = React.useState(null);

    React.useEffect(() => {
        // Initialize test parameters with defaults
        const initialParams = {};
        toolDef.parameters.forEach(param => {
            if (param.default !== null && param.default !== undefined) {
                initialParams[param.name] = param.default;
            } else if (param.type === 'number') {
                initialParams[param.name] = 0;
            } else if (param.type === 'checkbox') {
                initialParams[param.name] = false;
            } else {
                initialParams[param.name] = '';
            }
        });
        setTestParams(initialParams);
    }, [toolDef]);

    async function runTest() {
        setTesting(true);
        setError(null);
        setTestResult(null);

        try {
            const response = await fetch(`/api/toolbuilder/tools/${toolDef.id}/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ parameters: testParams })
            });

            const data = await response.json();

            if (response.ok) {
                setTestResult(data.result);
            } else {
                setError(data.error || 'Test failed');
            }

        } catch (err) {
            setError(err.message);
        } finally {
            setTesting(false);
        }
    }

    function updateParam(name, value) {
        setTestParams({
            ...testParams,
            [name]: value
        });
    }

    return (
        <div className="test-runner">
            <h2>Test Tool</h2>
            <p className="description">Test your tool with different input parameters</p>

            <div className="test-section">
                <h3>Input Parameters</h3>

                {toolDef.parameters.length === 0 && (
                    <p className="hint">No parameters defined for this tool</p>
                )}

                <div className="test-params">
                    {toolDef.parameters.map(param => (
                        <div key={param.name} className="form-group">
                            <label>
                                {param.name}
                                {param.required && <span className="required">*</span>}
                            </label>

                            {param.type === 'text' && (
                                <input
                                    type="text"
                                    value={testParams[param.name] || ''}
                                    onChange={(e) => updateParam(param.name, e.target.value)}
                                    placeholder={param.description}
                                />
                            )}

                            {param.type === 'number' && (
                                <input
                                    type="number"
                                    value={testParams[param.name] || ''}
                                    onChange={(e) => updateParam(param.name, parseFloat(e.target.value))}
                                    placeholder={param.description}
                                    min={param.validation?.min}
                                    max={param.validation?.max}
                                />
                            )}

                            {param.type === 'checkbox' && (
                                <input
                                    type="checkbox"
                                    checked={testParams[param.name] || false}
                                    onChange={(e) => updateParam(param.name, e.target.checked)}
                                />
                            )}

                            {param.type === 'dropdown' && (
                                <select
                                    value={testParams[param.name] || ''}
                                    onChange={(e) => updateParam(param.name, e.target.value)}
                                >
                                    <option value="">-- Select --</option>
                                    {(param.options || []).map(opt => (
                                        <option key={opt} value={opt}>{opt}</option>
                                    ))}
                                </select>
                            )}

                            {param.type === 'json' && (
                                <textarea
                                    value={testParams[param.name] || ''}
                                    onChange={(e) => updateParam(param.name, e.target.value)}
                                    placeholder='{"key": "value"}'
                                    rows="3"
                                />
                            )}

                            <small>{param.description}</small>
                        </div>
                    ))}
                </div>

                <button
                    className="btn-run-test"
                    onClick={runTest}
                    disabled={testing}
                >
                    <i className="fas fa-play"></i>
                    {testing ? 'Running Test...' : 'Run Test'}
                </button>
            </div>

            {error && (
                <div className="test-result error">
                    <h3><i className="fas fa-exclamation-triangle"></i> Test Failed</h3>
                    <pre>{error}</pre>
                </div>
            )}

            {testResult && (
                <div className="test-result success">
                    <h3><i className="fas fa-check-circle"></i> Test Successful</h3>
                    <pre>{JSON.stringify(testResult, null, 2)}</pre>
                </div>
            )}

            <style jsx>{`
                .test-runner {
                    max-width: 900px;
                }

                .test-runner h2 {
                    color: var(--text-dark);
                    margin-bottom: 0.5rem;
                }

                .description {
                    color: #64748b;
                    margin-bottom: 2rem;
                }

                .test-section {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1.5rem;
                }

                .test-section h3 {
                    color: var(--text-dark);
                    margin-bottom: 1rem;
                }

                .test-params {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    margin-bottom: 1.5rem;
                }

                .required {
                    color: var(--danger-color);
                    margin-left: 0.25rem;
                }

                .hint {
                    color: #94a3b8;
                    font-style: italic;
                }

                .btn-run-test {
                    background: var(--success-color);
                    color: white;
                    border: none;
                    padding: 1rem 2rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 1rem;
                    font-weight: 600;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }

                .btn-run-test:hover:not(:disabled) {
                    background: #059669;
                    transform: translateY(-2px);
                }

                .btn-run-test:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }

                .test-result {
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    border-left: 4px solid;
                }

                .test-result.success {
                    background: #d1fae5;
                    border-color: var(--success-color);
                }

                .test-result.success h3 {
                    color: #065f46;
                }

                .test-result.error {
                    background: #fee2e2;
                    border-color: var(--danger-color);
                }

                .test-result.error h3 {
                    color: #991b1b;
                }

                .test-result h3 {
                    margin-bottom: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .test-result pre {
                    background: rgba(0, 0, 0, 0.1);
                    padding: 1rem;
                    border-radius: 0.375rem;
                    overflow-x: auto;
                    font-size: 0.875rem;
                    line-height: 1.5;
                }
            `}</style>
        </div>
    );
}
