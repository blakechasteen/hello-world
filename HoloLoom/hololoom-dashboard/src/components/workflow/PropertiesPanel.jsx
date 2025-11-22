import React, { useEffect, useState } from 'react';
import { useReactFlow } from 'reactflow';
import { GlassPanel, GlassInput, GlassButton } from '../ui/Glass';

const PropertiesPanel = ({ selectedNodeId }) => {
    const { getNode, setNodes } = useReactFlow();
    const [nodeData, setNodeData] = useState(null);

    // Sync local state with selected node
    useEffect(() => {
        if (selectedNodeId) {
            const node = getNode(selectedNodeId);
            if (node) {
                setNodeData({ ...node.data });
            }
        } else {
            setNodeData(null);
        }
    }, [selectedNodeId, getNode]);

    const handleChange = (field, value) => {
        setNodeData((prev) => ({ ...prev, [field]: value }));

        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === selectedNodeId) {
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            [field]: value,
                        },
                    };
                }
                return node;
            })
        );
    };

    if (!selectedNodeId || !nodeData) {
        return (
            <div className="text-muted" style={{ padding: '1rem' }}>
                Select a node to configure its properties.
            </div>
        );
    }

    return (
        <div className="properties-form">
            <div className="form-group" style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                    Label
                </label>
                <GlassInput
                    value={nodeData.label || ''}
                    onChange={(e) => handleChange('label', e.target.value)}
                />
            </div>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
                <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                    Description
                </label>
                <GlassInput
                    value={nodeData.description || ''}
                    onChange={(e) => handleChange('description', e.target.value)}
                />
            </div>

            {/* Dynamic fields based on type could go here */}
            {nodeData.type === 'query' && (
                <div className="form-group" style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
                        Pattern
                    </label>
                    <select
                        className="glass-input"
                        value={nodeData.pattern || 'fast'}
                        onChange={(e) => handleChange('pattern', e.target.value)}
                        style={{ width: '100%' }}
                    >
                        <option value="bare">Bare (Lite)</option>
                        <option value="fast">Fast (Standard)</option>
                        <option value="fused">Fused (Deep)</option>
                    </select>
                </div>
            )}

            <div style={{ marginTop: '2rem', borderTop: '1px solid var(--glass-border)', paddingTop: '1rem' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>
                    Node ID: {selectedNodeId}
                </div>
            </div>
        </div>
    );
};

export default PropertiesPanel;
