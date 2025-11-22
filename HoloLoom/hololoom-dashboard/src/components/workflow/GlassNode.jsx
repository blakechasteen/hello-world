import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';
import { GlassPanel } from '../ui/Glass';

const GlassNode = ({ data, selected }) => {
    return (
        <GlassPanel
            intensity="medium"
            className={`glass-node ${selected ? 'selected' : ''}`}
            style={{
                padding: '10px',
                minWidth: '150px',
                border: selected ? '1px solid var(--color-accent-primary)' : undefined
            }}
        >
            <Handle type="target" position={Position.Left} className="glass-handle" />

            <div className="node-header" style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <div className={`node-icon ${data.type}`} style={{ marginRight: '8px' }}>
                    {data.icon || '⚡'}
                </div>
                <div className="node-title" style={{ fontWeight: 'bold', fontSize: '14px' }}>
                    {data.label}
                </div>
            </div>

            <div className="node-content" style={{ fontSize: '12px', color: 'var(--color-text-secondary)' }}>
                {data.description || 'Agent Node'}
            </div>

            <Handle type="source" position={Position.Right} className="glass-handle" />
        </GlassPanel>
    );
};

export default memo(GlassNode);
