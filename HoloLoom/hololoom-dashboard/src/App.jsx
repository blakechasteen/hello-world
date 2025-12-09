import React, { useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { GlassPanel, GlassButton } from './components/ui/Glass';
import WorkflowCanvas from './components/workflow/WorkflowCanvas';
import PropertiesPanel from './components/workflow/PropertiesPanel';
import AnalyticsDashboard from './components/analytics/AnalyticsDashboard';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

function App() {
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [view, setView] = useState('builder'); // 'builder' or 'monitor'

  // Initialize WebSocket connection
  const { isConnected, data } = useWebSocket('ws://localhost:8000/ws');

  const onDragStart = (event, nodeType, label, icon) => {
    event.dataTransfer.setData('application/reactflow', 'glass');
    event.dataTransfer.setData('application/nodeType', nodeType);
    event.dataTransfer.setData('application/label', label);
    event.dataTransfer.setData('application/icon', icon);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <ReactFlowProvider>
      <div className="app-container">
        {/* Header */}
        <GlassPanel intensity="high" className="app-header">
          <div className="logo-area">
            <div className="logo-icon">H</div>
            <h1 className="logo-text">HoloLoom <span className="text-gradient">Orchestrator</span></h1>
          </div>
          <div className="header-controls">
            {/* View Switcher */}
            <div className="view-switcher" style={{ display: 'flex', gap: '8px', marginRight: '16px' }}>
              <GlassButton
                variant={view === 'builder' ? 'primary' : 'secondary'}
                onClick={() => setView('builder')}
                style={{ fontSize: '0.9rem', padding: '0.4rem 0.8rem' }}
              >
                Builder
              </GlassButton>
              <GlassButton
                variant={view === 'monitor' ? 'primary' : 'secondary'}
                onClick={() => setView('monitor')}
                style={{ fontSize: '0.9rem', padding: '0.4rem 0.8rem' }}
              >
                Monitor
              </GlassButton>
            </div>

            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: isConnected ? '#10b981' : '#ef4444',
                marginRight: '1rem',
                boxShadow: isConnected ? '0 0 10px #10b981' : 'none',
                transition: 'all 0.3s'
              }}
              title={isConnected ? 'Connected to Server' : 'Disconnected'}
            />
            <div className="user-avatar">B</div>
          </div>
        </GlassPanel>

        <div className="main-content">
          {view === 'builder' ? (
            <>
              {/* Left Sidebar: Agent Palette */}
              <GlassPanel intensity="medium" className="sidebar-left">
                <div className="sidebar-header">
                  <h3>Agent Palette</h3>
                </div>
                <div className="agent-list">
                  <div className="agent-category">
                    <h4>Query Agents</h4>
                    <GlassPanel
                      intensity="low"
                      interactive
                      className="agent-item"
                      draggable
                      onDragStart={(event) => onDragStart(event, 'query', 'HoloLoom Query', 'Q')}
                    >
                      <span className="agent-icon query">Q</span>
                      <span>HoloLoom Query</span>
                    </GlassPanel>
                    <GlassPanel
                      intensity="low"
                      interactive
                      className="agent-item"
                      draggable
                      onDragStart={(event) => onDragStart(event, 'query', 'Memory Search', 'S')}
                    >
                      <span className="agent-icon query">S</span>
                      <span>Memory Search</span>
                    </GlassPanel>
                  </div>

                  <div className="agent-category">
                    <h4>Process Agents</h4>
                    <GlassPanel
                      intensity="low"
                      interactive
                      className="agent-item"
                      draggable
                      onDragStart={(event) => onDragStart(event, 'process', 'Embedder', 'E')}
                    >
                      <span className="agent-icon process">E</span>
                      <span>Embedder</span>
                    </GlassPanel>
                    <GlassPanel
                      intensity="low"
                      interactive
                      className="agent-item"
                      draggable
                      onDragStart={(event) => onDragStart(event, 'process', 'Synthesizer', 'S')}
                    >
                      <span className="agent-icon process">S</span>
                      <span>Synthesizer</span>
                    </GlassPanel>
                  </div>

                  <div className="agent-category">
                    <h4>Memory Agents</h4>
                    <GlassPanel
                      intensity="low"
                      interactive
                      className="agent-item"
                      draggable
                      onDragStart={(event) => onDragStart(event, 'memory', 'Memory Store', 'M')}
                    >
                      <span className="agent-icon memory">M</span>
                      <span>Memory Store</span>
                    </GlassPanel>
                  </div>
                </div>
              </GlassPanel>

              {/* Center: Canvas */}
              <main className="canvas-area">
                <WorkflowCanvas onNodeSelect={setSelectedNodeId} />
              </main>

              {/* Right Sidebar: Properties */}
              <GlassPanel intensity="medium" className="sidebar-right">
                <div className="sidebar-header">
                  <h3>Properties</h3>
                </div>
                <div className="properties-content">
                  <PropertiesPanel selectedNodeId={selectedNodeId} />
                </div>
              </GlassPanel>
            </>
          ) : (
            <main className="canvas-area" style={{ overflow: 'hidden' }}>
              <AnalyticsDashboard data={data} />
            </main>
          )}
        </div>
      </div>
    </ReactFlowProvider>
  );
}

export default App;
