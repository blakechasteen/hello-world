import React, { useState } from 'react';
import { ReactFlowProvider } from 'reactflow';
import { GlassPanel, GlassButton } from './components/ui/Glass';
import WorkflowCanvas from './components/workflow/WorkflowCanvas';
import PropertiesPanel from './components/workflow/PropertiesPanel';
import './App.css';

function App() {
  const [selectedNodeId, setSelectedNodeId] = useState(null);

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
            <GlassButton variant="primary">Run Workflow</GlassButton>
            <div className="user-avatar">B</div>
          </div>
        </GlassPanel>

        <div className="main-content">
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
        </div>
      </div>
    </ReactFlowProvider>
  );
}

export default App;
