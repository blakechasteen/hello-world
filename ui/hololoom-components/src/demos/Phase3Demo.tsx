/**
 * Phase 3 Demo - Collaborative Intelligence
 *
 * Showcases:
 * - Shared Memory Palace (multi-user graph navigation)
 * - Cursor Presence (see teammates' cursors)
 * - Collaborative Annotations (notes, reactions)
 * - Real-time CRDT sync
 * - Voice channel integration
 *
 * Phase 3: Collaborative Intelligence
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CursorOverlay, CollaborationToolbar } from '../collaboration/components';
import type {
  UserPresence,
  CollaborativeSession,
  Annotation,
  Reaction,
  CursorPosition,
} from '../collaboration/types';

// ============================================================================
// Mock Data for Demo
// ============================================================================

const MOCK_USERS: UserPresence[] = [
  {
    userId: 'user-1',
    name: 'Alice Chen',
    color: '#3b82f6',
    status: 'online',
    lastActive: Date.now(),
    cursor: { x: 300, y: 200, timestamp: Date.now() },
    focus: { type: 'node', targetId: 'node-thompson' },
  },
  {
    userId: 'user-2',
    name: 'Bob Smith',
    color: '#22c55e',
    status: 'online',
    lastActive: Date.now() - 5000,
    cursor: { x: 500, y: 350, timestamp: Date.now() },
    focus: { type: 'node', targetId: 'node-bayesian' },
  },
  {
    userId: 'user-3',
    name: 'Carol Davis',
    color: '#f59e0b',
    status: 'idle',
    lastActive: Date.now() - 60000,
    cursor: { x: 200, y: 400, timestamp: Date.now() },
  },
];

const MOCK_SESSION: CollaborativeSession = {
  sessionId: 'sess-abc123',
  name: 'Research Session',
  description: 'Exploring Thompson Sampling concepts',
  hostId: 'current-user',
  participants: ['current-user', 'user-1', 'user-2', 'user-3'],
  mode: 'edit',
  createdAt: Date.now() - 3600000,
  settings: {
    allowVoice: true,
    allowAnnotations: true,
    allowEditing: true,
    maxParticipants: 10,
    isPublic: false,
  },
};

const MOCK_ANNOTATIONS: Annotation[] = [
  {
    id: 'ann-1',
    type: 'insight',
    targetType: 'node',
    targetId: 'node-thompson',
    content: 'This is the key concept that ties exploration and exploitation together!',
    authorId: 'user-1',
    authorName: 'Alice Chen',
    createdAt: Date.now() - 1800000,
    updatedAt: Date.now() - 1800000,
    status: 'open',
    replies: [
      {
        id: 'reply-1',
        content: 'Agreed! Should we add more examples?',
        authorId: 'user-2',
        authorName: 'Bob Smith',
        createdAt: Date.now() - 1200000,
      },
    ],
    position: { x: 320, y: 180 },
  },
  {
    id: 'ann-2',
    type: 'question',
    targetType: 'node',
    targetId: 'node-bayesian',
    content: 'How does this relate to UCB?',
    authorId: 'user-2',
    authorName: 'Bob Smith',
    createdAt: Date.now() - 900000,
    updatedAt: Date.now() - 900000,
    status: 'open',
    replies: [],
    position: { x: 520, y: 330 },
  },
];

const MOCK_GRAPH_NODES = [
  { id: 'node-thompson', label: 'Thompson Sampling', x: 300, y: 200, type: 'concept' },
  { id: 'node-bayesian', label: 'Bayesian Methods', x: 500, y: 350, type: 'concept' },
  { id: 'node-explore', label: 'Exploration', x: 150, y: 300, type: 'strategy' },
  { id: 'node-exploit', label: 'Exploitation', x: 450, y: 150, type: 'strategy' },
  { id: 'node-bandit', label: 'Multi-Armed Bandit', x: 350, y: 450, type: 'problem' },
];

const MOCK_GRAPH_EDGES = [
  { source: 'node-thompson', target: 'node-bayesian', type: 'USES' },
  { source: 'node-thompson', target: 'node-explore', type: 'BALANCES' },
  { source: 'node-thompson', target: 'node-exploit', type: 'BALANCES' },
  { source: 'node-thompson', target: 'node-bandit', type: 'SOLVES' },
  { source: 'node-bayesian', target: 'node-bandit', type: 'APPLIES_TO' },
];

// ============================================================================
// Demo Component
// ============================================================================

export const Phase3Demo: React.FC = () => {
  // State
  const [session, setSession] = useState<CollaborativeSession | null>(null);
  const [users, setUsers] = useState<UserPresence[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isVoiceConnected, setIsVoiceConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isPushToTalk, setIsPushToTalk] = useState(false);
  const [speakingUsers, setSpeakingUsers] = useState<string[]>([]);
  const [annotationMode, setAnnotationMode] = useState<'none' | 'note' | 'question' | 'insight'>('none');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [toolbarCollapsed, setToolbarCollapsed] = useState(false);
  const [showAnnotations, setShowAnnotations] = useState(true);

  const currentUser: UserPresence = useMemo(() => ({
    userId: 'current-user',
    name: 'You',
    color: '#8b5cf6',
    status: 'online',
    lastActive: Date.now(),
  }), []);

  // Simulate cursor movement for demo
  useEffect(() => {
    if (!session) return;

    const interval = setInterval(() => {
      setUsers(prev => prev.map(user => ({
        ...user,
        cursor: user.cursor ? {
          ...user.cursor,
          x: user.cursor.x + (Math.random() - 0.5) * 20,
          y: user.cursor.y + (Math.random() - 0.5) * 20,
          timestamp: Date.now(),
        } : undefined,
        lastActive: user.status === 'online' ? Date.now() : user.lastActive,
      })));
    }, 100);

    return () => clearInterval(interval);
  }, [session]);

  // Simulate speaking indicators
  useEffect(() => {
    if (!isVoiceConnected) {
      setSpeakingUsers([]);
      return;
    }

    const interval = setInterval(() => {
      setSpeakingUsers(prev => {
        if (Math.random() > 0.7) {
          const randomUser = users[Math.floor(Math.random() * users.length)];
          if (randomUser && !prev.includes(randomUser.userId)) {
            return [...prev, randomUser.userId];
          }
        }
        if (prev.length > 0 && Math.random() > 0.5) {
          return prev.slice(1);
        }
        return prev;
      });
    }, 500);

    return () => clearInterval(interval);
  }, [isVoiceConnected, users]);

  // Handlers
  const handleCreateSession = useCallback(() => {
    setSession(MOCK_SESSION);
    setUsers(MOCK_USERS);
    setAnnotations(MOCK_ANNOTATIONS);
  }, []);

  const handleJoinSession = useCallback((sessionId: string) => {
    console.log('Joining session:', sessionId);
    setSession({ ...MOCK_SESSION, sessionId });
    setUsers(MOCK_USERS);
    setAnnotations(MOCK_ANNOTATIONS);
  }, []);

  const handleLeaveSession = useCallback(() => {
    setSession(null);
    setUsers([]);
    setAnnotations([]);
    setIsVoiceConnected(false);
  }, []);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(nodeId === selectedNode ? null : nodeId);
  }, [selectedNode]);

  const handleAddAnnotation = useCallback((nodeId: string) => {
    if (annotationMode === 'none') return;

    const newAnnotation: Annotation = {
      id: `ann-${Date.now()}`,
      type: annotationMode,
      targetType: 'node',
      targetId: nodeId,
      content: `New ${annotationMode} on ${nodeId}`,
      authorId: currentUser.userId,
      authorName: currentUser.name,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'open',
      replies: [],
    };

    setAnnotations(prev => [...prev, newAnnotation]);
    setAnnotationMode('none');
  }, [annotationMode, currentUser]);

  // Render
  return (
    <div className="phase3-demo">
      {/* Header */}
      <div className="demo-header">
        <h1>Phase 3: Collaborative Intelligence</h1>
        <p>Real-time collaboration with shared memory palace</p>
      </div>

      {/* Collaboration Toolbar */}
      <CollaborationToolbar
        session={session}
        currentUser={currentUser}
        users={users}
        isVoiceConnected={isVoiceConnected}
        isMuted={isMuted}
        isPushToTalk={isPushToTalk}
        speakingUsers={speakingUsers}
        annotationMode={annotationMode}
        onCreateSession={handleCreateSession}
        onJoinSession={handleJoinSession}
        onLeaveSession={handleLeaveSession}
        onToggleVoice={() => setIsVoiceConnected(!isVoiceConnected)}
        onToggleMute={() => setIsMuted(!isMuted)}
        onTogglePushToTalk={() => setIsPushToTalk(!isPushToTalk)}
        onSetAnnotationMode={setAnnotationMode}
        onInviteUser={() => alert('Invite dialog would open here')}
        onOpenSettings={() => alert('Settings dialog would open here')}
        position="top"
        collapsed={toolbarCollapsed}
        onToggleCollapsed={() => setToolbarCollapsed(!toolbarCollapsed)}
      />

      {/* Main Canvas Area */}
      <div className="canvas-container">
        {/* Graph Visualization */}
        <svg className="graph-canvas" viewBox="0 0 700 600">
          {/* Edges */}
          {MOCK_GRAPH_EDGES.map((edge, i) => {
            const source = MOCK_GRAPH_NODES.find(n => n.id === edge.source);
            const target = MOCK_GRAPH_NODES.find(n => n.id === edge.target);
            if (!source || !target) return null;

            return (
              <g key={i}>
                <line
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke="rgba(255,255,255,0.2)"
                  strokeWidth={2}
                />
                <text
                  x={(source.x + target.x) / 2}
                  y={(source.y + target.y) / 2 - 5}
                  fill="rgba(255,255,255,0.4)"
                  fontSize={10}
                  textAnchor="middle"
                >
                  {edge.type}
                </text>
              </g>
            );
          })}

          {/* Nodes */}
          {MOCK_GRAPH_NODES.map(node => {
            const isSelected = selectedNode === node.id;
            const focusedBy = users.filter(u => u.focus?.targetId === node.id);
            const nodeAnnotations = annotations.filter(a => a.targetId === node.id);

            return (
              <g
                key={node.id}
                transform={`translate(${node.x}, ${node.y})`}
                onClick={() => handleNodeClick(node.id)}
                style={{ cursor: 'pointer' }}
              >
                {/* User focus rings */}
                {focusedBy.map((user, i) => (
                  <motion.circle
                    key={user.userId}
                    r={35 + i * 5}
                    fill="none"
                    stroke={user.color}
                    strokeWidth={2}
                    strokeDasharray="5,5"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
                  />
                ))}

                {/* Node circle */}
                <motion.circle
                  r={isSelected ? 32 : 28}
                  fill={isSelected ? 'rgba(139, 92, 246, 0.3)' : 'rgba(59, 130, 246, 0.2)'}
                  stroke={isSelected ? '#8b5cf6' : '#3b82f6'}
                  strokeWidth={isSelected ? 3 : 2}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                />

                {/* Node label */}
                <text
                  y={4}
                  fill="white"
                  fontSize={11}
                  fontWeight={500}
                  textAnchor="middle"
                >
                  {node.label}
                </text>

                {/* Annotation indicator */}
                {nodeAnnotations.length > 0 && showAnnotations && (
                  <g transform="translate(20, -20)">
                    <circle r={10} fill="#f59e0b" />
                    <text y={4} fill="white" fontSize={10} fontWeight={600} textAnchor="middle">
                      {nodeAnnotations.length}
                    </text>
                  </g>
                )}

                {/* Add annotation button when in annotation mode */}
                {annotationMode !== 'none' && (
                  <g
                    transform="translate(-20, -20)"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAddAnnotation(node.id);
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <circle r={12} fill="#22c55e" />
                    <text y={4} fill="white" fontSize={14} textAnchor="middle">+</text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>

        {/* Cursor Overlay */}
        {session && (
          <CursorOverlay
            users={users}
            showTrails={true}
            showNames={true}
          />
        )}

        {/* Annotations Panel */}
        {showAnnotations && annotations.length > 0 && (
          <div className="annotations-panel">
            <div className="panel-header">
              <h3>Annotations ({annotations.length})</h3>
              <button onClick={() => setShowAnnotations(false)}>×</button>
            </div>
            <div className="annotations-list">
              {annotations.map(ann => (
                <div key={ann.id} className={`annotation-card ${ann.type}`}>
                  <div className="annotation-header">
                    <span className="annotation-type">
                      {ann.type === 'insight' ? '💡' : ann.type === 'question' ? '❓' : '📝'}
                    </span>
                    <span className="annotation-author">{ann.authorName}</span>
                  </div>
                  <p className="annotation-content">{ann.content}</p>
                  {ann.replies.length > 0 && (
                    <div className="annotation-replies">
                      {ann.replies.map(reply => (
                        <div key={reply.id} className="reply">
                          <strong>{reply.authorName}:</strong> {reply.content}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Status Bar */}
      <div className="status-bar">
        <div className="status-item">
          <span className={`status-dot ${session ? 'connected' : 'disconnected'}`} />
          <span>{session ? 'Connected' : 'Not in session'}</span>
        </div>
        {session && (
          <>
            <div className="status-item">
              <span>👥 {users.length + 1} participants</span>
            </div>
            <div className="status-item">
              <span>📝 {annotations.length} annotations</span>
            </div>
            {isVoiceConnected && (
              <div className="status-item">
                <span>🎧 Voice active</span>
              </div>
            )}
          </>
        )}
      </div>

      <style>{`
        .phase3-demo {
          width: 100%;
          min-height: 100vh;
          background: linear-gradient(135deg, #0f0f1e 0%, #1a1a3e 50%, #0f0f1e 100%);
          color: white;
          font-family: system-ui, -apple-system, sans-serif;
          position: relative;
          overflow: hidden;
        }

        .demo-header {
          text-align: center;
          padding: 80px 20px 20px;
        }

        .demo-header h1 {
          font-size: 2rem;
          background: linear-gradient(135deg, #3b82f6, #8b5cf6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          margin-bottom: 8px;
        }

        .demo-header p {
          color: rgba(255, 255, 255, 0.6);
        }

        .canvas-container {
          position: relative;
          width: 100%;
          height: calc(100vh - 200px);
          display: flex;
          justify-content: center;
          align-items: center;
        }

        .graph-canvas {
          width: 100%;
          max-width: 800px;
          height: auto;
        }

        .annotations-panel {
          position: absolute;
          right: 20px;
          top: 20px;
          width: 300px;
          max-height: 400px;
          background: rgba(26, 26, 46, 0.95);
          backdrop-filter: blur(10px);
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          overflow: hidden;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .panel-header h3 {
          margin: 0;
          font-size: 14px;
        }

        .panel-header button {
          background: none;
          border: none;
          color: rgba(255, 255, 255, 0.6);
          cursor: pointer;
          font-size: 18px;
        }

        .annotations-list {
          padding: 12px;
          max-height: 340px;
          overflow-y: auto;
        }

        .annotation-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          padding: 12px;
          margin-bottom: 8px;
        }

        .annotation-card.insight {
          border-left: 3px solid #f59e0b;
        }

        .annotation-card.question {
          border-left: 3px solid #3b82f6;
        }

        .annotation-card.note {
          border-left: 3px solid #22c55e;
        }

        .annotation-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .annotation-type {
          font-size: 16px;
        }

        .annotation-author {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.6);
        }

        .annotation-content {
          font-size: 13px;
          line-height: 1.4;
          margin: 0;
        }

        .annotation-replies {
          margin-top: 8px;
          padding-top: 8px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .reply {
          font-size: 12px;
          color: rgba(255, 255, 255, 0.7);
          margin-bottom: 4px;
        }

        .status-bar {
          position: fixed;
          bottom: 0;
          left: 0;
          right: 0;
          background: rgba(26, 26, 46, 0.95);
          backdrop-filter: blur(10px);
          padding: 12px 20px;
          display: flex;
          gap: 24px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        .status-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 13px;
          color: rgba(255, 255, 255, 0.7);
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }

        .status-dot.connected {
          background: #22c55e;
        }

        .status-dot.disconnected {
          background: #6b7280;
        }
      `}</style>
    </div>
  );
};

export default Phase3Demo;
