/**
 * CollaborationToolbar - Main toolbar for collaborative features
 *
 * Provides session management, voice controls, presence avatars,
 * and annotation tools in a compact, dockable toolbar.
 *
 * Phase 3: Collaborative Intelligence
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { UserPresence, CollaborativeSession } from '../types';

// ============================================================================
// Types
// ============================================================================

export interface CollaborationToolbarProps {
  session: CollaborativeSession | null;
  currentUser: UserPresence | null;
  users: UserPresence[];
  isVoiceConnected: boolean;
  isMuted: boolean;
  isPushToTalk: boolean;
  speakingUsers: string[];
  annotationMode: 'none' | 'note' | 'question' | 'insight';
  onCreateSession?: () => void;
  onJoinSession?: (sessionId: string) => void;
  onLeaveSession?: () => void;
  onToggleVoice?: () => void;
  onToggleMute?: () => void;
  onTogglePushToTalk?: () => void;
  onSetAnnotationMode?: (mode: 'none' | 'note' | 'question' | 'insight') => void;
  onInviteUser?: () => void;
  onOpenSettings?: () => void;
  position?: 'top' | 'bottom';
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getInitials(name: string): string {
  return name.split(' ').map(part => part[0]).join('').toUpperCase().slice(0, 2);
}

// ============================================================================
// Sub-components
// ============================================================================

const SessionControls: React.FC<{
  session: CollaborativeSession | null;
  onCreateSession?: () => void;
  onJoinSession?: (sessionId: string) => void;
  onLeaveSession?: () => void;
  onInviteUser?: () => void;
}> = ({ session, onCreateSession, onJoinSession, onLeaveSession, onInviteUser }) => {
  const [showJoinInput, setShowJoinInput] = useState(false);
  const [joinCode, setJoinCode] = useState('');

  const handleJoin = () => {
    if (joinCode.trim()) {
      onJoinSession?.(joinCode.trim());
      setShowJoinInput(false);
      setJoinCode('');
    }
  };

  if (!session) {
    return (
      <div className="session-controls">
        <button className="toolbar-btn primary" onClick={onCreateSession} title="Start session">
          <span className="icon">🚀</span>
          <span className="label">Start Session</span>
        </button>
        {showJoinInput ? (
          <div className="join-input-container">
            <input
              type="text"
              value={joinCode}
              onChange={(e) => setJoinCode(e.target.value)}
              placeholder="Session code"
              className="join-input"
              onKeyDown={(e) => e.key === 'Enter' && handleJoin()}
              autoFocus
            />
            <button className="toolbar-btn small" onClick={handleJoin}>Join</button>
            <button className="toolbar-btn small secondary" onClick={() => setShowJoinInput(false)}>✕</button>
          </div>
        ) : (
          <button className="toolbar-btn secondary" onClick={() => setShowJoinInput(true)} title="Join session">
            <span className="icon">🔗</span>
            <span className="label">Join</span>
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="session-controls active">
      <div className="session-info">
        <span className="session-indicator" />
        <span className="session-name">{session.name}</span>
        <span className="session-code">{session.sessionId.slice(0, 6)}</span>
      </div>
      <button className="toolbar-btn small" onClick={onInviteUser} title="Invite">
        <span className="icon">📨</span>
      </button>
      <button className="toolbar-btn small danger" onClick={onLeaveSession} title="Leave">
        <span className="icon">🚪</span>
      </button>
    </div>
  );
};

const VoiceControls: React.FC<{
  isConnected: boolean;
  isMuted: boolean;
  isPushToTalk: boolean;
  speakingUsers: string[];
  onToggleVoice?: () => void;
  onToggleMute?: () => void;
  onTogglePushToTalk?: () => void;
}> = ({ isConnected, isMuted, isPushToTalk, speakingUsers, onToggleVoice, onToggleMute, onTogglePushToTalk }) => (
  <div className={`voice-controls ${isConnected ? 'connected' : ''}`}>
    <button className={`toolbar-btn ${isConnected ? 'active' : ''}`} onClick={onToggleVoice} title={isConnected ? 'Disconnect voice' : 'Connect voice'}>
      <span className="icon">{isConnected ? '🎧' : '🔇'}</span>
    </button>
    {isConnected && (
      <>
        <button className={`toolbar-btn ${isMuted ? 'muted' : ''}`} onClick={onToggleMute} title={isMuted ? 'Unmute' : 'Mute'}>
          <span className="icon">{isMuted ? '🔇' : '🎤'}</span>
        </button>
        <button className={`toolbar-btn small ${isPushToTalk ? 'active' : ''}`} onClick={onTogglePushToTalk} title="Push-to-talk">
          <span className="icon">📢</span>
        </button>
        {speakingUsers.length > 0 && (
          <div className="speaking-indicator">
            <motion.div className="speaking-wave" animate={{ scale: [1, 1.2, 1] }} transition={{ repeat: Infinity, duration: 0.5 }} />
            <span>{speakingUsers.length} speaking</span>
          </div>
        )}
      </>
    )}
  </div>
);

const PresenceAvatars: React.FC<{
  users: UserPresence[];
  currentUser: UserPresence | null;
  maxVisible?: number;
}> = ({ users, currentUser, maxVisible = 5 }) => {
  const visibleUsers = users.slice(0, maxVisible);
  const overflow = users.length - maxVisible;

  return (
    <div className="presence-avatars">
      {currentUser && (
        <div className="avatar current" style={{ backgroundColor: currentUser.color }} title={`${currentUser.name} (you)`}>
          {currentUser.avatar ? <img src={currentUser.avatar} alt={currentUser.name} /> : <span>{getInitials(currentUser.name)}</span>}
        </div>
      )}
      <AnimatePresence>
        {visibleUsers.map((user, index) => (
          <motion.div
            key={user.userId}
            className={`avatar ${user.status}`}
            style={{ backgroundColor: user.color, zIndex: visibleUsers.length - index }}
            title={`${user.name} (${user.status})`}
            initial={{ scale: 0, x: -20 }}
            animate={{ scale: 1, x: 0 }}
            exit={{ scale: 0, x: -20 }}
            transition={{ delay: index * 0.05 }}
          >
            {user.avatar ? <img src={user.avatar} alt={user.name} /> : <span>{getInitials(user.name)}</span>}
            <span className={`status-dot ${user.status}`} />
          </motion.div>
        ))}
      </AnimatePresence>
      {overflow > 0 && <div className="avatar overflow" title={`+${overflow} more`}>+{overflow}</div>}
    </div>
  );
};

const AnnotationTools: React.FC<{
  mode: 'none' | 'note' | 'question' | 'insight';
  onSetMode?: (mode: 'none' | 'note' | 'question' | 'insight') => void;
}> = ({ mode, onSetMode }) => {
  const tools = [
    { id: 'note', icon: '📝', label: 'Add Note' },
    { id: 'question', icon: '❓', label: 'Ask Question' },
    { id: 'insight', icon: '💡', label: 'Share Insight' }
  ] as const;

  return (
    <div className="annotation-tools">
      {tools.map(tool => (
        <button
          key={tool.id}
          className={`toolbar-btn ${mode === tool.id ? 'active' : ''}`}
          onClick={() => onSetMode?.(mode === tool.id ? 'none' : tool.id)}
          title={tool.label}
        >
          <span className="icon">{tool.icon}</span>
        </button>
      ))}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const CollaborationToolbar: React.FC<CollaborationToolbarProps> = ({
  session, currentUser, users, isVoiceConnected, isMuted, isPushToTalk, speakingUsers, annotationMode,
  onCreateSession, onJoinSession, onLeaveSession, onToggleVoice, onToggleMute, onTogglePushToTalk,
  onSetAnnotationMode, onInviteUser, onOpenSettings, position = 'top', collapsed = false, onToggleCollapsed
}) => {
  return (
    <motion.div
      className={`collaboration-toolbar ${position} ${collapsed ? 'collapsed' : ''}`}
      initial={{ y: position === 'top' ? -60 : 60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', damping: 20 }}
    >
      <button className="collapse-toggle" onClick={onToggleCollapsed} title={collapsed ? 'Expand' : 'Collapse'}>
        <span className="icon">{collapsed ? (position === 'top' ? '▼' : '▲') : (position === 'top' ? '▲' : '▼')}</span>
      </button>

      <AnimatePresence>
        {!collapsed && (
          <motion.div className="toolbar-content" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
            <SessionControls session={session} onCreateSession={onCreateSession} onJoinSession={onJoinSession} onLeaveSession={onLeaveSession} onInviteUser={onInviteUser} />
            <div className="toolbar-divider" />
            <PresenceAvatars users={users} currentUser={currentUser} />
            {session && (
              <>
                <div className="toolbar-divider" />
                <VoiceControls isConnected={isVoiceConnected} isMuted={isMuted} isPushToTalk={isPushToTalk} speakingUsers={speakingUsers} onToggleVoice={onToggleVoice} onToggleMute={onToggleMute} onTogglePushToTalk={onTogglePushToTalk} />
                <div className="toolbar-divider" />
                <AnnotationTools mode={annotationMode} onSetMode={onSetAnnotationMode} />
              </>
            )}
            <div className="toolbar-spacer" />
            <button className="toolbar-btn" onClick={onOpenSettings} title="Settings"><span className="icon">⚙️</span></button>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default CollaborationToolbar;