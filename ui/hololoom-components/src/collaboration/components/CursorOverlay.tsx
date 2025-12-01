/**
 * CursorOverlay - Multi-user cursor visualization
 *
 * Shows other users' cursors with smooth interpolation,
 * name labels, and connection trails.
 *
 * Phase 3: Collaborative Intelligence
 */

import React, { useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { UserPresence, CursorPosition } from '../types';

// ============================================================================
// Types
// ============================================================================

export interface CursorOverlayProps {
  /** Other users' presence data */
  users: UserPresence[];
  /** Current viewport bounds */
  viewport?: {
    x: number;
    y: number;
    width: number;
    height: number;
    scale: number;
  };
  /** Show cursor trails */
  showTrails?: boolean;
  /** Show user names */
  showNames?: boolean;
  /** Cursor idle timeout (ms) */
  idleTimeout?: number;
  /** Custom cursor renderer */
  renderCursor?: (user: UserPresence) => React.ReactNode;
}

interface CursorTrailPoint {
  x: number;
  y: number;
  timestamp: number;
}

// ============================================================================
// Cursor Trail Hook
// ============================================================================

function useCursorTrails(
  users: UserPresence[],
  maxPoints: number = 10,
  maxAge: number = 500
) {
  const trailsRef = useRef<Map<string, CursorTrailPoint[]>>(new Map());

  useEffect(() => {
    const trails = trailsRef.current;
    const now = Date.now();

    // Update trails for each user
    users.forEach(user => {
      if (!user.cursor) return;

      let trail = trails.get(user.userId) || [];

      // Add new point
      trail.push({
        x: user.cursor.x,
        y: user.cursor.y,
        timestamp: now
      });

      // Remove old points
      trail = trail.filter(p => now - p.timestamp < maxAge);

      // Keep max points
      if (trail.length > maxPoints) {
        trail = trail.slice(-maxPoints);
      }

      trails.set(user.userId, trail);
    });

    // Clean up trails for users who left
    const userIds = new Set(users.map(u => u.userId));
    trails.forEach((_, id) => {
      if (!userIds.has(id)) {
        trails.delete(id);
      }
    });
  }, [users, maxPoints, maxAge]);

  return trailsRef.current;
}

// ============================================================================
// Cursor Component
// ============================================================================

interface UserCursorProps {
  user: UserPresence;
  showName: boolean;
  showTrail: boolean;
  trail: CursorTrailPoint[];
  isIdle: boolean;
}

const UserCursor: React.FC<UserCursorProps> = ({
  user,
  showName,
  showTrail,
  trail,
  isIdle
}) => {
  if (!user.cursor) return null;

  const { x, y } = user.cursor;

  return (
    <motion.g
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{
        opacity: isIdle ? 0.5 : 1,
        scale: 1,
        x,
        y
      }}
      exit={{ opacity: 0, scale: 0.5 }}
      transition={{
        type: 'spring',
        stiffness: 300,
        damping: 30,
        mass: 0.5
      }}
    >
      {/* Cursor trail */}
      {showTrail && trail.length > 1 && (
        <motion.path
          d={trailToPath(trail)}
          fill="none"
          stroke={user.color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeOpacity={0.4}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
        />
      )}

      {/* Cursor pointer */}
      <g transform="translate(-2, -2)">
        {/* Shadow */}
        <path
          d="M0 0 L0 18 L4.5 13.5 L8 20 L11 18.5 L7.5 12 L13 12 Z"
          fill="rgba(0,0,0,0.2)"
          transform="translate(1, 1)"
        />
        {/* Cursor body */}
        <path
          d="M0 0 L0 18 L4.5 13.5 L8 20 L11 18.5 L7.5 12 L13 12 Z"
          fill={user.color}
          stroke="white"
          strokeWidth={1.5}
        />
      </g>

      {/* User name label */}
      {showName && (
        <g transform="translate(16, 20)">
          {/* Label background */}
          <rect
            x={-4}
            y={-12}
            width={getTextWidth(user.name) + 12}
            height={18}
            rx={4}
            fill={user.color}
          />
          {/* Label text */}
          <text
            x={2}
            y={1}
            fill="white"
            fontSize={11}
            fontWeight={500}
            fontFamily="system-ui, -apple-system, sans-serif"
          >
            {user.name}
          </text>
        </g>
      )}

      {/* Focus indicator - pulsing ring when focused on something */}
      {user.focus && user.focus.type === 'node' && (
        <motion.circle
          r={8}
          fill="none"
          stroke={user.color}
          strokeWidth={2}
          animate={{
            r: [8, 12, 8],
            opacity: [1, 0.5, 1]
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        />
      )}
    </motion.g>
  );
};

// ============================================================================
// Helper Functions
// ============================================================================

function trailToPath(trail: CursorTrailPoint[]): string {
  if (trail.length < 2) return '';

  const points = trail.map(p => `${p.x},${p.y}`);
  return `M${points.join(' L')}`;
}

function getTextWidth(text: string): number {
  // Approximate text width (8px per character at 11px font)
  return text.length * 7;
}

// ============================================================================
// Main Component
// ============================================================================

export const CursorOverlay: React.FC<CursorOverlayProps> = ({
  users,
  viewport,
  showTrails = true,
  showNames = true,
  idleTimeout = 30000,
  renderCursor
}) => {
  const trails = useCursorTrails(users);
  const now = Date.now();

  // Filter to users with valid cursors
  const visibleUsers = useMemo(() => {
    return users.filter(user => {
      if (!user.cursor) return false;
      if (user.status === 'offline') return false;
      return true;
    });
  }, [users]);

  // Check if cursor is within viewport
  const isInViewport = (cursor: CursorPosition): boolean => {
    if (!viewport) return true;
    const { x, y, width, height } = viewport;
    return (
      cursor.x >= x &&
      cursor.x <= x + width &&
      cursor.y >= y &&
      cursor.y <= y + height
    );
  };

  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        overflow: 'visible',
        zIndex: 1000
      }}
    >
      <AnimatePresence>
        {visibleUsers.map(user => {
          if (!user.cursor) return null;

          // Skip if outside viewport
          if (!isInViewport(user.cursor)) return null;

          const isIdle = now - user.lastActive > idleTimeout;
          const trail = trails.get(user.userId) || [];

          // Use custom renderer if provided
          if (renderCursor) {
            return (
              <g key={user.userId}>
                {renderCursor(user)}
              </g>
            );
          }

          return (
            <UserCursor
              key={user.userId}
              user={user}
              showName={showNames}
              showTrail={showTrails}
              trail={trail}
              isIdle={isIdle}
            />
          );
        })}
      </AnimatePresence>
    </svg>
  );
};

// ============================================================================
// Minimap Cursor Indicator
// ============================================================================

export interface MinimapCursorsProps {
  users: UserPresence[];
  minimapBounds: { width: number; height: number };
  worldBounds: { width: number; height: number };
}

export const MinimapCursors: React.FC<MinimapCursorsProps> = ({
  users,
  minimapBounds,
  worldBounds
}) => {
  const scaleX = minimapBounds.width / worldBounds.width;
  const scaleY = minimapBounds.height / worldBounds.height;

  return (
    <svg
      width={minimapBounds.width}
      height={minimapBounds.height}
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
    >
      {users.map(user => {
        if (!user.cursor) return null;

        const x = user.cursor.x * scaleX;
        const y = user.cursor.y * scaleY;

        return (
          <motion.circle
            key={user.userId}
            cx={x}
            cy={y}
            r={3}
            fill={user.color}
            stroke="white"
            strokeWidth={1}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
          />
        );
      })}
    </svg>
  );
};

// ============================================================================
// Cursor Prediction (for smoother remote cursors)
// ============================================================================

export interface PredictedCursor {
  x: number;
  y: number;
  velocityX: number;
  velocityY: number;
  predictedX: number;
  predictedY: number;
}

export function predictCursorPosition(
  current: CursorPosition,
  previous: CursorPosition | null,
  deltaTime: number,
  predictionMs: number = 50
): PredictedCursor {
  if (!previous) {
    return {
      x: current.x,
      y: current.y,
      velocityX: 0,
      velocityY: 0,
      predictedX: current.x,
      predictedY: current.y
    };
  }

  const velocityX = (current.x - previous.x) / deltaTime;
  const velocityY = (current.y - previous.y) / deltaTime;

  return {
    x: current.x,
    y: current.y,
    velocityX,
    velocityY,
    predictedX: current.x + velocityX * predictionMs,
    predictedY: current.y + velocityY * predictionMs
  };
}

export default CursorOverlay;