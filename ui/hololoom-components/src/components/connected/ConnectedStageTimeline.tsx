/**
 * ConnectedStageTimeline
 *
 * Connected wrapper that wires StageTimeline to the Zustand store.
 * Handles store subscriptions and forwards state to the presentation component.
 *
 * Part of HoloLoom Phase 1 Frontend (November 2025)
 */

import React, { useCallback, useMemo } from 'react';
import { StageTimeline } from '../timeline/StageTimeline';
import { useHoloLoomStore } from '../../store';
import type { StageData } from '../timeline/StageTimeline';
import type { WeavingStage } from '../../types';

// ============================================================================
// Stage Display Names
// ============================================================================

const STAGE_DISPLAY_NAMES: Record<WeavingStage, string> = {
  loom_command: 'Pattern Selection',
  chrono_trigger: 'Temporal Window',
  yarn_graph: 'Memory Retrieval',
  resonance_shed: 'Feature Extraction',
  dot_plasma: 'Feature Fusion',
  warp_space: 'Tensor Operations',
  convergence: 'Decision Making',
  spacetime: 'Response Weaving',
  reflection: 'Learning',
};

// ============================================================================
// Props
// ============================================================================

export interface ConnectedStageTimelineProps {
  /** Whether to animate transitions */
  animated?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when a stage is clicked */
  onStageClick?: (stageId: WeavingStage) => void;
  /** Callback when a stage is hovered */
  onStageHover?: (stageId: WeavingStage | null) => void;
}

// ============================================================================
// Component
// ============================================================================

export const ConnectedStageTimeline: React.FC<ConnectedStageTimelineProps> = ({
  animated = true,
  className,
  onStageClick,
  onStageHover,
}) => {
  // Select state from store
  const stages = useHoloLoomStore(state => state.stages);
  const activeStage = useHoloLoomStore(state => state.activeStage);
  const streamState = useHoloLoomStore(state => state.streamState);
  const tokensGenerated = useHoloLoomStore(state => state.tokensGenerated);
  const startTime = useHoloLoomStore(state => state.startTime);

  // Calculate token rate (tokens per second)
  const tokenRate = useMemo(() => {
    if (!startTime || tokensGenerated === 0) return 0;
    const elapsedSeconds = (Date.now() - startTime) / 1000;
    if (elapsedSeconds < 0.1) return 0;
    return Math.round(tokensGenerated / elapsedSeconds);
  }, [startTime, tokensGenerated]);

  // Keep last 50 token rate samples for history
  const tokenRateHistoryRef = React.useRef<number[]>([]);
  React.useEffect(() => {
    if (streamState === 'streaming' && tokenRate > 0) {
      tokenRateHistoryRef.current = [...tokenRateHistoryRef.current.slice(-49), tokenRate];
    } else if (streamState === 'idle') {
      tokenRateHistoryRef.current = [];
    }
  }, [tokenRate, streamState]);

  // Transform store stages to StageData format
  const stageData = useMemo((): StageData[] => {
    return stages.map(stage => ({
      id: stage.stage,
      label: STAGE_DISPLAY_NAMES[stage.stage] || stage.stage,
      status: stage.status,
      duration: stage.duration_ms,
      metrics: stage.metrics
        ? {
            memoryNodes: stage.metrics.memories_retrieved,
            confidence: stage.metrics.confidence,
            tokensProcessed: stage.metrics.tokens_processed,
          }
        : undefined,
    }));
  }, [stages]);

  // Handle stage click
  const handleStageClick = useCallback(
    (stage: StageData) => {
      onStageClick?.(stage.id as WeavingStage);
    },
    [onStageClick]
  );

  // Handle stage hover
  const handleStageHover = useCallback(
    (stage: StageData | null) => {
      onStageHover?.(stage?.id as WeavingStage | null);
    },
    [onStageHover]
  );

  return (
    <StageTimeline
      stages={stageData}
      activeStage={activeStage}
      tokenRate={tokenRate}
      tokenRateHistory={tokenRateHistoryRef.current}
      animated={animated}
      className={className}
      onStageClick={handleStageClick}
      onStageHover={handleStageHover}
    />
  );
};

ConnectedStageTimeline.displayName = 'ConnectedStageTimeline';

export default ConnectedStageTimeline;
