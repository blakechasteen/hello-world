'use client';

import { useEffect } from 'react';
import { useProgressSubscription } from '@hololoom/api-client';
import type { ProgressEvent } from '@hololoom/api-client';
import type { WeavingAction } from '@/contexts/WeavingContext';

/**
 * Bridge between the backend's WebSocket progress events and the WeavingContext reducer.
 *
 * Maps ProgressEvent → WeavingAction so that the consciousness shell's three panes
 * react to live weaving progress: Memory Palace highlights activated nodes,
 * Awareness shows stage waterfall, Active Thinking updates confidence badge.
 *
 * Graceful degradation: if WebSocket fails to connect, chat still works via POST.
 * The consciousness shell just doesn't show live progress.
 */
export function useWeavingStream(
  dispatch: React.Dispatch<WeavingAction>,
  jobId: string | null
) {
  useProgressSubscription(jobId, (event: ProgressEvent) => {
    switch (event.status) {
      case 'running':
        dispatch({
          type: 'STAGE_UPDATE',
          stage: event.stage,
          stageName: event.stageName,
          progress: event.progress,
        });
        break;

      case 'complete':
        dispatch({
          type: 'STAGE_COMPLETE',
          stage: event.stage,
          durationMs: event.elapsedMs,
        });
        break;

      case 'error':
        // On error, complete with what we have — don't crash the UI
        dispatch({
          type: 'QUERY_COMPLETE',
          confidence: 0,
          toolUsed: 'error',
          latencyMs: event.elapsedMs,
          memoryIds: [],
        });
        break;

      // 'pending' — no action needed, stage hasn't started yet
    }
  });
}
