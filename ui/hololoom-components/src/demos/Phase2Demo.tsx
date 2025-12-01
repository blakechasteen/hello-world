/**
 * Phase 2 Demo: Thinking Made Visible
 *
 * Integrated demonstration of all Phase 2 visualization components:
 * - Streaming Consciousness Timeline (9-stage weaving cycle)
 * - Knowledge Graph Animation (activation spreading)
 * - Confidence Landscape (decision terrain)
 *
 * @module demos/Phase2Demo
 */

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { StageTimeline } from '../components/timeline/StageTimeline';
import { KnowledgeGraph2D } from '../components/graph/KnowledgeGraph2D';
import { ConfidenceLandscape } from '../components/landscape/ConfidenceLandscape';
import { ConfidenceTable } from '../components/landscape/ConfidenceTable';
import type { StageTimingData, StageStatus } from '../store/slices/timelineSlice';
import type { GraphNode, GraphEdge, GraphViewport } from '../store/slices/graphSlice';

// ============================================================================
// Mock Data Generators
// ============================================================================

/**
 * Generate mock timeline stages for the 9-step weaving cycle
 */
function generateMockTimelineStages(progress: number): StageTimingData[] {
  const stages = [
    { id: 'loom_command', name: 'Loom Command', duration: 15 },
    { id: 'chrono_trigger', name: 'Chrono Trigger', duration: 25 },
    { id: 'yarn_graph', name: 'Yarn Graph', duration: 45 },
    { id: 'resonance_shed', name: 'Resonance Shed', duration: 85 },
    { id: 'dot_plasma', name: 'Dot Plasma', duration: 35 },
    { id: 'warp_space', name: 'Warp Space', duration: 65 },
    { id: 'convergence', name: 'Convergence', duration: 40 },
    { id: 'tool_execution', name: 'Tool Execution', duration: 95 },
    { id: 'spacetime', name: 'Spacetime', duration: 25 },
  ];

  const totalDuration = stages.reduce((sum, s) => sum + s.duration, 0);
  let elapsed = 0;

  return stages.map((stage) => {
    const startTime = elapsed;
    elapsed += stage.duration;
    const stageProgress = progress * totalDuration;

    let status: StageStatus = 'pending';
    let actualDuration: number | undefined;

    if (stageProgress >= startTime + stage.duration) {
      status = 'completed';
      actualDuration = stage.duration + Math.random() * 10 - 5;
    } else if (stageProgress >= startTime) {
      status = 'active';
      actualDuration = stageProgress - startTime;
    }

    return {
      id: stage.id,
      name: stage.name,
      status,
      startTime,
      expectedDuration: stage.duration,
      actualDuration,
      tokensProcessed: status === 'completed' ? Math.floor(Math.random() * 500 + 100) : undefined,
    };
  });
}

/**
 * Generate mock knowledge graph nodes
 */
function generateMockGraphNodes(activationProgress: number): GraphNode[] {
  const nodeTypes: Array<'query' | 'concept' | 'entity' | 'fact' | 'memory'> = [
    'query',
    'concept',
    'concept',
    'entity',
    'entity',
    'fact',
    'fact',
    'memory',
    'memory',
    'memory',
  ];

  const labels = [
    'Query: Thompson Sampling',
    'Bayesian Methods',
    'Exploration',
    'Multi-Armed Bandit',
    'Posterior Distribution',
    'Regret Minimization',
    'UCB Algorithm',
    'Decision Theory',
    'Probability',
    'Statistics',
  ];

  const pathNodes = new Set(['node-0', 'node-1', 'node-3', 'node-5']);

  return labels.map((label, i) => {
    const angle = (i / labels.length) * Math.PI * 2;
    const radius = i === 0 ? 0 : 150 + Math.random() * 100;

    // Calculate activation based on distance from query node
    const distance = i;
    const activationDelay = distance * 0.1;
    const activation = Math.max(
      0,
      Math.min(1, (activationProgress - activationDelay) * 2)
    );

    return {
      id: `node-${i}`,
      label,
      type: nodeTypes[i % nodeTypes.length],
      importance: 1 - i * 0.08,
      x: 400 + Math.cos(angle) * radius,
      y: 300 + Math.sin(angle) * radius,
      vx: 0,
      vy: 0,
      activation: i === 0 ? 1 : activation,
      targetActivation: i === 0 ? 1 : activation,
      isOnPath: pathNodes.has(`node-${i}`),
    };
  });
}

/**
 * Generate mock knowledge graph edges
 */
function generateMockGraphEdges(): GraphEdge[] {
  const edges: GraphEdge[] = [
    { id: 'e1', source: 'node-0', target: 'node-1', type: 'MENTIONS', weight: 0.9 },
    { id: 'e2', source: 'node-0', target: 'node-2', type: 'MENTIONS', weight: 0.8 },
    { id: 'e3', source: 'node-1', target: 'node-3', type: 'USES', weight: 0.85 },
    { id: 'e4', source: 'node-1', target: 'node-4', type: 'USES', weight: 0.75 },
    { id: 'e5', source: 'node-3', target: 'node-5', type: 'LEADS_TO', weight: 0.9 },
    { id: 'e6', source: 'node-2', target: 'node-6', type: 'IS_A', weight: 0.7 },
    { id: 'e7', source: 'node-5', target: 'node-7', type: 'PART_OF', weight: 0.8 },
    { id: 'e8', source: 'node-4', target: 'node-8', type: 'IS_A', weight: 0.65 },
    { id: 'e9', source: 'node-8', target: 'node-9', type: 'IS_A', weight: 0.6 },
  ];

  return edges;
}

/**
 * Generate mock confidence grid
 */
function generateMockConfidenceGrid(
  width: number,
  height: number,
  seed: number
): number[][] {
  const grid: number[][] = [];

  for (let row = 0; row < height; row++) {
    const rowData: number[] = [];
    for (let col = 0; col < width; col++) {
      // Create a gradient with some noise
      const baseValue =
        0.3 +
        0.4 * Math.sin((row / height) * Math.PI) *
        Math.cos((col / width) * Math.PI);

      // Add deterministic "noise" based on position
      const noise =
        0.15 *
        Math.sin(row * 2.5 + seed) *
        Math.cos(col * 1.7 + seed * 0.5);

      rowData.push(Math.max(0, Math.min(1, baseValue + noise)));
    }
    grid.push(rowData);
  }

  return grid;
}

// ============================================================================
// Main Demo Component
// ============================================================================

export const Phase2Demo: React.FC = () => {
  // State
  const [darkMode, setDarkMode] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showTable, setShowTable] = useState(false);

  // Graph viewport state
  const [viewport, setViewport] = useState<GraphViewport>({
    width: 800,
    height: 600,
    zoom: 1,
    panX: 0,
    panY: 0,
  });

  // Generate mock data
  const timelineStages = useMemo(
    () => generateMockTimelineStages(progress),
    [progress]
  );

  const graphNodes = useMemo(
    () => generateMockGraphNodes(progress),
    [progress]
  );

  const graphEdges = useMemo(() => generateMockGraphEdges(), []);

  const confidenceGrid = useMemo(
    () => generateMockConfidenceGrid(8, 6, progress * 10),
    [progress]
  );

  const xLabels = useMemo(
    () => ['Answer', 'Research', 'Verify', 'Explore', 'Clarify', 'Summarize', 'Compare', 'Analyze'],
    []
  );

  const yLabels = useMemo(
    () => ['Bayesian', 'Explore', 'UCB', 'Regret', 'Bandit', 'Stats'],
    []
  );

  // Animation loop
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 1) {
          setIsRunning(false);
          return 1;
        }
        return Math.min(1, p + 0.01);
      });
    }, 50);

    return () => clearInterval(interval);
  }, [isRunning]);

  // Handlers
  const handleStart = useCallback(() => {
    setProgress(0);
    setIsRunning(true);
  }, []);

  const handlePause = useCallback(() => {
    setIsRunning((r) => !r);
  }, []);

  const handleReset = useCallback(() => {
    setIsRunning(false);
    setProgress(0);
  }, []);

  const handleViewportChange = useCallback((update: Partial<GraphViewport>) => {
    setViewport((v) => ({ ...v, ...update }));
  }, []);

  // Calculate stats
  const activeStage = timelineStages.find((s) => s.status === 'active');
  const completedStages = timelineStages.filter((s) => s.status === 'completed').length;
  const totalTokens = timelineStages.reduce(
    (sum, s) => sum + (s.tokensProcessed ?? 0),
    0
  );

  // Base styles
  const bgColor = darkMode ? 'bg-gray-900' : 'bg-gray-50';
  const textColor = darkMode ? 'text-gray-100' : 'text-gray-900';
  const cardBg = darkMode ? 'bg-gray-800' : 'bg-white';
  const borderColor = darkMode ? 'border-gray-700' : 'border-gray-200';

  return (
    <div className={`min-h-screen ${bgColor} ${textColor} p-6`}>
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Phase 2: Thinking Made Visible</h1>
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Real-time visualization of HoloLoom's cognitive processes
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Dark mode toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`px-3 py-1.5 rounded-lg text-sm ${cardBg} border ${borderColor}`}
            >
              {darkMode ? '☀️ Light' : '🌙 Dark'}
            </button>

            {/* Control buttons */}
            <div className="flex gap-2">
              <button
                onClick={handleStart}
                disabled={isRunning}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium ${
                  isRunning
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                }`}
              >
                ▶ Start
              </button>
              <button
                onClick={handlePause}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium ${cardBg} border ${borderColor}`}
              >
                {isRunning ? '⏸ Pause' : '▶ Resume'}
              </button>
              <button
                onClick={handleReset}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium ${cardBg} border ${borderColor}`}
              >
                ↺ Reset
              </button>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className={`mt-4 h-2 rounded-full ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
          <div
            className="h-full bg-indigo-600 rounded-full transition-all duration-100"
            style={{ width: `${progress * 100}%` }}
          />
        </div>

        {/* Stats */}
        <div className="mt-2 flex gap-6 text-sm">
          <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
            Progress: <span className="font-mono">{(progress * 100).toFixed(0)}%</span>
          </span>
          <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
            Stage: <span className="font-medium">{activeStage?.name ?? 'Idle'}</span>
          </span>
          <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
            Completed: <span className="font-mono">{completedStages}/9</span>
          </span>
          <span className={darkMode ? 'text-gray-400' : 'text-gray-500'}>
            Tokens: <span className="font-mono">{totalTokens.toLocaleString()}</span>
          </span>
        </div>
      </div>

      {/* Main content grid */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Streaming Consciousness Timeline */}
        <div className={`${cardBg} rounded-xl border ${borderColor} p-4`}>
          <h2 className="text-lg font-semibold mb-3">
            Streaming Consciousness Timeline
          </h2>
          <StageTimeline
            stages={timelineStages}
            currentTokenRate={Math.random() * 100 + 50}
            totalTokens={totalTokens}
            darkMode={darkMode}
            showTokenRate={true}
            showProgress={true}
            className="h-[280px]"
          />
        </div>

        {/* Knowledge Graph Animation */}
        <div className={`${cardBg} rounded-xl border ${borderColor} p-4`}>
          <h2 className="text-lg font-semibold mb-3">
            Knowledge Graph Animation
          </h2>
          <KnowledgeGraph2D
            nodes={graphNodes}
            edges={graphEdges}
            viewport={viewport}
            queryNodeId="node-0"
            animated={true}
            darkMode={darkMode}
            showLabels={true}
            showEdgeTypes={false}
            onViewportChange={handleViewportChange}
            className="h-[280px]"
          />
        </div>

        {/* Confidence Landscape */}
        <div className={`${cardBg} rounded-xl border ${borderColor} p-4 lg:col-span-2`}>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">
              Confidence Landscape
            </h2>
            <button
              onClick={() => setShowTable(!showTable)}
              className={`px-3 py-1 rounded text-sm ${cardBg} border ${borderColor}`}
            >
              {showTable ? '📊 Show Heatmap' : '📋 Show Table'}
            </button>
          </div>

          {showTable ? (
            <ConfidenceTable
              grid={confidenceGrid}
              width={8}
              height={6}
              xLabels={xLabels}
              yLabels={yLabels}
              darkMode={darkMode}
              showPercentage={true}
              sortable={true}
              caption="Confidence values for tool-topic combinations"
              className="max-h-[350px]"
            />
          ) : (
            <ConfidenceLandscape
              grid={confidenceGrid}
              width={8}
              height={6}
              xLabels={xLabels}
              yLabels={yLabels}
              darkMode={darkMode}
              animated={true}
              showContourLines={true}
              showValues={true}
              containerWidth={800}
              containerHeight={350}
              className="mx-auto"
            />
          )}
        </div>
      </div>

      {/* Footer */}
      <div className={`max-w-7xl mx-auto mt-6 text-center text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
        <p>
          Phase 2 of the HoloLoom UX Roadmap: "Thinking Made Visible"
        </p>
        <p className="mt-1">
          Real-time visualization of the 9-step weaving cycle, knowledge activation,
          and decision confidence.
        </p>
      </div>
    </div>
  );
};

export default Phase2Demo;
