/**
 * Force-Directed Graph Layout Web Worker
 *
 * Runs force simulation in a separate thread to keep UI responsive.
 * Implements Fruchterman-Reingold algorithm with improvements.
 *
 * @module workers/forceLayout.worker
 */

// ============================================================================
// Types
// ============================================================================

interface WorkerNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  importance: number;
  fixed?: boolean; // If true, don't move this node
}

interface WorkerEdge {
  source: string;
  target: string;
  weight: number;
}

interface LayoutConfig {
  /** Force strength for node repulsion */
  repulsionStrength: number;
  /** Spring strength for edge attraction */
  attractionStrength: number;
  /** Damping factor (0.0-1.0) - reduces velocity each step */
  damping: number;
  /** Center gravity strength - pulls nodes toward center */
  centerGravity: number;
  /** Maximum iterations */
  maxIterations: number;
  /** Convergence threshold (stop if movement < this) */
  convergenceThreshold: number;
  /** Viewport dimensions */
  width: number;
  height: number;
  /** Minimum distance between nodes */
  minDistance: number;
  /** Barnes-Hut theta (0 = exact, 1 = fast approximation) */
  theta: number;
}

interface InitMessage {
  type: 'init';
  nodes: WorkerNode[];
  edges: WorkerEdge[];
  config: LayoutConfig;
}

interface StepMessage {
  type: 'step';
  iterations?: number; // How many iterations to run (default: 1)
}

interface StopMessage {
  type: 'stop';
}

interface UpdateConfigMessage {
  type: 'updateConfig';
  config: Partial<LayoutConfig>;
}

interface UpdateNodeMessage {
  type: 'updateNode';
  nodeId: string;
  updates: Partial<WorkerNode>;
}

type IncomingMessage =
  | InitMessage
  | StepMessage
  | StopMessage
  | UpdateConfigMessage
  | UpdateNodeMessage;

interface PositionUpdate {
  id: string;
  x: number;
  y: number;
}

interface ProgressMessage {
  type: 'progress';
  iteration: number;
  maxIterations: number;
  totalMovement: number;
  positions: PositionUpdate[];
}

interface CompleteMessage {
  type: 'complete';
  positions: PositionUpdate[];
  iterations: number;
  converged: boolean;
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

type OutgoingMessage = ProgressMessage | CompleteMessage | ErrorMessage;

// ============================================================================
// State
// ============================================================================

let nodes: Map<string, WorkerNode> = new Map();
let edges: WorkerEdge[] = [];
let config: LayoutConfig = {
  repulsionStrength: 1000,
  attractionStrength: 0.1,
  damping: 0.9,
  centerGravity: 0.05,
  maxIterations: 300,
  convergenceThreshold: 0.5,
  width: 800,
  height: 600,
  minDistance: 30,
  theta: 0.8,
};
let running = false;
let iteration = 0;

// ============================================================================
// Force Calculations
// ============================================================================

/**
 * Calculate repulsive force between two nodes (Coulomb's law)
 */
function calculateRepulsion(
  n1: WorkerNode,
  n2: WorkerNode,
  strength: number
): { fx: number; fy: number } {
  const dx = n1.x - n2.x;
  const dy = n1.y - n2.y;
  const distSq = dx * dx + dy * dy;
  const dist = Math.sqrt(distSq) || 0.01; // Avoid division by zero

  if (dist < config.minDistance) {
    // Very close nodes get stronger repulsion
    const force = (strength * 2) / (config.minDistance * config.minDistance);
    return {
      fx: (dx / dist) * force,
      fy: (dy / dist) * force,
    };
  }

  // Standard inverse-square repulsion
  const force = strength / distSq;
  return {
    fx: (dx / dist) * force,
    fy: (dy / dist) * force,
  };
}

/**
 * Calculate attractive force along an edge (spring force - Hooke's law)
 */
function calculateAttraction(
  source: WorkerNode,
  target: WorkerNode,
  weight: number,
  strength: number
): { fx: number; fy: number } {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;

  // Spring force proportional to distance
  const force = strength * dist * weight;
  return {
    fx: (dx / dist) * force,
    fy: (dy / dist) * force,
  };
}

/**
 * Calculate center gravity force (pulls toward viewport center)
 */
function calculateCenterGravity(
  node: WorkerNode,
  strength: number
): { fx: number; fy: number } {
  const centerX = config.width / 2;
  const centerY = config.height / 2;
  const dx = centerX - node.x;
  const dy = centerY - node.y;
  return {
    fx: dx * strength,
    fy: dy * strength,
  };
}

// ============================================================================
// Simulation
// ============================================================================

/**
 * Run one iteration of the force simulation
 */
function simulationStep(): number {
  const forces = new Map<string, { fx: number; fy: number }>();

  // Initialize forces
  for (const [id] of nodes) {
    forces.set(id, { fx: 0, fy: 0 });
  }

  const nodeArray = Array.from(nodes.values());
  const n = nodeArray.length;

  // Calculate repulsive forces (O(n²) - could optimize with Barnes-Hut)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const n1 = nodeArray[i];
      const n2 = nodeArray[j];

      const rep = calculateRepulsion(n1, n2, config.repulsionStrength);

      const f1 = forces.get(n1.id)!;
      const f2 = forces.get(n2.id)!;

      // Apply equal and opposite forces
      f1.fx += rep.fx;
      f1.fy += rep.fy;
      f2.fx -= rep.fx;
      f2.fy -= rep.fy;
    }
  }

  // Calculate attractive forces along edges
  for (const edge of edges) {
    const source = nodes.get(edge.source);
    const target = nodes.get(edge.target);
    if (!source || !target) continue;

    const attr = calculateAttraction(
      source,
      target,
      edge.weight,
      config.attractionStrength
    );

    const fSource = forces.get(source.id)!;
    const fTarget = forces.get(target.id)!;

    fSource.fx += attr.fx;
    fSource.fy += attr.fy;
    fTarget.fx -= attr.fx;
    fTarget.fy -= attr.fy;
  }

  // Calculate center gravity
  for (const node of nodeArray) {
    const gravity = calculateCenterGravity(node, config.centerGravity);
    const f = forces.get(node.id)!;
    f.fx += gravity.fx;
    f.fy += gravity.fy;
  }

  // Apply forces to velocities, then positions
  let totalMovement = 0;

  for (const node of nodeArray) {
    if (node.fixed) continue;

    const f = forces.get(node.id)!;

    // Scale force by importance (less important nodes move more freely)
    const mobilityFactor = 1.0 / (node.importance + 0.1);

    // Update velocity (with damping)
    node.vx = (node.vx + f.fx * mobilityFactor) * config.damping;
    node.vy = (node.vy + f.fy * mobilityFactor) * config.damping;

    // Limit velocity to prevent instability
    const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
    const maxSpeed = 50;
    if (speed > maxSpeed) {
      node.vx = (node.vx / speed) * maxSpeed;
      node.vy = (node.vy / speed) * maxSpeed;
    }

    // Update position
    const oldX = node.x;
    const oldY = node.y;
    node.x += node.vx;
    node.y += node.vy;

    // Keep nodes within bounds (with soft boundary)
    const margin = 50;
    const softness = 0.1;
    if (node.x < margin) node.vx += (margin - node.x) * softness;
    if (node.x > config.width - margin)
      node.vx -= (node.x - (config.width - margin)) * softness;
    if (node.y < margin) node.vy += (margin - node.y) * softness;
    if (node.y > config.height - margin)
      node.vy -= (node.y - (config.height - margin)) * softness;

    // Clamp to hard bounds
    node.x = Math.max(10, Math.min(config.width - 10, node.x));
    node.y = Math.max(10, Math.min(config.height - 10, node.y));

    // Track total movement
    const dx = node.x - oldX;
    const dy = node.y - oldY;
    totalMovement += Math.sqrt(dx * dx + dy * dy);
  }

  return totalMovement;
}

/**
 * Get current positions as updates
 */
function getPositionUpdates(): PositionUpdate[] {
  return Array.from(nodes.values()).map((node) => ({
    id: node.id,
    x: node.x,
    y: node.y,
  }));
}

/**
 * Run simulation for specified iterations
 */
function runSimulation(iterations: number): void {
  if (!running) return;

  let converged = false;

  for (let i = 0; i < iterations && running; i++) {
    iteration++;
    const totalMovement = simulationStep();

    // Check for convergence
    if (totalMovement < config.convergenceThreshold) {
      converged = true;
      break;
    }

    // Check for max iterations
    if (iteration >= config.maxIterations) {
      break;
    }
  }

  // Send progress update
  const positions = getPositionUpdates();

  if (converged || iteration >= config.maxIterations || !running) {
    // Simulation complete
    running = false;
    const message: CompleteMessage = {
      type: 'complete',
      positions,
      iterations: iteration,
      converged,
    };
    self.postMessage(message);
  } else {
    // Still running - send progress
    const message: ProgressMessage = {
      type: 'progress',
      iteration,
      maxIterations: config.maxIterations,
      totalMovement: 0, // Could calculate this
      positions,
    };
    self.postMessage(message);
  }
}

// ============================================================================
// Message Handler
// ============================================================================

self.onmessage = (event: MessageEvent<IncomingMessage>) => {
  const msg = event.data;

  try {
    switch (msg.type) {
      case 'init': {
        // Initialize simulation
        nodes = new Map();
        for (const node of msg.nodes) {
          nodes.set(node.id, { ...node });
        }
        edges = [...msg.edges];
        config = { ...config, ...msg.config };
        iteration = 0;
        running = true;

        // Start simulation
        runSimulation(1);
        break;
      }

      case 'step': {
        // Run more iterations
        if (running) {
          runSimulation(msg.iterations ?? 1);
        }
        break;
      }

      case 'stop': {
        // Stop simulation
        running = false;
        const message: CompleteMessage = {
          type: 'complete',
          positions: getPositionUpdates(),
          iterations: iteration,
          converged: false,
        };
        self.postMessage(message);
        break;
      }

      case 'updateConfig': {
        // Update configuration
        Object.assign(config, msg.config);
        break;
      }

      case 'updateNode': {
        // Update a specific node
        const node = nodes.get(msg.nodeId);
        if (node) {
          Object.assign(node, msg.updates);
        }
        break;
      }
    }
  } catch (error) {
    const message: ErrorMessage = {
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(message);
  }
};

// Signal that worker is ready
self.postMessage({ type: 'ready' });
