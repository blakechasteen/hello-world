/**
 * Force-Directed Graph Layout Algorithm
 *
 * Implements Fruchterman-Reingold algorithm for positioning graph nodes.
 * Used by Yarn Graph visualization component.
 */

export class ForceDirectedLayout {
  constructor(width = 900, height = 700, options = {}) {
    this.width = width
    this.height = height
    this.iterations = options.iterations || 300
    this.attractionStrength = options.attractionStrength || 0.01
    this.repulsionStrength = options.repulsionStrength || 1000
    this.damping = options.damping || 0.8
  }

  /**
   * Compute layout positions for nodes
   * @param {Array} nodes - Array of node objects with id
   * @param {Array} edges - Array of edge objects with src, dst
   * @returns {Array} Nodes with updated x, y positions
   */
  layout(nodes, edges) {
    if (!nodes || nodes.length === 0) {
      return []
    }

    // Initialize random positions
    const layoutNodes = nodes.map(node => ({
      ...node,
      x: Math.random() * this.width * 0.6 + this.width * 0.2,
      y: Math.random() * this.height * 0.6 + this.height * 0.2,
      vx: 0,
      vy: 0,
    }))

    // Build adjacency map for fast lookup
    const adjacency = new Map()
    edges.forEach(edge => {
      if (!adjacency.has(edge.src)) {
        adjacency.set(edge.src, new Set())
      }
      adjacency.get(edge.src).add(edge.dst)
    })

    // Simulation loop
    for (let iteration = 0; iteration < this.iterations; iteration++) {
      const cooling = 1.0 - (iteration / this.iterations)

      // Reset velocities (with damping)
      layoutNodes.forEach(node => {
        node.vx *= this.damping
        node.vy *= this.damping
      })

      // Repulsion forces (all pairs)
      for (let i = 0; i < layoutNodes.length; i++) {
        for (let j = i + 1; j < layoutNodes.length; j++) {
          const nodeA = layoutNodes[i]
          const nodeB = layoutNodes[j]

          const dx = nodeB.x - nodeA.x
          const dy = nodeB.y - nodeA.y
          const distanceSq = Math.max(dx * dx + dy * dy, 1.0)
          const distance = Math.sqrt(distanceSq)

          // Repulsion (inverse square)
          const force = this.repulsionStrength / distanceSq
          const fx = (dx / distance) * force
          const fy = (dy / distance) * force

          nodeA.vx -= fx
          nodeA.vy -= fy
          nodeB.vx += fx
          nodeB.vy += fy
        }
      }

      // Attraction forces (edges)
      edges.forEach(edge => {
        const srcNode = layoutNodes.find(n => n.id === edge.src)
        const dstNode = layoutNodes.find(n => n.id === edge.dst)

        if (srcNode && dstNode) {
          const dx = dstNode.x - srcNode.x
          const dy = dstNode.y - srcNode.y
          const distance = Math.sqrt(dx * dx + dy * dy)

          if (distance > 0) {
            // Spring force (linear)
            const weight = edge.weight || 1.0
            const force = this.attractionStrength * distance * weight
            const fx = (dx / distance) * force
            const fy = (dy / distance) * force

            srcNode.vx += fx
            srcNode.vy += fy
            dstNode.vx -= fx
            dstNode.vy -= fy
          }
        }
      })

      // Update positions
      layoutNodes.forEach(node => {
        node.x += node.vx * cooling
        node.y += node.vy * cooling

        // Keep within bounds (with margin)
        const margin = 50
        node.x = Math.max(margin, Math.min(this.width - margin, node.x))
        node.y = Math.max(margin, Math.min(this.height - margin, node.y))
      })
    }

    return layoutNodes
  }
}

/**
 * Edge type color mapping (matches Python backend)
 */
export const EDGE_COLORS = {
  'is_a': '#3b82f6',        // Blue (taxonomy)
  'uses': '#10b981',         // Green (functional)
  'mentions': '#6b7280',     // Gray (reference)
  'leads_to': '#f59e0b',     // Orange (causal)
  'part_of': '#8b5cf6',      // Purple (composition)
  'in_time': '#06b6d4',      // Cyan (temporal)
  'occurred_at': '#14b8a6',  // Teal (event)
  'unknown': '#d1d5db'       // Light gray (fallback)
}

/**
 * Get color for edge type
 */
export function getEdgeColor(edgeType) {
  const normalizedType = (edgeType || 'unknown').toLowerCase()
  return EDGE_COLORS[normalizedType] || EDGE_COLORS.unknown
}

/**
 * Calculate node size based on degree
 */
export function getNodeSize(degree, minSize = 8, maxSize = 24, minDegree = 0, maxDegree = 10) {
  if (maxDegree === minDegree) {
    return (minSize + maxSize) / 2
  }
  const factor = (degree - minDegree) / (maxDegree - minDegree)
  return minSize + factor * (maxSize - minSize)
}
