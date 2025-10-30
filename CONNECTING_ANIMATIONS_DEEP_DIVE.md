# Connecting Animations: The Deep Dive

**Date**: 2025-10-29
**Context**: Going beyond basic patterns into advanced theory and implementation
**Philosophy**: "Animation is applied perceptual psychology"

---

## Part 1: Mathematical Foundations

### 1.1 Spring Physics (Natural Motion)

**Why springs?** They create motion that feels **organic**, not robotic.

#### The Math: Hooke's Law + Damping

```javascript
class Spring {
    constructor(stiffness = 170, damping = 26, mass = 1) {
        this.k = stiffness;   // Spring constant (N/m)
        this.c = damping;     // Damping coefficient
        this.m = mass;        // Mass (kg)

        this.position = 0;
        this.velocity = 0;
        this.target = 0;
    }

    // Euler integration (simple, fast)
    updateEuler(dt = 0.016) {  // 60fps = ~16ms
        const F_spring = -this.k * (this.position - this.target);
        const F_damping = -this.c * this.velocity;
        const F_total = F_spring + F_damping;

        const acceleration = F_total / this.m;
        this.velocity += acceleration * dt;
        this.position += this.velocity * dt;

        return this.position;
    }

    // Runge-Kutta 4th order (accurate, smooth)
    updateRK4(dt = 0.016) {
        const derive = (pos, vel) => {
            const F_spring = -this.k * (pos - this.target);
            const F_damping = -this.c * vel;
            return {
                velocity: vel,
                acceleration: (F_spring + F_damping) / this.m
            };
        };

        const k1 = derive(this.position, this.velocity);
        const k2 = derive(
            this.position + k1.velocity * dt/2,
            this.velocity + k1.acceleration * dt/2
        );
        const k3 = derive(
            this.position + k2.velocity * dt/2,
            this.velocity + k2.acceleration * dt/2
        );
        const k4 = derive(
            this.position + k3.velocity * dt,
            this.velocity + k3.acceleration * dt
        );

        this.velocity += (k1.acceleration + 2*k2.acceleration + 2*k3.acceleration + k4.acceleration) * dt/6;
        this.position += (k1.velocity + 2*k2.velocity + 2*k3.velocity + k4.velocity) * dt/6;

        return this.position;
    }

    // Check if settled
    isAtRest(threshold = 0.01) {
        return Math.abs(this.position - this.target) < threshold &&
               Math.abs(this.velocity) < threshold;
    }
}
```

**Usage for smooth animations:**

```javascript
// Traditional easing (feels mechanical)
element.transition().duration(500).style('opacity', 1);

// Spring physics (feels organic)
const spring = new Spring(170, 26, 1);  // iOS default values
spring.target = 1;

function animate() {
    const opacity = spring.updateRK4();
    element.style('opacity', opacity);

    if (!spring.isAtRest()) {
        requestAnimationFrame(animate);
    }
}
animate();
```

**Tuning springs for different feels:**

```javascript
// Bouncy (low damping)
new Spring(200, 10, 1)  // Overshoots, oscillates

// Smooth (critical damping: c = 2√(km))
new Spring(170, 26, 1)  // iOS default - no overshoot, fast settle

// Sluggish (high damping)
new Spring(170, 50, 1)  // Slow, heavy feel

// Responsive (high stiffness)
new Spring(400, 40, 1)  // Snappy, quick response
```

---

### 1.2 Bezier Curves (Spatial Relationships)

**Beyond simple quadratic—use cubic Bezier for full control.**

#### The Math: De Casteljau's Algorithm

```javascript
class CubicBezier {
    constructor(p0, p1, p2, p3) {
        this.p0 = p0;  // Start point
        this.p1 = p1;  // First control point
        this.p2 = p2;  // Second control point
        this.p3 = p3;  // End point
    }

    // Get point at parameter t (0 to 1)
    getPoint(t) {
        // De Casteljau's algorithm (numerically stable)
        const u = 1 - t;

        const a = {
            x: u * this.p0.x + t * this.p1.x,
            y: u * this.p0.y + t * this.p1.y
        };
        const b = {
            x: u * this.p1.x + t * this.p2.x,
            y: u * this.p1.y + t * this.p2.y
        };
        const c = {
            x: u * this.p2.x + t * this.p3.x,
            y: u * this.p2.y + t * this.p3.y
        };

        const d = {
            x: u * a.x + t * b.x,
            y: u * a.y + t * b.y
        };
        const e = {
            x: u * b.x + t * c.x,
            y: u * b.y + t * c.y
        };

        return {
            x: u * d.x + t * e.x,
            y: u * d.y + t * e.y
        };
    }

    // Get tangent (velocity) at parameter t
    getTangent(t) {
        const u = 1 - t;

        return {
            x: 3*u*u*(this.p1.x - this.p0.x) +
               6*u*t*(this.p2.x - this.p1.x) +
               3*t*t*(this.p3.x - this.p2.x),
            y: 3*u*u*(this.p1.y - this.p0.y) +
               6*u*t*(this.p2.y - this.p1.y) +
               3*t*t*(this.p3.y - this.p2.y)
        };
    }

    // Get curvature at parameter t
    getCurvature(t) {
        const tangent = this.getTangent(t);
        const u = 1 - t;

        const d2x = 6*u*(this.p2.x - 2*this.p1.x + this.p0.x) +
                    6*t*(this.p3.x - 2*this.p2.x + this.p1.x);
        const d2y = 6*u*(this.p2.y - 2*this.p1.y + this.p0.y) +
                    6*t*(this.p3.y - 2*this.p2.y + this.p1.y);

        const num = tangent.x * d2y - tangent.y * d2x;
        const denom = Math.pow(tangent.x*tangent.x + tangent.y*tangent.y, 1.5);

        return num / denom;
    }

    // Arc length (numerical integration)
    getLength(steps = 100) {
        let length = 0;
        let prevPoint = this.getPoint(0);

        for (let i = 1; i <= steps; i++) {
            const t = i / steps;
            const point = this.getPoint(t);

            const dx = point.x - prevPoint.x;
            const dy = point.y - prevPoint.y;
            length += Math.sqrt(dx*dx + dy*dy);

            prevPoint = point;
        }

        return length;
    }
}
```

**Semantic curve shapes:**

```javascript
// Data flow (smooth S-curve)
const dataFlow = new CubicBezier(
    {x: 100, y: 200},   // Start
    {x: 200, y: 100},   // Pull upward
    {x: 400, y: 100},   // Keep elevated
    {x: 500, y: 200}    // Land smoothly
);

// Causation (sharp turn)
const causation = new CubicBezier(
    {x: 100, y: 200},
    {x: 150, y: 200},   // Straight initially
    {x: 450, y: 100},   // Sharp turn
    {x: 500, y: 100}
);

// Feedback loop (circular)
const feedback = new CubicBezier(
    {x: 100, y: 200},
    {x: 100, y: 50},    // Arc out
    {x: 500, y: 50},    // Arc across
    {x: 500, y: 200}
);
```

---

### 1.3 Force-Directed Graph (Relationship Discovery)

**Simulate physics to reveal hidden structure.**

#### The Math: Fruchterman-Reingold Algorithm

```javascript
class ForceDirectedGraph {
    constructor(nodes, edges, width, height) {
        this.nodes = nodes.map(n => ({
            ...n,
            x: Math.random() * width,
            y: Math.random() * height,
            vx: 0,
            vy: 0
        }));
        this.edges = edges;
        this.width = width;
        this.height = height;

        // Physics parameters
        this.k = Math.sqrt((width * height) / nodes.length);  // Ideal spring length
        this.temperature = width / 10;  // Initial temperature
        this.cooling = 0.95;  // Cooling rate
    }

    // Attractive force (edges pull connected nodes together)
    attractiveForce(distance) {
        return (distance * distance) / this.k;
    }

    // Repulsive force (all nodes push each other apart)
    repulsiveForce(distance) {
        return (this.k * this.k) / distance;
    }

    // Single iteration
    step() {
        // Calculate repulsive forces (all pairs)
        for (let i = 0; i < this.nodes.length; i++) {
            this.nodes[i].fx = 0;
            this.nodes[i].fy = 0;

            for (let j = 0; j < this.nodes.length; j++) {
                if (i === j) continue;

                const dx = this.nodes[i].x - this.nodes[j].x;
                const dy = this.nodes[i].y - this.nodes[j].y;
                const distance = Math.sqrt(dx*dx + dy*dy) || 0.01;

                const force = this.repulsiveForce(distance);
                this.nodes[i].fx += (dx / distance) * force;
                this.nodes[i].fy += (dy / distance) * force;
            }
        }

        // Calculate attractive forces (edges only)
        for (const edge of this.edges) {
            const source = this.nodes[edge.source];
            const target = this.nodes[edge.target];

            const dx = source.x - target.x;
            const dy = source.y - target.y;
            const distance = Math.sqrt(dx*dx + dy*dy) || 0.01;

            const force = this.attractiveForce(distance);
            const fx = (dx / distance) * force;
            const fy = (dy / distance) * force;

            source.fx -= fx;
            source.fy -= fy;
            target.fx += fx;
            target.fy += fy;
        }

        // Apply forces with temperature
        for (const node of this.nodes) {
            const dx = node.fx;
            const dy = node.fy;
            const displacement = Math.sqrt(dx*dx + dy*dy) || 0.01;

            // Limit movement by temperature
            node.x += (dx / displacement) * Math.min(displacement, this.temperature);
            node.y += (dy / displacement) * Math.min(displacement, this.temperature);

            // Keep in bounds
            node.x = Math.max(0, Math.min(this.width, node.x));
            node.y = Math.max(0, Math.min(this.height, node.y));
        }

        // Cool down
        this.temperature *= this.cooling;
    }

    // Run until stable
    simulate(maxIterations = 100) {
        for (let i = 0; i < maxIterations && this.temperature > 1; i++) {
            this.step();
        }
    }
}
```

**Use for dashboard connections:**

```javascript
// Build graph from dashboard relationships
const nodes = [
    {id: 'storage-latency', label: 'Storage Latency'},
    {id: 'search-latency', label: 'Search Latency'},
    {id: 'throughput', label: 'Throughput'},
    {id: 'confidence', label: 'Confidence'},
    {id: 'quality', label: 'Quality'}
];

const edges = [
    {source: 0, target: 1, weight: 1.0},  // storage → search
    {source: 0, target: 2, weight: 0.8},  // storage → throughput
    {source: 1, target: 3, weight: 0.9},  // search → confidence
    {source: 3, target: 4, weight: 1.0}   // confidence → quality
];

// Simulate layout
const graph = new ForceDirectedGraph(nodes, edges, 800, 600);
graph.simulate(100);

// Animate to discovered positions
nodes.forEach((node, i) => {
    const chartElement = d3.select(`#chart-${node.id}`);
    chartElement.transition()
        .duration(2000)
        .ease(d3.easeCubicInOut)
        .style('left', `${graph.nodes[i].x}px`)
        .style('top', `${graph.nodes[i].y}px`);
});
```

---

## Part 2: Perceptual Psychology Foundations

### 2.1 Pre-Attentive Processing

**Human visual system processes certain attributes <250ms unconsciously.**

#### Exploitable Pre-Attentive Attributes

```javascript
const preAttentiveAttributes = {
    // Color
    hue: {
        processingTime: 50,
        usage: 'Categorization (related items same hue)',
        example: 'All latency metrics in blue, all quality metrics in green'
    },

    // Motion
    direction: {
        processingTime: 80,
        usage: 'Flow indication (particles moving A→B)',
        example: 'Data flowing from source to aggregate'
    },

    // Position
    spatial: {
        processingTime: 100,
        usage: 'Grouping (proximity = relatedness)',
        example: 'Related charts cluster together'
    },

    // Size
    magnitude: {
        processingTime: 120,
        usage: 'Importance (bigger = more important)',
        example: 'Active chart scales up 10%'
    },

    // Orientation
    angle: {
        processingTime: 150,
        usage: 'Direction (arrows, flow lines)',
        example: 'Connection arrows point from cause to effect'
    }
};
```

**Animation strategy:**

```javascript
class PreAttentiveAnimation {
    // WRONG: Color change requires conscious attention
    highlightSlow(element) {
        element.transition().duration(300)
            .style('background-color', '#3498db');  // Slow to notice
    }

    // RIGHT: Motion triggers pre-attentive system
    highlightFast(element) {
        element.transition().duration(150)
            .style('transform', 'scale(1.05)');  // Instant notice
    }

    // BEST: Combine multiple pre-attentive cues
    highlightOptimal(element) {
        element.transition().duration(150)
            .style('transform', 'scale(1.05)')      // Size (magnitude)
            .style('box-shadow', '0 0 20px #0ff')   // Glow (additive color)
            .style('transform-origin', 'center');    // Position (spatial)
    }
}
```

---

### 2.2 Gestalt Principles

**Brain organizes visual elements into groups.**

#### Proximity (Near = Related)

```javascript
// Bad: Random positions
charts.forEach((chart, i) => {
    chart.style.left = `${Math.random() * 800}px`;
    chart.style.top = `${Math.random() * 600}px`;
});

// Good: Related charts cluster
const clusters = {
    latency: [chart1, chart2],
    quality: [chart3, chart4],
    throughput: [chart5, chart6]
};

Object.entries(clusters).forEach(([category, charts], clusterIndex) => {
    const clusterX = (clusterIndex % 3) * 300;
    const clusterY = Math.floor(clusterIndex / 3) * 250;

    charts.forEach((chart, i) => {
        chart.style.left = `${clusterX + (i % 2) * 140}px`;
        chart.style.top = `${clusterY + Math.floor(i / 2) * 100}px`;
    });
});
```

#### Similarity (Look alike = Related)

```javascript
// Apply consistent visual grammar
const metricStyles = {
    latency: { color: '#3498db', icon: '⚡' },
    quality: { color: '#2ecc71', icon: '✓' },
    error: { color: '#e74c3c', icon: '⚠' }
};

function styleMetric(chart, category) {
    const style = metricStyles[category];

    chart.style.borderColor = style.color;
    chart.querySelector('.icon').textContent = style.icon;
    chart.querySelector('.trend-line').style.stroke = style.color;
}
```

#### Continuity (Smooth paths preferred)

```javascript
// Bad: Jagged connection
const zigzag = `M ${x1} ${y1} L ${x2} ${y1} L ${x2} ${y2}`;

// Good: Smooth curve
const smooth = `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;

// Best: Natural arc following visual flow
const arc = d3.arc()
    .innerRadius(0)
    .outerRadius((x2 - x1))
    .startAngle(0)
    .endAngle(Math.PI / 2);
```

#### Common Fate (Move together = Related)

```javascript
// When one chart updates, animate related charts
function updateMetric(metricId, newValue) {
    const relatedCharts = findRelatedCharts(metricId);

    // All related charts animate together (common fate)
    relatedCharts.forEach((chart, i) => {
        chart.transition()
            .delay(i * 100)  // Slight stagger for visibility
            .duration(600)
            .style('opacity', 0.5)
            .transition()
            .duration(600)
            .style('opacity', 1);
    });
}
```

---

### 2.3 Change Blindness vs. Change Detection

**Rapid changes are invisible unless attention is directed.**

#### The Problem

```javascript
// BAD: Instant update (user misses change)
function updateChart(data) {
    chart.data(data);
    chart.render();  // Blink and you miss it
}
```

#### The Solution: Pre-Announce Changes

```javascript
// GOOD: Telegraph → Pause → Change → Confirm
async function updateChart(data) {
    // 1. Telegraph (draw attention)
    chart.transition().duration(200)
        .style('box-shadow', '0 0 30px rgba(255, 215, 0, 0.8)');

    await sleep(300);

    // 2. Change (with smooth transition)
    chart.data(data);
    chart.transition().duration(800)
        .call(chart.render);

    await sleep(800);

    // 3. Confirm (flash briefly)
    chart.transition().duration(200)
        .style('box-shadow', '0 0 30px rgba(46, 204, 113, 0.8)')
        .transition().duration(500)
        .style('box-shadow', 'none');
}
```

---

## Part 3: Advanced Animation Patterns

### 3.1 Morphing Between Chart Types

**Smoothly transform one visualization into another.**

```javascript
class ChartMorph {
    // Bar chart → Line chart
    barToLine(barChart, lineChart) {
        const bars = barChart.selectAll('rect');
        const data = bars.data();

        // 1. Shrink bars to thin lines
        bars.transition().duration(400)
            .attr('width', 2)
            .attr('x', d => d.x)  // Center on original position
            .style('opacity', 0.5);

        // 2. Connect tops with path
        const line = d3.line()
            .x(d => d.x)
            .y(d => d.height)
            .curve(d3.curveMonotoneX);

        lineChart.append('path')
            .datum(data)
            .attr('d', line)
            .attr('stroke', '#3498db')
            .attr('fill', 'none')
            .style('opacity', 0)
            .transition().duration(400).delay(200)
            .style('opacity', 1);

        // 3. Remove bars
        bars.transition().duration(200).delay(600)
            .style('opacity', 0)
            .remove();
    }

    // Scatter → Heatmap (binning animation)
    scatterToHeatmap(scatter, heatmap, bins = 10) {
        const points = scatter.selectAll('circle');
        const data = points.data();

        // 1. Group points into bins
        const grouped = d3.group(data,
            d => Math.floor(d.x / bins),
            d => Math.floor(d.y / bins)
        );

        // 2. Animate points to bin centers
        points.transition().duration(800)
            .attr('cx', d => (Math.floor(d.x / bins) + 0.5) * bins)
            .attr('cy', d => (Math.floor(d.y / bins) + 0.5) * bins)
            .attr('r', 2);

        // 3. Fade out points, fade in heatmap
        points.transition().delay(800).duration(400)
            .style('opacity', 0);

        // 4. Draw heatmap cells
        grouped.forEach((yCells, xBin) => {
            yCells.forEach((points, yBin) => {
                heatmap.append('rect')
                    .attr('x', xBin * bins)
                    .attr('y', yBin * bins)
                    .attr('width', bins)
                    .attr('height', bins)
                    .attr('fill', colorScale(points.length))
                    .style('opacity', 0)
                    .transition().delay(1000).duration(400)
                    .style('opacity', 1);
            });
        });
    }

    // Tree → Sunburst (hierarchy transformation)
    treeToSunburst(tree, sunburst) {
        const nodes = tree.selectAll('.node');

        // 1. Calculate sunburst positions
        const partition = d3.partition()
            .size([2 * Math.PI, radius]);

        const root = d3.hierarchy(treeData);
        partition(root);

        // 2. Animate nodes to radial positions
        nodes.transition().duration(1000)
            .ease(d3.easeCubicInOut)
            .attrTween('transform', function(d) {
                const node = root.find(n => n.data.id === d.id);
                const angle = (node.x0 + node.x1) / 2;
                const radius = (node.y0 + node.y1) / 2;

                return d3.interpolateString(
                    `translate(${d.x}, ${d.y})`,
                    `translate(${radius * Math.cos(angle)}, ${radius * Math.sin(angle)})`
                );
            });

        // 3. Morph shapes
        tree.selectAll('circle')
            .transition().duration(1000)
            .attrTween('d', function(d) {
                const node = root.find(n => n.data.id === d.id);
                const arc = d3.arc()
                    .startAngle(node.x0)
                    .endAngle(node.x1)
                    .innerRadius(node.y0)
                    .outerRadius(node.y1);

                return d3.interpolate(
                    `M ${-d.radius} 0 A ${d.radius} ${d.radius} 0 1 1 ${d.radius} 0 Z`,
                    arc()
                );
            });
    }
}
```

---

### 3.2 Semantic Zoom (Multi-Scale Navigation)

**Different details at different zoom levels.**

```javascript
class SemanticZoom {
    constructor(svg, data) {
        this.svg = svg;
        this.data = data;
        this.zoom = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', (event) => this.zoomed(event));

        svg.call(this.zoom);

        // Define level-of-detail thresholds
        this.lodLevels = [
            { minScale: 0, maxScale: 1, detail: 'overview' },
            { minScale: 1, maxScale: 3, detail: 'medium' },
            { minScale: 3, maxScale: 10, detail: 'detailed' }
        ];
    }

    zoomed(event) {
        const { transform } = event;
        const scale = transform.k;

        // Determine current LOD
        const lod = this.lodLevels.find(
            level => scale >= level.minScale && scale < level.maxScale
        ).detail;

        // Update visualization based on LOD
        switch(lod) {
            case 'overview':
                this.showOverview(transform);
                break;
            case 'medium':
                this.showMedium(transform);
                break;
            case 'detailed':
                this.showDetailed(transform);
                break;
        }
    }

    showOverview(transform) {
        // Show only aggregate metrics
        this.svg.selectAll('.sparkline').style('opacity', 0);
        this.svg.selectAll('.detail-text').style('opacity', 0);
        this.svg.selectAll('.summary').style('opacity', 1);

        // Large markers
        this.svg.selectAll('.marker')
            .attr('r', 8 / transform.k);  // Scale-independent size
    }

    showMedium(transform) {
        // Show sparklines
        this.svg.selectAll('.sparkline')
            .transition().duration(300)
            .style('opacity', 1);

        this.svg.selectAll('.summary').style('opacity', 0);
        this.svg.selectAll('.detail-text').style('opacity', 0);

        // Medium markers
        this.svg.selectAll('.marker')
            .attr('r', 5 / transform.k);
    }

    showDetailed(transform) {
        // Show full detail
        this.svg.selectAll('.detail-text')
            .transition().duration(300)
            .style('opacity', 1);

        this.svg.selectAll('.sparkline').style('opacity', 1);
        this.svg.selectAll('.summary').style('opacity', 0);

        // Small markers
        this.svg.selectAll('.marker')
            .attr('r', 3 / transform.k);

        // Show tooltips
        this.svg.selectAll('.tooltip-trigger')
            .on('mouseenter', this.showTooltip);
    }
}
```

---

### 3.3 Hierarchical Edge Bundling (Reduce Visual Clutter)

**Bundle related connections into smooth curves.**

```javascript
class HierarchicalEdgeBundling {
    constructor(nodes, edges, hierarchy) {
        this.nodes = nodes;
        this.edges = edges;
        this.hierarchy = hierarchy;
    }

    // Find path through hierarchy
    findPath(source, target) {
        const sourcePath = this.getAncestors(source);
        const targetPath = this.getAncestors(target);

        // Find lowest common ancestor
        let lca = null;
        for (let i = 0; i < Math.min(sourcePath.length, targetPath.length); i++) {
            if (sourcePath[i] === targetPath[i]) {
                lca = sourcePath[i];
            } else {
                break;
            }
        }

        // Build path: source → ... → LCA → ... → target
        const path = [];

        // Source to LCA
        for (let node = source; node !== lca; node = node.parent) {
            path.push(node);
        }
        path.push(lca);

        // LCA to target (reverse)
        const targetToLCA = [];
        for (let node = target; node !== lca; node = node.parent) {
            targetToLCA.push(node);
        }
        path.push(...targetToLCA.reverse());

        return path;
    }

    // Create bundled curve
    createBundle(source, target, beta = 0.85) {
        const path = this.findPath(source, target);

        // Interpolate positions with bundling strength beta
        const points = path.map((node, i) => {
            const t = i / (path.length - 1);

            // Direct line point
            const directX = (1 - t) * source.x + t * target.x;
            const directY = (1 - t) * source.y + t * target.y;

            // Hierarchical path point
            const hierarchyX = node.x;
            const hierarchyY = node.y;

            // Blend based on beta (0 = straight, 1 = full bundle)
            return {
                x: (1 - beta) * directX + beta * hierarchyX,
                y: (1 - beta) * directY + beta * hierarchyY
            };
        });

        // Create smooth curve through points
        const line = d3.line()
            .x(d => d.x)
            .y(d => d.y)
            .curve(d3.curveBundle.beta(beta));

        return line(points);
    }

    // Render all edges
    render(svg) {
        const bundledEdges = this.edges.map(edge => ({
            ...edge,
            path: this.createBundle(edge.source, edge.target)
        }));

        svg.selectAll('.edge')
            .data(bundledEdges)
            .join('path')
            .attr('class', 'edge')
            .attr('d', d => d.path)
            .attr('stroke', '#999')
            .attr('stroke-width', 1)
            .attr('fill', 'none')
            .attr('opacity', 0.3);
    }
}
```

---

## Part 4: Production Implementation

### 4.1 Performance at Scale

**Handling 1000+ charts with smooth animations.**

```javascript
class PerformantAnimationEngine {
    constructor() {
        this.activeAnimations = new Set();
        this.animationFrame = null;
        this.maxConcurrent = 50;  // GPU limit
        this.queue = [];
    }

    // Batch animations to prevent frame drops
    animate(element, properties, duration, easing = d3.easeCubicOut) {
        if (this.activeAnimations.size >= this.maxConcurrent) {
            // Queue for later
            this.queue.push({ element, properties, duration, easing });
            return;
        }

        const animation = {
            element,
            properties,
            duration,
            easing,
            startTime: performance.now(),
            initialValues: {}
        };

        // Capture initial values
        Object.keys(properties).forEach(prop => {
            animation.initialValues[prop] = this.getValue(element, prop);
        });

        this.activeAnimations.add(animation);

        if (!this.animationFrame) {
            this.animationFrame = requestAnimationFrame(
                this.tick.bind(this)
            );
        }
    }

    tick(currentTime) {
        const completed = [];

        for (const anim of this.activeAnimations) {
            const elapsed = currentTime - anim.startTime;
            const progress = Math.min(elapsed / anim.duration, 1);
            const easedProgress = anim.easing(progress);

            // Update properties
            Object.entries(anim.properties).forEach(([prop, target]) => {
                const initial = anim.initialValues[prop];
                const current = initial + (target - initial) * easedProgress;
                this.setValue(anim.element, prop, current);
            });

            if (progress >= 1) {
                completed.push(anim);
            }
        }

        // Remove completed
        completed.forEach(anim => this.activeAnimations.delete(anim));

        // Process queue
        while (this.queue.length > 0 &&
               this.activeAnimations.size < this.maxConcurrent) {
            const next = this.queue.shift();
            this.animate(next.element, next.properties, next.duration, next.easing);
        }

        // Continue loop if needed
        if (this.activeAnimations.size > 0 || this.queue.length > 0) {
            this.animationFrame = requestAnimationFrame(this.tick.bind(this));
        } else {
            this.animationFrame = null;
        }
    }

    getValue(element, property) {
        const computed = window.getComputedStyle(element);
        const value = computed.getPropertyValue(property);
        return parseFloat(value) || 0;
    }

    setValue(element, property, value) {
        if (property.startsWith('transform-')) {
            // Handle transform properties specially
            const transform = element.style.transform || '';
            const updated = this.updateTransform(transform, property, value);
            element.style.transform = updated;
        } else {
            element.style[property] = value;
        }
    }

    updateTransform(current, property, value) {
        const transforms = {
            'transform-translateX': `translateX(${value}px)`,
            'transform-translateY': `translateY(${value}px)`,
            'transform-scale': `scale(${value})`,
            'transform-rotate': `rotate(${value}deg)`
        };

        // Parse current transform
        const parts = current.match(/(\w+)\([^)]+\)/g) || [];
        const updated = parts.filter(p => !p.startsWith(property.split('-')[1]));
        updated.push(transforms[property]);

        return updated.join(' ');
    }
}
```

---

### 4.2 Accessibility (Reduced Motion)

**Respect user preferences.**

```javascript
class AccessibleAnimation {
    constructor() {
        this.prefersReducedMotion = window.matchMedia(
            '(prefers-reduced-motion: reduce)'
        ).matches;

        // Listen for changes
        window.matchMedia('(prefers-reduced-motion: reduce)')
            .addEventListener('change', (e) => {
                this.prefersReducedMotion = e.matches;
                this.updateAnimations();
            });
    }

    animate(element, from, to) {
        if (this.prefersReducedMotion) {
            // Instant transition
            element.style.opacity = to.opacity;
            element.style.transform = to.transform;

            // But still show connection (static)
            this.showStaticConnection(element);
        } else {
            // Smooth animation
            element.transition()
                .duration(600)
                .style('opacity', to.opacity)
                .style('transform', to.transform);

            this.showAnimatedConnection(element);
        }
    }

    showStaticConnection(element) {
        // Draw permanent line instead of animated particles
        const target = element.getAttribute('data-connected-to');
        if (target) {
            const line = this.drawStaticLine(element, target);
            line.style.strokeDasharray = 'none';  // Solid line
        }
    }

    showAnimatedConnection(element) {
        // Flow particles
        const target = element.getAttribute('data-connected-to');
        if (target) {
            this.createParticles(element, target);
        }
    }
}
```

---

## Part 5: Real HoloLoom Integration

### 5.1 Actual Pipeline Connections

**Map real HoloLoom components:**

```javascript
const hololoomPipeline = {
    nodes: [
        { id: 'input', label: 'Input Adapters', component: 'SpinningWheel' },
        { id: 'motif', label: 'Motif Detection', component: 'MotifDetector' },
        { id: 'embedding', label: 'Embeddings', component: 'SpectralEmbedder' },
        { id: 'resonance', label: 'Resonance Shed', component: 'ResonanceShed' },
        { id: 'warp', label: 'Warp Space', component: 'WarpSpace' },
        { id: 'memory', label: 'Memory', component: 'YarnGraph' },
        { id: 'policy', label: 'Policy Engine', component: 'UnifiedPolicy' },
        { id: 'convergence', label: 'Convergence', component: 'ConvergenceEngine' },
        { id: 'output', label: 'Spacetime Fabric', component: 'Spacetime' }
    ],

    edges: [
        { source: 'input', target: 'motif', type: 'text-flow' },
        { source: 'input', target: 'embedding', type: 'text-flow' },
        { source: 'motif', target: 'resonance', type: 'features' },
        { source: 'embedding', target: 'resonance', type: 'features' },
        { source: 'memory', target: 'resonance', type: 'context' },
        { source: 'resonance', target: 'warp', type: 'dotplasma' },
        { source: 'warp', target: 'policy', type: 'tensioned' },
        { source: 'policy', target: 'convergence', type: 'distribution' },
        { source: 'convergence', target: 'output', type: 'decision' },
        { source: 'output', target: 'memory', type: 'feedback' }
    ]
};

// Visualize with force-directed layout
const simulation = d3.forceSimulation(hololoomPipeline.nodes)
    .force('link', d3.forceLink(hololoomPipeline.edges).id(d => d.id))
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(400, 300))
    .force('collision', d3.forceCollide().radius(60));

// Animate data flow
function animateQuery(queryText) {
    // Show data packet traveling through pipeline
    const packet = {
        data: queryText,
        position: 'input',
        path: ['input', 'motif', 'resonance', 'warp', 'policy', 'convergence', 'output']
    };

    animatePacket(packet);
}
```

This is getting very deep. Should I continue with Part 6 (Instrumentation & Analytics), Part 7 (A/B Testing Framework), and create advanced working prototypes?

