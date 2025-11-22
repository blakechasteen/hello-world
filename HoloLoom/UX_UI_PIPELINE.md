# HoloLoom UX/UI Initiative: Agent Orchestrator Dashboard Pipeline

## 1. Vision
To create a world-class, premium Agent Orchestrator Dashboard that empowers users to visually design, monitor, and optimize HoloLoom agent workflows. The interface will prioritize aesthetics, responsiveness, and deep interactivity, utilizing a modern tech stack to deliver a "wow" experience.

## 2. Technology Stack
- **Framework**: React (via Vite)
- **Language**: TypeScript (for robustness)
- **Styling**: Vanilla CSS (CSS Variables for theming, Glassmorphism, Animations)
- **State Management**: React Context / Hooks (or Zustand if complexity grows)
- **Visualization**: React Flow (for the node editor), Recharts (for analytics)
- **Backend Integration**: REST API / WebSocket (connecting to `dashboard_server.py`)

## 3. Implementation Roadmap

### Phase 1: Foundation & Design System (Completed)
- [x] Initialize Vite + React project (`hololoom-dashboard`)
- [x] Define Design System (Colors, Typography, Shadows) in `index.css`
- [x] Set up basic layout (Sidebar, Canvas Area, Properties Panel)
- [x] **NEW**: Abstract Glassmorphism into reusable components (`GlassPanel`, `GlassButton`, `GlassInput`) for consistent UI depth and texture.

### Phase 2: Core Workflow Builder (Current Focus)
- [ ] **Canvas Integration**: Install and configure React Flow.
- [ ] **Custom Node Styling**: Create custom node types that utilize the `GlassPanel` abstraction.
- [ ] **Drag & Drop Logic**: Connect the Sidebar palette to the React Flow canvas.
- [ ] **Interactive Edges**: Style connections with animated gradients to simulate data flow.

### Phase 3: Interactivity & Configuration
- [ ] **Properties Panel**: Dynamic configuration forms based on selected node.
- [ ] **State Management**: Robust state for the workflow graph (nodes + edges).
- [ ] **Validation**: Real-time validation of connections (e.g., matching data types).

### Phase 4: Real-time Orchestration & Monitoring
- [ ] **Live Execution**: Visual feedback during workflow execution (active nodes pulsing, data flowing on edges).
- [ ] **Analytics**: Integrate charts/graphs for performance metrics (latency, token usage).
- [ ] **Backend Sync**: Connect to `weaving_orchestrator.py` / `dashboard_server.py` for actual execution.

### Phase 5: Polish & "Wow" Factors
- [ ] **Micro-animations**: Hover effects, entry animations, smooth transitions.
- [ ] **Themes**: Dark/Light mode (defaulting to a sleek Dark Mode).
- [ ] **Keyboard Shortcuts**: Power user controls.

## 4. Design Aesthetics (The "HoloLoom Look")
- **Palette**: Deep space blues/purples (`#0f172a`, `#1e1b4b`) accented with neon cyan (`#06b6d4`) and magenta (`#d946ef`).
- **Surface**: Glassmorphism (translucent backgrounds with blur filters).
- **Typography**: Inter or Outfit (clean, modern sans-serif).
- **Feel**: Fluid, holographic, futuristic.

## 5. Next Steps
1. Refactor `App.jsx` to use new `Glass*` components.
2. Install `reactflow`.
3. Implement the main canvas.
