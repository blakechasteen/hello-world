# HoloLoom Web Dashboard

Production-quality React dashboard for HoloLoom's agentic intelligence system.

## Features

### 1. **Main Dashboard**
- **System Status**: Real-time system health, uptime, query counts, latency metrics
- **Query Interface**: Interactive query submission with 4 reasoning modes
  - Direct (~150ms)
  - Verify (~600ms)
  - Research (~900ms)
  - Plan & Execute (~750ms)
- **Recent Queries**: Live feed of recent system activity
- **Performance Metrics**: Real-time latency tracking with charts

### 2. **Yarn Graph Visualization**
- **Interactive Force-Directed Graph**: D3.js-powered knowledge graph
- **Node Features**:
  - Size based on degree (importance)
  - Click to view details
  - Hover effects
- **Edge Features**:
  - Semantic colors by relationship type
  - Directed arrows
  - Filter by edge type
- **Controls**:
  - Adjust max nodes (10-100)
  - Layout speed (fast/normal/slow)
  - Toggle node labels
  - Filter edge types

### 3. **Real-Time Monitoring**
- **Prometheus Metrics**:
  - API requests, latency, cache hit rate
  - Memory usage, CPU, active connections
  - Error rates with trends
- **Alerts Panel**:
  - Critical/Warning/Info alerts
  - Acknowledge and filter
  - Real-time notifications
- **Historical Charts**:
  - Multi-axis line charts
  - Time range selection (1h, 6h, 24h, 7d)
  - Latency, throughput, error rate trends

### 4. **Real-Time Updates**
- WebSocket connection to backend
- Live query completion notifications
- Metrics streaming
- Alert notifications
- System status changes

## Tech Stack

- **React 18.2**: UI framework
- **Vite 5.0**: Build tool & dev server
- **React Router 6**: Client-side routing
- **D3.js 7.8**: Graph visualizations
- **Chart.js 4.4**: Performance charts
- **Tailwind CSS 3.3**: Styling
- **Lucide React**: Icons
- **Axios**: HTTP client
- **WebSocket**: Real-time communication

## Prerequisites

- Node.js 18+ and npm/yarn
- HoloLoom backend running on `http://localhost:8000`
- (Optional) Prometheus metrics endpoint

## Quick Start

### 1. Install Dependencies

```bash
cd HoloLoom/web_dashboard/frontend
npm install
```

### 2. Configure Environment (Optional)

Create `.env` file in project root:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

### 3. Start Development Server

```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

### 4. Build for Production

```bash
npm run build
```

Built files will be in `dist/` directory.

### 5. Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── Dashboard/       # Dashboard view components
│   │   │   ├── SystemStatus.jsx
│   │   │   ├── QueryInterface.jsx
│   │   │   ├── RecentQueries.jsx
│   │   │   └── PerformanceMetrics.jsx
│   │   ├── Graph/          # Graph visualization
│   │   │   ├── YarnGraphVisualization.jsx
│   │   │   └── GraphControls.jsx
│   │   ├── Monitoring/     # Monitoring components
│   │   │   ├── PrometheusMetrics.jsx
│   │   │   ├── AlertsPanel.jsx
│   │   │   └── HistoricalCharts.jsx
│   │   └── Layout/         # Layout components
│   │       ├── Header.jsx
│   │       ├── Sidebar.jsx
│   │       └── Footer.jsx
│   ├── views/              # Page views
│   │   ├── DashboardView.jsx
│   │   ├── GraphView.jsx
│   │   └── MonitoringView.jsx
│   ├── hooks/              # Custom React hooks
│   │   ├── useWebSocket.js
│   │   ├── useHoloLoom.js
│   │   └── usePrometheus.js
│   ├── utils/              # Utilities
│   │   ├── api.js          # API client
│   │   ├── websocket.js    # WebSocket client
│   │   └── graphLayout.js  # Force-directed layout
│   ├── App.jsx             # Main app component
│   ├── main.jsx            # Entry point
│   └── index.css           # Global styles
├── public/                 # Static assets
├── package.json           # Dependencies
├── vite.config.js        # Vite configuration
├── tailwind.config.js    # Tailwind configuration
└── README.md             # This file
```

## API Integration

The dashboard integrates with the HoloLoom FastAPI backend:

### HTTP Endpoints

- `GET /health` - Health check
- `POST /query` - Submit query
- `GET /stats` - System statistics
- `GET /audit-trail` - Recent queries
- `POST /memories/add` - Add memory
- `GET /metrics` - Prometheus metrics
- `GET /graph/data` - Knowledge graph data

### WebSocket Endpoint

- `ws://localhost:8000/ws` - Real-time updates

**Event Types**:
- `connected` - Connection established
- `query_completed` - Query finished
- `stats_update` - Stats updated
- `metrics_update` - Metrics updated
- `alert` - New alert

**Client → Server Messages**:
```json
{
  "type": "ping",
  "payload": {}
}
```

**Server → Client Messages**:
```json
{
  "type": "query_completed",
  "payload": {
    "query_id": "...",
    "confidence": 0.92,
    "duration_ms": 150
  }
}
```

## Component Documentation

### SystemStatus

Displays real-time system health metrics:
- System status (online/offline)
- Total queries & success rate
- Average latency (P95)
- Memory shards count

**Auto-refresh**: Every 10 seconds
**WebSocket**: Listens for `stats_update` events

### QueryInterface

Interactive query submission:
- 4 reasoning modes (Direct, Verify, Research, Plan & Execute)
- Real-time response display
- Confidence scores
- Verification results (if mode=verify)

**State Management**: Local state for query text, mode, results

### YarnGraphVisualization

Force-directed knowledge graph:
- D3.js-powered layout
- Fruchterman-Reingold algorithm
- Semantic edge colors
- Interactive node selection

**Performance**: Limits to 50 nodes by default (configurable 10-100)

### PrometheusMetrics

Live Prometheus metrics:
- API requests, latency, cache hit rate
- Memory, CPU, connections
- Error rates with trends

**Auto-refresh**: Every 10 seconds (configurable)

## Custom Hooks

### useWebSocket(autoConnect)

Manages WebSocket connection:

```jsx
const { isConnected, subscribe, send } = useWebSocket(true)

// Subscribe to events
useEffect(() => {
  return subscribe('query_completed', (data) => {
    console.log('Query completed:', data)
  })
}, [subscribe])

// Send message
send('ping', {})
```

### useHoloLoom()

API client for HoloLoom:

```jsx
const { query, getStats, loading, error } = useHoloLoom()

// Submit query
const result = await query("What is Thompson Sampling?", {
  mode: 'verify',
  maxSteps: 5
})

// Get stats
const stats = await getStats()
```

### usePrometheus(refreshInterval)

Prometheus metrics:

```jsx
const { metrics, loading, error, refresh } = usePrometheus(5000)

// Manual refresh
refresh()
```

## Styling

Uses **Tailwind CSS 3.3** with custom configuration:

### Custom Colors

```js
'hololoom-primary': '#667eea'
'hololoom-secondary': '#764ba2'
'hololoom-accent': '#f093fb'
```

### Responsive Breakpoints

- `sm`: 640px
- `md`: 768px
- `lg`: 1024px
- `xl`: 1280px

### Custom Animations

- `pulse-slow`: 3s pulse
- `spinner`: Loading spinner
- `node-pulse`: Graph node highlight

## Performance Optimization

### Code Splitting

Vite automatically splits vendor code:
- `react-vendor`: React, React DOM, React Router
- `d3-vendor`: D3.js
- `chart-vendor`: Chart.js, Recharts

### Build Optimization

```bash
npm run build
```

**Output**:
- Minified JS/CSS
- Source maps (for debugging)
- Tree-shaking (unused code removed)
- Lazy loading (route-based)

### Performance Tips

1. **Lazy Load Routes**:
   ```jsx
   const Dashboard = lazy(() => import('./views/DashboardView'))
   ```

2. **Memoize Expensive Computations**:
   ```jsx
   const layoutNodes = useMemo(() =>
     layout.layout(nodes, edges),
     [nodes, edges]
   )
   ```

3. **Debounce API Calls**:
   ```jsx
   const debouncedSearch = useDebounce(searchTerm, 300)
   ```

## Troubleshooting

### WebSocket Connection Failed

**Problem**: Cannot connect to `ws://localhost:8000/ws`

**Solutions**:
1. Ensure backend is running: `uvicorn HoloLoom.server.agentic_api:app --reload`
2. Check CORS settings in backend
3. Verify WebSocket endpoint exists
4. Check browser console for errors

### Graph Not Rendering

**Problem**: Yarn Graph shows empty canvas

**Solutions**:
1. Check `/graph/data` endpoint returns data
2. Verify D3.js is installed: `npm list d3`
3. Check browser console for errors
4. Reduce `maxNodes` filter (50 → 10)

### Charts Not Displaying

**Problem**: Performance/Historical charts not showing

**Solutions**:
1. Verify Chart.js installation: `npm list chart.js`
2. Check data format (must be array)
3. Ensure container has height (`style={{ height: '300px' }}`)

### Build Errors

**Problem**: `npm run build` fails

**Solutions**:
1. Clear node_modules: `rm -rf node_modules && npm install`
2. Clear cache: `rm -rf dist .vite`
3. Update dependencies: `npm update`
4. Check for TypeScript errors (if using TS)

## Development

### Hot Module Replacement (HMR)

Vite provides instant HMR:
- Edit files → see changes immediately
- No full page reload
- State preservation

### Debugging

**React DevTools**:
1. Install browser extension
2. Inspect component hierarchy
3. View props/state
4. Track re-renders

**Redux DevTools** (if using Redux):
1. Install extension
2. Time-travel debugging
3. Action history

### Linting

```bash
npm run lint
```

**ESLint Config**:
- React recommended rules
- React Hooks rules
- React Refresh rules

## Deployment

### Static Hosting

Build and deploy to static hosts:

```bash
# Build
npm run build

# Deploy to Netlify
netlify deploy --prod --dir=dist

# Deploy to Vercel
vercel --prod

# Deploy to GitHub Pages
npm run build && gh-pages -d dist
```

### Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

```bash
docker build -t hololoom-dashboard .
docker run -p 80:80 hololoom-dashboard
```

### Environment Variables

Production `.env`:

```env
VITE_API_URL=https://api.hololoom.com
VITE_WS_URL=wss://api.hololoom.com/ws
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

**Code Style**:
- Follow ESLint rules
- Use Prettier for formatting
- Write meaningful commit messages
- Add tests for new features

## License

This project is part of HoloLoom and follows the same license.

## Support

- **Documentation**: See `CLAUDE.md` in repository root
- **Issues**: Open GitHub issue
- **Discussions**: GitHub Discussions

## Roadmap

### Phase 1 ✅ (Complete)
- Dashboard view with system status
- Query interface
- Yarn Graph visualization
- Real-time monitoring

### Phase 2 🚧 (In Progress)
- Advanced graph filtering
- Custom metrics dashboards
- Export capabilities (PDF, PNG)
- Dark mode

### Phase 3 📋 (Planned)
- Multi-user collaboration
- Workflow builder integration
- Advanced analytics
- Mobile responsive design

### Phase 4 📋 (Future)
- AI-powered insights
- Predictive analytics
- Custom plugins
- Embedding library

---

**Built with ❤️ by the HoloLoom Team**

**Last Updated**: November 2025
