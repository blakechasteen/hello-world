# HoloLoom Skill Marketplace Web UI

**Created**: November 22, 2025
**Purpose**: React-based web interface for skill discovery and management

## Overview

Modern, responsive web UI for browsing, searching, and managing HoloLoom skills. Built with React, TypeScript, and Vite.

## Features

- **Skill Browser**: Grid layout with search and filtering
- **Real-time Updates**: WebSocket integration for live install progress
- **Skill Management**: Install, upgrade, and remove skills
- **Ratings & Reviews**: Rate skills and view community ratings
- **Statistics Dashboard**: Marketplace analytics and metrics
- **Responsive Design**: Works on desktop and mobile

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **TanStack Query** - Data fetching and caching
- **Axios** - HTTP client
- **Zustand** - State management (if needed)
- **Lucide React** - Icons

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.12+ with FastAPI backend running

### Installation

```bash
# Install dependencies
cd skills/marketplace/web
npm install
```

### Development

```bash
# Start dev server (http://localhost:3000)
npm run dev
```

This will start the Vite dev server with:
- Hot module replacement (HMR)
- Proxy to FastAPI backend (http://localhost:8000)
- WebSocket proxy for real-time updates

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Architecture

```
React Frontend (Port 3000)
    ↓ HTTP/WebSocket
FastAPI Backend (Port 8000)
    ↓
SkillCatalog + PackageManager
```

### Components

**Main Components**:
- `<SkillBrowser />` - Main marketplace view
- `<SkillCard />` - Individual skill display
- `<SearchBar />` - Search and filter interface

**API Client**:
- `MarketplaceClient` - Type-safe API wrapper
- WebSocket support for real-time updates

### State Management

Uses TanStack Query for server state:
- Automatic caching and invalidation
- Optimistic updates
- Background refetching

```typescript
const { data: skills } = useQuery({
  queryKey: ['skills', category, sortBy],
  queryFn: () => marketplaceClient.listSkills(category, sortBy),
});
```

## Usage

### Starting the Full Stack

**Terminal 1 - Backend**:
```bash
cd skills/marketplace
python api.py
```

**Terminal 2 - Frontend**:
```bash
cd web
npm run dev
```

**Access**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### WebSocket Integration

The UI automatically connects to WebSocket for real-time updates:

```typescript
// Listen for install progress
marketplaceClient.onWebSocketMessage('install_complete', (msg) => {
  console.log(`Installed: ${msg.skill_id}`);
  // Refresh skills list
});
```

## Configuration

### API Endpoint

Update `src/api/client.ts` if backend runs on different port:

```typescript
const client = new MarketplaceClient('http://localhost:8000');
```

### CORS

Backend allows these origins by default:
- http://localhost:3000 (Vite default)
- http://localhost:5173 (Vite alternate)

## File Structure

```
web/
├── src/
│   ├── components/
│   │   ├── SkillBrowser.tsx
│   │   ├── SkillCard.tsx
│   │   └── SearchBar.tsx
│   ├── api/
│   │   └── client.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── styles.css
├── package.json
├── vite.config.ts
├── tsconfig.json
└── index.html
```

## Development Tips

### Hot Module Replacement

Vite provides instant updates without full page reload:
- Edit components → See changes immediately
- Preserves React state during updates

### TypeScript

All API types are defined in `client.ts`:
- `Skill` - Skill metadata
- `SearchResult` - Search result with relevance score
- `InstallResponse` - Installation result

### Debugging

**React DevTools**:
```bash
# Install browser extension for React debugging
```

**Network Tab**:
- View API requests/responses
- Monitor WebSocket messages

## Troubleshooting

**Backend not responding**:
```bash
# Check if backend is running
curl http://localhost:8000/health
```

**WebSocket not connecting**:
- Check CORS settings in backend
- Verify proxy configuration in vite.config.ts

**Build errors**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Performance

**Optimizations**:
- React.memo for expensive components
- TanStack Query caching (30s stale time)
- Lazy loading for large skill lists
- WebSocket for real-time updates (avoid polling)

**Bundle Size**:
- Production build: ~150KB gzipped
- Code splitting by route (future)

## Future Enhancements

- [ ] Skill comparison view
- [ ] Advanced filters (requirements, rating, etc.)
- [ ] Installation wizard with dependency tree
- [ ] Skill ratings and reviews UI
- [ ] User authentication and profiles
- [ ] Favorite/bookmark skills
- [ ] Dark/light theme toggle

## License

Part of HoloLoom project.
