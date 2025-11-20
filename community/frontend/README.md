# Community Platform - Frontend

Modern React + TypeScript frontend for the Community Platform.

## Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool (fast HMR)
- **Redux Toolkit** - State management
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Tailwind CSS** - Utility-first styling
- **Socket.IO Client** - Real-time features

## Project Structure

```
src/
├── components/       # Reusable UI components
├── pages/           # Page components (routes)
├── store/           # Redux store and slices
│   ├── authSlice.ts    # Authentication state
│   ├── postsSlice.ts   # Posts state
│   └── index.ts        # Store configuration
├── api/             # API client
│   └── client.ts       # Axios HTTP client
├── hooks/           # Custom React hooks
│   └── redux.ts        # Typed Redux hooks
├── utils/           # Utility functions
├── types/           # TypeScript type definitions
│   └── index.ts        # API types
├── App.tsx          # Main app component
├── main.tsx         # Entry point
└── index.css        # Global styles
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend server running on http://localhost:8000

### Installation

```bash
cd community/frontend
npm install
```

### Development

```bash
# Start development server (with HMR)
npm run dev

# App will be available at http://localhost:3000
```

### Building for Production

```bash
# Type check
npm run type-check

# Build
npm run build

# Preview production build
npm run preview
```

## Features Implemented

### Core Architecture

✅ **TypeScript Configuration**
- Strict type checking
- Path aliases (@/components, @/store, etc.)
- Type-safe API client

✅ **API Client** (`src/api/client.ts`)
- Axios-based HTTP client
- Automatic JWT token management
- Request/response interceptors
- Token refresh on 401 errors

✅ **Redux Store** (`src/store/`)
- Authentication slice (login, register, logout)
- Posts slice (CRUD operations, voting)
- TypeScript-safe hooks (useAppDispatch, useAppSelector)

✅ **React Router**
- Protected routes
- Authentication flow
- Client-side navigation

✅ **Tailwind CSS**
- Utility-first styling
- Responsive design
- Custom color palette

## API Integration

The frontend connects to the backend API at `http://localhost:8000/api/v1`.

### Authentication Flow

1. User registers/logs in
2. Backend returns JWT access + refresh tokens
3. Tokens stored in localStorage
4. All subsequent requests include Authorization header
5. Automatic token refresh on expiration

### State Management

Redux Toolkit manages global state:

```typescript
// Login example
import { useAppDispatch } from '@/hooks/redux';
import { loginUser } from '@/store/authSlice';

const dispatch = useAppDispatch();
await dispatch(loginUser({ username, password }));
```

## Development Roadmap

### Phase 1: Core UI (Completed)
- ✅ Project setup (Vite, TypeScript, Tailwind)
- ✅ API client with authentication
- ✅ Redux store (auth, posts)
- ✅ Basic routing

### Phase 2: Pages & Components (Next)
- Login/Register pages with forms
- Home feed with post list
- Post detail page with comments
- Community pages
- User profile pages
- Create post/comment forms

### Phase 3: Real-Time Features
- WebSocket connection (Socket.IO)
- Live notifications
- Typing indicators
- Real-time post updates

### Phase 4: Advanced Features
- Markdown editor for posts
- Image upload
- Infinite scroll
- Search functionality
- Dark mode

## Configuration

### Environment Variables

Create `.env` file:

```
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=http://localhost:8001
```

### Proxy Configuration

Vite dev server proxies API requests to avoid CORS:

```typescript
// vite.config.ts
proxy: {
  '/api': 'http://localhost:8000',
  '/socket.io': 'http://localhost:8001',
}
```

## TypeScript Types

All API types match the backend Pydantic models:

- `User` - User profile and authentication
- `Post` - Post with author, community, votes
- `Comment` - Threaded comments with path
- `Community` - Community with members
- `AuthResponse` - Login/register response
- Request/Response types for all endpoints

## Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Contributing

1. Follow TypeScript strict mode
2. Use functional components with hooks
3. Follow Tailwind CSS utilities
4. Keep components small and focused
5. Write meaningful commit messages

## License

MIT
