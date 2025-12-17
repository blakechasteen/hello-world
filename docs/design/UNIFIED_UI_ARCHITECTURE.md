# HoloLoom Unified UI Architecture

> **Version**: 1.0.0 (December 2025)
> **Philosophy**: "Write once, weave everywhere"

---

## Executive Summary

This document defines the architecture for a **unified UI platform** that exposes all 140+ HoloLoom features across Web, Desktop, VS Code, and Mobile—with no compromises.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Core Framework** | React + TypeScript | Existing library (`ui/hololoom-components`), cross-platform via Electron/Capacitor |
| **State Management** | Redux Toolkit + RTK Query | Already in use, excellent DevTools, predictable |
| **Styling** | Tailwind + CSS Variables | Design tokens, theme switching, excellent DX |
| **Build System** | Turborepo Monorepo | Shared packages, incremental builds, caching |
| **Desktop** | Electron (via Tauri later) | Quick MVP, can migrate to Tauri for performance |
| **Mobile** | Capacitor PWA | Share React code, native when needed |
| **VS Code** | Webview + Extension API | Webview uses same React components |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HoloLoom Unified UI                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │    Web      │ │  Desktop    │ │   VS Code   │ │   Mobile    │  │
│  │  (Browser)  │ │ (Electron)  │ │ (Extension) │ │    (PWA)    │  │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘  │
│         │               │               │               │          │
│         └───────────────┴───────┬───────┴───────────────┘          │
│                                 │                                   │
│  ┌──────────────────────────────┴──────────────────────────────┐   │
│  │                    @hololoom/ui-core                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │Components│ │  Hooks   │ │  State   │ │  Utils   │       │   │
│  │  │  Library │ │ Library  │ │  Slices  │ │ Library  │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  └──────────────────────────────┬──────────────────────────────┘   │
│                                 │                                   │
│  ┌──────────────────────────────┴──────────────────────────────┐   │
│  │                  @hololoom/design-system                     │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │  Tokens  │ │  Themes  │ │Tailwind  │ │  Icons   │       │   │
│  │  │  (JSON)  │ │   (CSS)  │ │  Plugin  │ │  (SVG)   │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                         Backend Layer                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   FastAPI + WebSocket                         │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │  │
│  │  │Agentic  │ │  RAG    │ │ Safety  │ │ Memory  │ │Learning│ │  │
│  │  │  API    │ │  API    │ │  API    │ │  API    │ │  API   │ │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Monorepo Structure

```
hololoom-ui/
├── apps/
│   ├── web/                    # Main web application
│   │   ├── src/
│   │   │   ├── app/           # Next.js app router
│   │   │   ├── features/      # Feature-specific code
│   │   │   └── pages/         # Page components
│   │   ├── public/
│   │   └── package.json
│   │
│   ├── desktop/                # Electron application
│   │   ├── src/
│   │   │   ├── main/          # Electron main process
│   │   │   ├── preload/       # Preload scripts
│   │   │   └── renderer/      # Uses @hololoom/ui-core
│   │   └── package.json
│   │
│   ├── vscode/                 # VS Code extension
│   │   ├── src/
│   │   │   ├── extension.ts   # Extension entry
│   │   │   ├── webview/       # Webview panels (React)
│   │   │   ├── views/         # TreeView providers
│   │   │   └── commands/      # Extension commands
│   │   └── package.json
│   │
│   └── mobile/                 # Capacitor PWA
│       ├── src/
│       │   ├── app/           # React app
│       │   └── native/        # Native plugins
│       ├── ios/               # iOS project
│       ├── android/           # Android project
│       └── package.json
│
├── packages/
│   ├── design-system/          # @hololoom/design-system
│   │   ├── tokens/
│   │   │   ├── colors.json
│   │   │   ├── typography.json
│   │   │   ├── spacing.json
│   │   │   └── index.ts
│   │   ├── themes/
│   │   │   ├── tufte.css
│   │   │   ├── modern.css
│   │   │   ├── terminal.css
│   │   │   └── cosmic.css     # Overlay applied to all
│   │   ├── tailwind/
│   │   │   └── preset.ts      # Tailwind preset
│   │   ├── icons/
│   │   │   └── index.tsx      # Icon components
│   │   └── package.json
│   │
│   ├── ui-core/                # @hololoom/ui-core
│   │   ├── components/
│   │   │   ├── primitives/    # Button, Input, Card, etc.
│   │   │   ├── thread/        # Thread, ThreadList
│   │   │   ├── panel/         # LoomPanel, PanelGroup
│   │   │   ├── confidence/    # ConfidenceIndicator, Gauge
│   │   │   ├── safety/        # SafetyGate, SafetyBadge
│   │   │   ├── memory/        # MemoryNavigator, GraphView
│   │   │   ├── chat/          # ChatInterface, MessageList
│   │   │   ├── workflow/      # WorkflowCanvas, AgentNode
│   │   │   ├── analytics/     # Dashboard panels
│   │   │   └── index.ts
│   │   ├── hooks/
│   │   │   ├── useHoloLoom.ts # Main API hook
│   │   │   ├── useSafety.ts
│   │   │   ├── useMemory.ts
│   │   │   ├── useStreaming.ts
│   │   │   └── index.ts
│   │   ├── store/
│   │   │   ├── slices/
│   │   │   ├── middleware/
│   │   │   └── index.ts
│   │   └── package.json
│   │
│   ├── api-client/             # @hololoom/api-client
│   │   ├── client.ts          # HTTP/WS client
│   │   ├── types.ts           # API types
│   │   ├── hooks.ts           # RTK Query hooks
│   │   └── package.json
│   │
│   └── shared/                 # @hololoom/shared
│       ├── types/             # Shared TypeScript types
│       ├── utils/             # Shared utilities
│       └── package.json
│
├── tools/
│   ├── token-generator/       # Generate tokens from Figma
│   └── icon-generator/        # Generate icon components
│
├── turbo.json                  # Turborepo config
├── package.json                # Workspace root
├── tsconfig.base.json          # Base TypeScript config
└── README.md
```

---

## Package Details

### @hololoom/design-system

The source of truth for all visual design.

```typescript
// packages/design-system/tokens/index.ts

export const colors = {
  cosmic: {
    void: '#0a0a0f',
    voidSoft: '#12121a',
    starlight: '#e8e4ff',
    nebula: '#6366f1',
    aurora: '#22d3ee',
  },
  semantic: {
    confidence: {
      high: '#22c55e',
      good: '#84cc16',
      moderate: '#eab308',
      low: '#f97316',
      veryLow: '#ef4444',
    },
    safety: {
      safe: '#22c55e',
      caution: '#eab308',
      warning: '#f97316',
      danger: '#ef4444',
      critical: '#dc2626',
    },
  },
} as const;

export const typography = {
  fontFamily: {
    sans: "'Inter', sans-serif",
    mono: "'JetBrains Mono', monospace",
    display: "'Satoshi', 'Inter', sans-serif",
    cosmic: "'Cinzel', Georgia, serif",
  },
  fontSize: {
    xs: '0.64rem',
    sm: '0.8rem',
    base: '1rem',
    md: '1.25rem',
    lg: '1.563rem',
    xl: '1.953rem',
    '2xl': '2.441rem',
    '3xl': '3.052rem',
  },
} as const;

export const spacing = {
  px: '1px',
  0: '0',
  0.5: '0.125rem',
  1: '0.25rem',
  2: '0.5rem',
  3: '0.75rem',
  4: '1rem',
  5: '1.25rem',
  6: '1.5rem',
  8: '2rem',
  10: '2.5rem',
  12: '3rem',
  16: '4rem',
  20: '5rem',
  24: '6rem',
} as const;
```

### Tailwind Preset

```typescript
// packages/design-system/tailwind/preset.ts

import { colors, typography, spacing } from '../tokens';

export const holoLoomPreset = {
  theme: {
    extend: {
      colors: {
        cosmic: colors.cosmic,
        confidence: colors.semantic.confidence,
        safety: colors.semantic.safety,
      },
      fontFamily: typography.fontFamily,
      fontSize: typography.fontSize,
      spacing: spacing,
    },
  },
  plugins: [
    // Custom plugin for theme switching
    function ({ addBase, addUtilities }) {
      addBase({
        ':root': {
          '--theme': 'modern',
        },
      });

      addUtilities({
        '.theme-tufte': { '--theme': 'tufte' },
        '.theme-modern': { '--theme': 'modern' },
        '.theme-terminal': { '--theme': 'terminal' },
      });
    },
  ],
};
```

### @hololoom/ui-core

The shared component library.

```typescript
// packages/ui-core/components/index.ts

// Primitives
export { Button } from './primitives/Button';
export { Input } from './primitives/Input';
export { Card } from './primitives/Card';
export { Badge } from './primitives/Badge';
export { Avatar } from './primitives/Avatar';
export { Tooltip } from './primitives/Tooltip';
export { Dialog } from './primitives/Dialog';
export { DropdownMenu } from './primitives/DropdownMenu';
export { Command } from './primitives/Command'; // Command palette

// Thread System
export { Thread } from './thread/Thread';
export { ThreadList } from './thread/ThreadList';
export { ThreadConnection } from './thread/ThreadConnection';

// Panel System
export { LoomPanel } from './panel/LoomPanel';
export { PanelGroup } from './panel/PanelGroup';
export { ResizablePanel } from './panel/ResizablePanel';

// Confidence
export { ConfidenceIndicator } from './confidence/ConfidenceIndicator';
export { ConfidenceGauge } from './confidence/ConfidenceGauge';
export { ConfidenceTrajectory } from './confidence/ConfidenceTrajectory';

// Safety (Priority #1)
export { SafetyGate } from './safety/SafetyGate';
export { SafetyBadge } from './safety/SafetyBadge';
export { SafetyDashboard } from './safety/SafetyDashboard';
export { AuditTrail } from './safety/AuditTrail';
export { RiskIndicator } from './safety/RiskIndicator';

// Chat (Priority #2)
export { ChatInterface } from './chat/ChatInterface';
export { MessageList } from './chat/MessageList';
export { Message } from './chat/Message';
export { ChatInput } from './chat/ChatInput';
export { ReasoningChain } from './chat/ReasoningChain';
export { SourceAttribution } from './chat/SourceAttribution';

// Memory (Priority #3)
export { MemoryNavigator } from './memory/MemoryNavigator';
export { KnowledgeGraph } from './memory/KnowledgeGraph';
export { TimeTravel } from './memory/TimeTravel';
export { PatternDiscovery } from './memory/PatternDiscovery';

// Workflow (Priority #5)
export { WorkflowCanvas } from './workflow/WorkflowCanvas';
export { AgentNode } from './workflow/AgentNode';
export { AgentPalette } from './workflow/AgentPalette';
export { ConnectionLine } from './workflow/ConnectionLine';

// Analytics (Priority #7)
export { AnalyticsDashboard } from './analytics/AnalyticsDashboard';
export { MetricCard } from './analytics/MetricCard';
export { CacheGauge } from './analytics/CacheGauge';
export { StageWaterfall } from './analytics/StageWaterfall';
export { LearningStats } from './analytics/LearningStats';
```

### Component Example: SafetyGate

```tsx
// packages/ui-core/components/safety/SafetyGate.tsx

import { cva, type VariantProps } from 'class-variance-authority';
import { AlertTriangle, CheckCircle, XCircle, AlertOctagon } from 'lucide-react';
import { Button } from '../primitives/Button';
import { cn } from '../../utils/cn';

const safetyGateVariants = cva(
  'relative rounded-lg border-2 p-4 backdrop-blur-sm transition-all duration-300',
  {
    variants: {
      level: {
        low: 'border-safety-safe/30 bg-safety-safe/5',
        medium: 'border-safety-caution/30 bg-safety-caution/5',
        high: 'border-safety-warning/30 bg-safety-warning/5',
        critical: 'border-safety-critical/30 bg-safety-critical/5 animate-pulse-subtle',
      },
      status: {
        pending: 'opacity-100',
        approved: 'opacity-70',
        blocked: 'opacity-90',
        escalated: 'ring-2 ring-offset-2 ring-safety-critical',
      },
    },
    defaultVariants: {
      level: 'medium',
      status: 'pending',
    },
  }
);

const levelIcons = {
  low: CheckCircle,
  medium: AlertTriangle,
  high: AlertTriangle,
  critical: AlertOctagon,
};

const levelColors = {
  low: 'text-safety-safe',
  medium: 'text-safety-caution',
  high: 'text-safety-warning',
  critical: 'text-safety-critical',
};

export interface SafetyGateProps
  extends VariantProps<typeof safetyGateVariants> {
  action: string;
  reason?: string;
  requiresApproval?: boolean;
  onApprove?: () => void;
  onBlock?: () => void;
  onEscalate?: () => void;
  auditTrailId?: string;
  className?: string;
}

export function SafetyGate({
  level = 'medium',
  status = 'pending',
  action,
  reason,
  requiresApproval = false,
  onApprove,
  onBlock,
  onEscalate,
  auditTrailId,
  className,
}: SafetyGateProps) {
  const Icon = levelIcons[level || 'medium'];
  const iconColor = levelColors[level || 'medium'];

  return (
    <div
      className={cn(safetyGateVariants({ level, status }), className)}
      role="alert"
      aria-live="assertive"
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <Icon className={cn('h-5 w-5 mt-0.5', iconColor)} aria-hidden />
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-sm">
            Safety Gate: {level?.toUpperCase()}
          </h4>
          <p className="text-sm text-muted-foreground mt-1">
            Action: <code className="font-mono">{action}</code>
          </p>
          {reason && (
            <p className="text-sm text-muted-foreground mt-2">{reason}</p>
          )}
        </div>
      </div>

      {/* Actions */}
      {requiresApproval && status === 'pending' && (
        <div className="flex items-center gap-2 mt-4">
          <Button
            variant="default"
            size="sm"
            onClick={onApprove}
            className="bg-safety-safe hover:bg-safety-safe/90"
          >
            <CheckCircle className="h-4 w-4 mr-1" />
            Approve
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onBlock}
            className="border-safety-danger text-safety-danger hover:bg-safety-danger/10"
          >
            <XCircle className="h-4 w-4 mr-1" />
            Block
          </Button>
          {level === 'critical' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onEscalate}
              className="text-safety-critical"
            >
              Escalate
            </Button>
          )}
        </div>
      )}

      {/* Audit Trail Link */}
      {auditTrailId && (
        <div className="mt-3 pt-3 border-t border-current/10">
          <a
            href={`/audit/${auditTrailId}`}
            className="text-xs text-muted-foreground hover:underline"
          >
            View audit trail: {auditTrailId.slice(0, 8)}...
          </a>
        </div>
      )}
    </div>
  );
}
```

### @hololoom/api-client

Unified API access across all platforms.

```typescript
// packages/api-client/client.ts

import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const holoLoomApi = createApi({
  reducerPath: 'holoLoomApi',
  baseQuery: fetchBaseQuery({
    baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  }),
  tagTypes: ['Memory', 'Safety', 'Workflow', 'Analytics'],
  endpoints: (builder) => ({
    // === SAFETY (Priority #1) ===
    getSafetyStatus: builder.query<SafetyStatus, void>({
      query: () => '/safety/status',
      providesTags: ['Safety'],
    }),

    getAuditTrail: builder.query<AuditEntry[], AuditQuery>({
      query: (params) => ({
        url: '/safety/audit',
        params,
      }),
      providesTags: ['Safety'],
    }),

    gateAction: builder.mutation<GateResult, GateRequest>({
      query: (body) => ({
        url: '/safety/gate',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['Safety'],
    }),

    // === CHAT/RAG (Priority #2) ===
    query: builder.mutation<QueryResponse, QueryRequest>({
      query: (body) => ({
        url: '/query',
        method: 'POST',
        body,
      }),
    }),

    streamQuery: builder.query<void, QueryRequest>({
      queryFn: () => ({ data: undefined }),
      async onCacheEntryAdded(
        arg,
        { updateCachedData, cacheDataLoaded, cacheEntryRemoved }
      ) {
        // WebSocket streaming implementation
      },
    }),

    // === MEMORY (Priority #3) ===
    getMemories: builder.query<Memory[], MemoryQuery>({
      query: (params) => ({
        url: '/memory/recall',
        params,
      }),
      providesTags: ['Memory'],
    }),

    navigateMemory: builder.mutation<NavigationResult, NavigationRequest>({
      query: (body) => ({
        url: '/memory/navigate',
        method: 'POST',
        body,
      }),
    }),

    timeTravel: builder.query<MemorySnapshot, string>({
      query: (timestamp) => `/memory/time-travel?timestamp=${timestamp}`,
    }),

    // === WORKFLOWS (Priority #5) ===
    executeWorkflow: builder.mutation<WorkflowResult, Workflow>({
      query: (body) => ({
        url: '/workflow/execute',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['Workflow'],
    }),

    // === ANALYTICS (Priority #7) ===
    getAnalytics: builder.query<Analytics, AnalyticsQuery>({
      query: (params) => ({
        url: '/analytics',
        params,
      }),
      providesTags: ['Analytics'],
    }),
  }),
});

export const {
  // Safety
  useGetSafetyStatusQuery,
  useGetAuditTrailQuery,
  useGateActionMutation,
  // Chat
  useQueryMutation,
  // Memory
  useGetMemoriesQuery,
  useNavigateMemoryMutation,
  useTimeTravelQuery,
  // Workflow
  useExecuteWorkflowMutation,
  // Analytics
  useGetAnalyticsQuery,
} = holoLoomApi;
```

---

## Platform-Specific Implementations

### Web Application (Next.js)

```typescript
// apps/web/src/app/layout.tsx

import { ThemeProvider } from '@hololoom/ui-core';
import { StoreProvider } from './store-provider';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <StoreProvider>
          <ThemeProvider
            defaultTheme="modern"
            themes={['tufte', 'modern', 'terminal']}
            cosmicOverlay={true}
          >
            {children}
          </ThemeProvider>
        </StoreProvider>
      </body>
    </html>
  );
}
```

### Desktop Application (Electron)

```typescript
// apps/desktop/src/main/main.ts

import { app, BrowserWindow, ipcMain } from 'electron';
import { join } from 'path';

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 768,
    frame: false, // Custom titlebar
    titleBarStyle: 'hiddenInset', // macOS
    webPreferences: {
      preload: join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Load the web app
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'));
  }
}

// Desktop-specific features
ipcMain.handle('system:tray', async (event, action) => {
  // System tray management
});

ipcMain.handle('file:open', async (event, options) => {
  // Native file dialog
});

ipcMain.handle('notification:show', async (event, notification) => {
  // Native notifications
});
```

### VS Code Extension

```typescript
// apps/vscode/src/extension.ts

import * as vscode from 'vscode';
import { SafetyWebviewProvider } from './webview/SafetyWebviewProvider';
import { MemoryTreeProvider } from './views/MemoryTreeProvider';
import { HoloLoomCommands } from './commands';

export function activate(context: vscode.ExtensionContext) {
  // Register webview providers (React components)
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'hololoom.safetyDashboard',
      new SafetyWebviewProvider(context.extensionUri)
    )
  );

  // Register tree views
  const memoryTree = new MemoryTreeProvider();
  vscode.window.registerTreeDataProvider('hololoom.memoryExplorer', memoryTree);

  // Register commands
  const commands = new HoloLoomCommands(context);
  context.subscriptions.push(
    vscode.commands.registerCommand('hololoom.query', commands.query),
    vscode.commands.registerCommand('hololoom.openSafety', commands.openSafety),
    vscode.commands.registerCommand('hololoom.navigateMemory', commands.navigateMemory),
    vscode.commands.registerCommand('hololoom.runWorkflow', commands.runWorkflow)
  );

  // Status bar
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.text = '$(shield) HoloLoom';
  statusBar.command = 'hololoom.openSafety';
  statusBar.show();
}

// Webview Provider using React components
class SafetyWebviewProvider implements vscode.WebviewViewProvider {
  constructor(private readonly extensionUri: vscode.Uri) {}

  resolveWebviewView(webviewView: vscode.WebviewView) {
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.extensionUri],
    };

    // Load the bundled React app
    webviewView.webview.html = this.getHtml(webviewView.webview);
  }

  private getHtml(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'dist', 'webview.js')
    );
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, 'dist', 'webview.css')
    );

    return `<!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <link href="${styleUri}" rel="stylesheet">
        </head>
        <body>
          <div id="root"></div>
          <script src="${scriptUri}"></script>
        </body>
      </html>`;
  }
}
```

### Mobile Application (Capacitor)

```typescript
// apps/mobile/src/app/App.tsx

import { IonApp, IonRouterOutlet, setupIonicReact } from '@ionic/react';
import { IonReactRouter } from '@ionic/react-router';
import { Route } from 'react-router-dom';
import { ThemeProvider } from '@hololoom/ui-core';
import { StoreProvider } from './store';

// Pages (using shared components)
import { ChatPage } from './pages/Chat';
import { SafetyPage } from './pages/Safety';
import { MemoryPage } from './pages/Memory';
import { SettingsPage } from './pages/Settings';

// Native plugins
import { Haptics, ImpactStyle } from '@capacitor/haptics';
import { StatusBar, Style } from '@capacitor/status-bar';

setupIonicReact();

export function App() {
  // Setup native features
  useEffect(() => {
    StatusBar.setStyle({ style: Style.Dark });
  }, []);

  return (
    <IonApp>
      <StoreProvider>
        <ThemeProvider defaultTheme="modern" cosmicOverlay>
          <IonReactRouter>
            <IonRouterOutlet>
              <Route exact path="/" component={ChatPage} />
              <Route path="/safety" component={SafetyPage} />
              <Route path="/memory" component={MemoryPage} />
              <Route path="/settings" component={SettingsPage} />
            </IonRouterOutlet>
          </IonReactRouter>
        </ThemeProvider>
      </StoreProvider>
    </IonApp>
  );
}

// Mobile-specific hook for haptics
export function useHaptics() {
  return {
    impact: (style: ImpactStyle = ImpactStyle.Medium) => {
      Haptics.impact({ style });
    },
    notification: (type: 'success' | 'warning' | 'error') => {
      Haptics.notification({ type });
    },
  };
}
```

---

## State Management Architecture

### Store Structure

```typescript
// packages/ui-core/store/index.ts

import { configureStore } from '@reduxjs/toolkit';
import { holoLoomApi } from '@hololoom/api-client';

// Slices
import { safetySlice } from './slices/safetySlice';
import { chatSlice } from './slices/chatSlice';
import { memorySlice } from './slices/memorySlice';
import { workflowSlice } from './slices/workflowSlice';
import { analyticsSlice } from './slices/analyticsSlice';
import { uiSlice } from './slices/uiSlice';

export const createStore = (preloadedState?: Partial<RootState>) => {
  return configureStore({
    reducer: {
      // API
      [holoLoomApi.reducerPath]: holoLoomApi.reducer,

      // Feature slices
      safety: safetySlice.reducer,
      chat: chatSlice.reducer,
      memory: memorySlice.reducer,
      workflow: workflowSlice.reducer,
      analytics: analyticsSlice.reducer,

      // UI state
      ui: uiSlice.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware({
        serializableCheck: {
          ignoredActions: ['persist/PERSIST'],
        },
      }).concat(holoLoomApi.middleware),
    preloadedState,
  });
};

export type AppStore = ReturnType<typeof createStore>;
export type RootState = ReturnType<AppStore['getState']>;
export type AppDispatch = AppStore['dispatch'];
```

### Safety Slice (Priority #1)

```typescript
// packages/ui-core/store/slices/safetySlice.ts

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface SafetyState {
  // Current status
  overallLevel: 'safe' | 'caution' | 'warning' | 'danger' | 'critical';
  activeGates: SafetyGate[];
  pendingApprovals: number;

  // Audit trail
  recentAuditEntries: AuditEntry[];

  // Monitoring
  metrics: {
    actionsGated: number;
    actionsApproved: number;
    actionsBlocked: number;
    avgLatency: number;
  };

  // UI state
  dashboardOpen: boolean;
  selectedGateId: string | null;
}

const initialState: SafetyState = {
  overallLevel: 'safe',
  activeGates: [],
  pendingApprovals: 0,
  recentAuditEntries: [],
  metrics: {
    actionsGated: 0,
    actionsApproved: 0,
    actionsBlocked: 0,
    avgLatency: 0,
  },
  dashboardOpen: false,
  selectedGateId: null,
};

export const safetySlice = createSlice({
  name: 'safety',
  initialState,
  reducers: {
    setOverallLevel(state, action: PayloadAction<SafetyState['overallLevel']>) {
      state.overallLevel = action.payload;
    },

    addGate(state, action: PayloadAction<SafetyGate>) {
      state.activeGates.push(action.payload);
      state.pendingApprovals = state.activeGates.filter(
        (g) => g.status === 'pending' && g.requiresApproval
      ).length;
    },

    updateGate(state, action: PayloadAction<{ id: string; updates: Partial<SafetyGate> }>) {
      const gate = state.activeGates.find((g) => g.id === action.payload.id);
      if (gate) {
        Object.assign(gate, action.payload.updates);
      }
      state.pendingApprovals = state.activeGates.filter(
        (g) => g.status === 'pending' && g.requiresApproval
      ).length;
    },

    removeGate(state, action: PayloadAction<string>) {
      state.activeGates = state.activeGates.filter((g) => g.id !== action.payload);
    },

    addAuditEntry(state, action: PayloadAction<AuditEntry>) {
      state.recentAuditEntries.unshift(action.payload);
      if (state.recentAuditEntries.length > 100) {
        state.recentAuditEntries.pop();
      }
    },

    toggleDashboard(state) {
      state.dashboardOpen = !state.dashboardOpen;
    },

    selectGate(state, action: PayloadAction<string | null>) {
      state.selectedGateId = action.payload;
    },
  },
});

export const safetyActions = safetySlice.actions;
```

---

## Feature Implementation by Priority

### Priority #1: Safety Dashboard

**Files to create:**

```
packages/ui-core/components/safety/
├── SafetyDashboard.tsx      # Main dashboard layout
├── SafetyGate.tsx           # Individual gate component
├── SafetyBadge.tsx          # Status badge
├── AuditTrail.tsx           # Audit log viewer
├── RiskIndicator.tsx        # Risk level visualization
├── SafetyTimeline.tsx       # Temporal safety view
├── DeceptionAlert.tsx       # Deception detection
├── ConvergenceMonitor.tsx   # Instrumental convergence
└── index.ts
```

**Key features:**
- Real-time safety status overview
- Action gating with human-in-the-loop
- Audit trail with searchable history
- Risk level trending
- Deception detection alerts
- Convergence behavior monitoring

### Priority #2: Chat + RAG Interface

**Files to create:**

```
packages/ui-core/components/chat/
├── ChatInterface.tsx        # Main chat container
├── MessageList.tsx          # Message display
├── Message.tsx              # Individual message
├── ChatInput.tsx            # Input with mode selector
├── ReasoningChain.tsx       # Multi-step reasoning viz
├── SourceAttribution.tsx    # Source display
├── StreamingText.tsx        # Streaming text display
├── ModeSelector.tsx         # DIRECT/VERIFY/RESEARCH/PLAN
├── ConfidenceOverlay.tsx    # Message confidence
└── index.ts
```

**Key features:**
- Multi-modal input (text, voice, file)
- 4 reasoning modes with visual distinction
- Streaming responses with reasoning chain
- Source attribution with navigation
- Confidence indicators per message
- Safety integration (gates inline)

### Priority #3: Memory Navigator

**Files to create:**

```
packages/ui-core/components/memory/
├── MemoryNavigator.tsx      # Main navigation UI
├── KnowledgeGraph.tsx       # Force-directed graph
├── TimeTravel.tsx           # Temporal snapshots
├── PatternDiscovery.tsx     # Pattern visualization
├── MemoryList.tsx           # List view
├── MemoryDetail.tsx         # Single memory detail
├── NavigationControls.tsx   # Direction controls
├── ConstellationView.tsx    # Cluster visualization
└── index.ts
```

**Key features:**
- 4-direction navigation (forward/backward/sideways/deep)
- Interactive knowledge graph
- Time-travel with slider
- Pattern discovery (loops, clusters, resonance, threads)
- Multiple views (graph, list, timeline, constellation)

---

## Build & Development

### Turborepo Configuration

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "build/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": {},
    "test": {
      "dependsOn": ["build"]
    },
    "typecheck": {
      "dependsOn": ["^typecheck"]
    }
  }
}
```

### Development Commands

```bash
# Install dependencies
pnpm install

# Start all apps in development
pnpm dev

# Start specific app
pnpm dev --filter=web
pnpm dev --filter=desktop
pnpm dev --filter=vscode

# Build all packages
pnpm build

# Build specific package
pnpm build --filter=@hololoom/ui-core

# Type check
pnpm typecheck

# Lint
pnpm lint

# Test
pnpm test
```

---

## Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- [x] Design system specification
- [ ] Monorepo setup with Turborepo
- [ ] Design token pipeline
- [ ] Core primitives (Button, Input, Card, etc.)
- [ ] Theme switching infrastructure

### Phase 2: Safety First (Week 3-4) — Priority #1
- [ ] SafetyDashboard component
- [ ] SafetyGate with approval flow
- [ ] AuditTrail viewer
- [ ] API integration
- [ ] Real-time updates via WebSocket

### Phase 3: Chat + RAG (Week 5-6) — Priority #2
- [ ] ChatInterface with streaming
- [ ] ReasoningChain visualization
- [ ] Mode selector (4 modes)
- [ ] Source attribution
- [ ] Safety integration in chat

### Phase 4: Memory (Week 7-8) — Priority #3
- [ ] MemoryNavigator with 4 directions
- [ ] KnowledgeGraph (force-directed)
- [ ] TimeTravel interface
- [ ] PatternDiscovery UI

### Phase 5: Multi-Device (Week 9-10) — Priority #4
- [ ] Cross-device sync UI
- [ ] Device pairing flow
- [ ] Sync status indicators
- [ ] Conflict resolution

### Phase 6: Workflow (Week 11-12) — Priority #5
- [ ] WorkflowCanvas
- [ ] Agent palette (18 types)
- [ ] Connection drawing
- [ ] Execution visualization

### Phase 7: Voice + Analytics (Week 13-14) — Priority #6, #7
- [ ] Voice input integration
- [ ] Voice correction UI
- [ ] Analytics dashboard
- [ ] Real-time metrics

### Phase 8: Polish + AR (Week 15-16) — Priority #8
- [ ] AR foundation
- [ ] Cross-platform polish
- [ ] Performance optimization
- [ ] Accessibility audit

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time to First Meaningful Paint** | <1.5s | Lighthouse |
| **Time to Interactive** | <3s | Lighthouse |
| **Bundle Size (core)** | <150KB gzipped | Bundlewatch |
| **Accessibility Score** | >95 | Lighthouse |
| **Component Coverage** | 100% of features | Manual audit |
| **Platform Parity** | 95%+ feature parity | Feature matrix |
| **Test Coverage** | >80% | Jest/Vitest |

---

*"Write once, weave everywhere—the cosmic loom of unified experience."*
