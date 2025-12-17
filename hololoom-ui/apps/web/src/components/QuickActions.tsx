'use client';

import { Card, CardHeader, CardBody, Button, Badge } from '@hololoom/design-system';

interface QuickAction {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  href: string;
  badge?: string;
  gradient: string;
}

const quickActions: QuickAction[] = [
  {
    id: 'chat',
    title: 'Start Chat',
    description: 'Ask questions and get intelligent answers',
    icon: <ChatIcon />,
    href: '/chat',
    gradient: 'from-cosmic-nebula/20 to-cosmic-nebula/5',
  },
  {
    id: 'memory',
    title: 'Explore Memory',
    description: 'Navigate your knowledge graph',
    icon: <MemoryIcon />,
    href: '/memory',
    gradient: 'from-cosmic-aurora/20 to-cosmic-aurora/5',
  },
  {
    id: 'workflow',
    title: 'Build Workflow',
    description: 'Create visual multi-agent pipelines',
    icon: <WorkflowIcon />,
    href: '/workflows',
    badge: 'Visual',
    gradient: 'from-cosmic-starlight/20 to-cosmic-starlight/5',
  },
  {
    id: 'safety',
    title: 'Safety Dashboard',
    description: 'Monitor alignment and safety metrics',
    icon: <SafetyIcon />,
    href: '/safety',
    badge: 'Live',
    gradient: 'from-confidence-high/20 to-confidence-high/5',
  },
  {
    id: 'analytics',
    title: 'View Analytics',
    description: 'Track performance and insights',
    icon: <AnalyticsIcon />,
    href: '/analytics',
    gradient: 'from-cosmic-nebula/20 to-cosmic-aurora/5',
  },
  {
    id: 'ingest',
    title: 'Ingest Content',
    description: 'Add documents, code, or media',
    icon: <IngestIcon />,
    href: '/ingest',
    badge: '47 formats',
    gradient: 'from-cosmic-starlight/20 to-cosmic-nebula/5',
  },
];

export function QuickActions() {
  return (
    <div>
      <h2 className="text-2xl font-bold text-center mb-4">Quick Actions</h2>
      <p className="text-fg-secondary text-center max-w-2xl mx-auto mb-10">
        Jump into HoloLoom's powerful features. Each action leverages the full
        multi-scale memory system with Thompson Sampling optimization.
      </p>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {quickActions.map((action) => (
          <a
            key={action.id}
            href={action.href}
            className="group block"
          >
            <Card
              variant="interactive"
              className={`h-full bg-gradient-to-br ${action.gradient} hover:scale-[1.02] transition-transform`}
            >
              <CardHeader className="flex flex-row items-start justify-between">
                <div className="w-10 h-10 rounded-lg bg-bg-secondary/50 flex items-center justify-center text-fg-secondary group-hover:text-cosmic-nebula transition-colors">
                  {action.icon}
                </div>
                {action.badge && (
                  <Badge variant="secondary" size="sm">
                    {action.badge}
                  </Badge>
                )}
              </CardHeader>
              <CardBody className="pt-0">
                <h3 className="font-semibold text-fg-primary mb-1 group-hover:text-cosmic-nebula transition-colors">
                  {action.title}
                </h3>
                <p className="text-sm text-fg-tertiary">
                  {action.description}
                </p>
              </CardBody>
            </Card>
          </a>
        ))}
      </div>

      {/* Secondary actions */}
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <Button variant="ghost" size="sm" onClick={() => window.location.href = '/docs'}>
          <DocsIcon />
          <span>Documentation</span>
        </Button>
        <Button variant="ghost" size="sm" onClick={() => window.location.href = '/api'}>
          <ApiIcon />
          <span>API Reference</span>
        </Button>
        <Button variant="ghost" size="sm" onClick={() => window.location.href = '/settings'}>
          <SettingsIcon />
          <span>Settings</span>
        </Button>
      </div>
    </div>
  );
}

// Icon components
function ChatIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4c0-1.1.9-2 2-2h8a2 2 0 0 1 2 2v5Z" />
      <path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1" />
    </svg>
  );
}

function MemoryIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4" />
      <path d="M12 18v4" />
      <path d="m4.93 4.93 2.83 2.83" />
      <path d="m16.24 16.24 2.83 2.83" />
      <path d="M2 12h4" />
      <path d="M18 12h4" />
      <path d="m4.93 19.07 2.83-2.83" />
      <path d="m16.24 7.76 2.83-2.83" />
    </svg>
  );
}

function WorkflowIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="8" height="4" x="8" y="2" rx="1" ry="1" />
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
      <path d="M9 12h6" />
      <path d="M9 16h6" />
    </svg>
  );
}

function SafetyIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function AnalyticsIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 3v18h18" />
      <path d="m19 9-5 5-4-4-3 3" />
    </svg>
  );
}

function IngestIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}

function DocsIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="mr-2"
    >
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  );
}

function ApiIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="mr-2"
    >
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="mr-2"
    >
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}
