'use client';

import { Button, Badge } from '@hololoom/design-system';
import type { Workflow } from '../../app/workflow/page';

interface WorkflowToolbarProps {
  workflow: Workflow;
  isExecuting: boolean;
  onExecute: () => void;
  onSave: () => void;
  onLoad: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onNameChange: (name: string) => void;
  onTogglePalette: () => void;
  onToggleInspector: () => void;
  showPalette: boolean;
  showInspector: boolean;
}

export function WorkflowToolbar({
  workflow,
  isExecuting,
  onExecute,
  onSave,
  onLoad,
  onNameChange,
  onTogglePalette,
  onToggleInspector,
  showPalette,
  showInspector,
}: WorkflowToolbarProps) {
  return (
    <div className="flex-shrink-0 h-14 px-4 border-b border-border-primary bg-bg-primary flex items-center justify-between">
      {/* Left: Workflow Info */}
      <div className="flex items-center gap-4">
        <input
          type="text"
          value={workflow.name}
          onChange={(e) => onNameChange(e.target.value)}
          className="text-lg font-semibold text-fg-primary bg-transparent border-none focus:outline-none focus:ring-0 w-64"
          placeholder="Workflow Name"
        />
        <Badge variant="secondary" size="sm">
          {workflow.nodes.length} nodes
        </Badge>
        <Badge variant="secondary" size="sm">
          {workflow.connections.length} connections
        </Badge>
      </div>

      {/* Center: Actions */}
      <div className="flex items-center gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={onTogglePalette}
          className={showPalette ? 'bg-cosmic-nebula/20' : ''}
        >
          <PaletteIcon />
          <span className="ml-1.5">Agents</span>
        </Button>

        <Button
          variant="secondary"
          size="sm"
          onClick={onToggleInspector}
          className={showInspector ? 'bg-cosmic-nebula/20' : ''}
        >
          <InspectorIcon />
          <span className="ml-1.5">Inspector</span>
        </Button>

        <div className="w-px h-6 bg-border-primary mx-2" />

        <label>
          <input
            type="file"
            accept=".json"
            onChange={onLoad}
            className="hidden"
          />
          <Button variant="secondary" size="sm" as="span" className="cursor-pointer">
            <UploadIcon />
            <span className="ml-1.5">Load</span>
          </Button>
        </label>

        <Button variant="secondary" size="sm" onClick={onSave}>
          <DownloadIcon />
          <span className="ml-1.5">Save</span>
        </Button>

        <div className="w-px h-6 bg-border-primary mx-2" />

        <Button
          variant="default"
          size="sm"
          onClick={onExecute}
          disabled={isExecuting || workflow.nodes.length === 0}
        >
          {isExecuting ? (
            <>
              <LoadingSpinner />
              <span className="ml-1.5">Executing...</span>
            </>
          ) : (
            <>
              <PlayIcon />
              <span className="ml-1.5">Execute</span>
            </>
          )}
        </Button>
      </div>

      {/* Right: Status */}
      <div className="flex items-center gap-3 text-sm text-fg-tertiary">
        <span>
          Last saved: {workflow.updatedAt.toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}

function PaletteIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="7" height="7" x="3" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="14" rx="1" />
      <rect width="7" height="7" x="3" y="14" rx="1" />
    </svg>
  );
}

function InspectorIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="6 3 20 12 6 21 6 3" />
    </svg>
  );
}

function LoadingSpinner() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}
