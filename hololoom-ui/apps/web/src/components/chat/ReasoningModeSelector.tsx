'use client';

import { useState } from 'react';
import type { ReasoningMode } from '../../app/chat/page';

interface ReasoningModeSelectorProps {
  mode: ReasoningMode;
  onChange: (mode: ReasoningMode) => void;
}

interface ModeConfig {
  id: ReasoningMode;
  name: string;
  shortName: string;
  description: string;
  icon: React.ReactNode;
  latency: string;
  color: string;
}

const modes: ModeConfig[] = [
  {
    id: 'direct',
    name: 'Direct Answer',
    shortName: 'Direct',
    description: 'Single-pass answer for simple queries',
    icon: <DirectIcon />,
    latency: '~150ms',
    color: 'bg-safety-safe',
  },
  {
    id: 'verify',
    name: 'Verify & Confirm',
    shortName: 'Verify',
    description: 'Cross-reference with multiple sources',
    icon: <VerifyIcon />,
    latency: '~600ms',
    color: 'bg-cosmic-aurora',
  },
  {
    id: 'research',
    name: 'Deep Research',
    shortName: 'Research',
    description: 'Multi-query exploration and synthesis',
    icon: <ResearchIcon />,
    latency: '~900ms',
    color: 'bg-cosmic-nebula',
  },
  {
    id: 'plan_execute',
    name: 'Plan & Execute',
    shortName: 'Plan',
    description: 'Step-by-step task decomposition',
    icon: <PlanIcon />,
    latency: '~750ms',
    color: 'bg-cosmic-stardust',
  },
];

export function ReasoningModeSelector({ mode, onChange }: ReasoningModeSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const currentMode = modes.find((m) => m.id === mode) || modes[0];

  return (
    <div className="relative">
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-bg-secondary rounded-lg border border-border-primary hover:border-cosmic-nebula transition-colors"
      >
        <span className={`w-2 h-2 rounded-full ${currentMode.color}`} />
        <span className="text-sm font-medium text-fg-primary">
          {currentMode.shortName}
        </span>
        <ChevronIcon isOpen={isOpen} />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute right-0 top-full mt-2 w-72 bg-bg-primary rounded-xl border border-border-primary shadow-xl z-50 overflow-hidden">
            <div className="p-2">
              {modes.map((modeConfig) => (
                <button
                  key={modeConfig.id}
                  onClick={() => {
                    onChange(modeConfig.id);
                    setIsOpen(false);
                  }}
                  className={`w-full flex items-start gap-3 p-3 rounded-lg transition-colors ${
                    mode === modeConfig.id
                      ? 'bg-cosmic-nebula/10'
                      : 'hover:bg-bg-secondary'
                  }`}
                >
                  {/* Icon */}
                  <div
                    className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                      mode === modeConfig.id
                        ? 'bg-cosmic-nebula text-white'
                        : 'bg-bg-tertiary text-fg-tertiary'
                    }`}
                  >
                    {modeConfig.icon}
                  </div>

                  {/* Content */}
                  <div className="flex-1 text-left">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-fg-primary">
                        {modeConfig.name}
                      </span>
                      {mode === modeConfig.id && (
                        <CheckIcon />
                      )}
                    </div>
                    <p className="text-xs text-fg-tertiary mt-0.5">
                      {modeConfig.description}
                    </p>
                    <span className="text-xs text-fg-tertiary mt-1 inline-block">
                      ⚡ {modeConfig.latency}
                    </span>
                  </div>
                </button>
              ))}
            </div>

            {/* Footer */}
            <div className="border-t border-border-primary px-4 py-2 bg-bg-secondary/50">
              <p className="text-xs text-fg-tertiary">
                Select a reasoning mode based on your query complexity
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ChevronIcon({ isOpen }: { isOpen: boolean }) {
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
      className={`text-fg-tertiary transition-transform ${isOpen ? 'rotate-180' : ''}`}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function CheckIcon() {
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
      className="text-cosmic-nebula"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function DirectIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function VerifyIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ResearchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function PlanIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </svg>
  );
}
