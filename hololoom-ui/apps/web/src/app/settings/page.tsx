'use client';

import { useState } from 'react';
import { Navigation } from '../../components/Navigation';
import { Button, Card, Badge } from '@hololoom/design-system';

type Theme = 'tufte' | 'modern' | 'terminal';
type SafetyLevel = 'relaxed' | 'standard' | 'strict';

interface SettingsState {
  // Appearance
  theme: Theme;
  cosmicOverlay: boolean;
  compactMode: boolean;
  animationsEnabled: boolean;

  // API Configuration
  apiEndpoint: string;
  apiTimeout: number;
  enableStreaming: boolean;

  // Memory Settings
  cacheEnabled: boolean;
  cacheMaxSize: number;
  retentionDays: number;
  autoConsolidate: boolean;

  // Safety Settings
  safetyLevel: SafetyLevel;
  humanInLoopRequired: boolean;
  auditLogging: boolean;
  maxRiskTolerance: number;

  // Learning Settings
  enableLearning: boolean;
  explorationRate: number;
  learningUpdateInterval: number;

  // Notifications
  notifyOnHighRisk: boolean;
  notifyOnLowConfidence: boolean;
  notifyOnErrors: boolean;
}

const defaultSettings: SettingsState = {
  theme: 'modern',
  cosmicOverlay: true,
  compactMode: false,
  animationsEnabled: true,
  apiEndpoint: 'http://localhost:8000',
  apiTimeout: 30000,
  enableStreaming: true,
  cacheEnabled: true,
  cacheMaxSize: 10000,
  retentionDays: 30,
  autoConsolidate: true,
  safetyLevel: 'standard',
  humanInLoopRequired: true,
  auditLogging: true,
  maxRiskTolerance: 0.7,
  enableLearning: true,
  explorationRate: 0.1,
  learningUpdateInterval: 60,
  notifyOnHighRisk: true,
  notifyOnLowConfidence: true,
  notifyOnErrors: true,
};

export default function SettingsPage() {
  const [settings, setSettings] = useState<SettingsState>(defaultSettings);
  const [activeSection, setActiveSection] = useState('appearance');
  const [hasChanges, setHasChanges] = useState(false);

  const updateSetting = <K extends keyof SettingsState>(
    key: K,
    value: SettingsState[K]
  ) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    // In production, save to localStorage or backend
    localStorage.setItem('hololoom-settings', JSON.stringify(settings));
    setHasChanges(false);
  };

  const handleReset = () => {
    setSettings(defaultSettings);
    setHasChanges(true);
  };

  const sections = [
    { id: 'appearance', label: 'Appearance', icon: <PaletteIcon /> },
    { id: 'api', label: 'API Configuration', icon: <ServerIcon /> },
    { id: 'memory', label: 'Memory', icon: <DatabaseIcon /> },
    { id: 'safety', label: 'Safety', icon: <ShieldIcon /> },
    { id: 'learning', label: 'Learning', icon: <BrainIcon /> },
    { id: 'notifications', label: 'Notifications', icon: <BellIcon /> },
  ];

  return (
    <div className="min-h-screen bg-bg-primary">
      <Navigation />

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-fg-primary">Settings</h1>
            <p className="text-fg-tertiary mt-1">
              Configure your HoloLoom experience
            </p>
          </div>
          <div className="flex items-center gap-3">
            {hasChanges && (
              <Badge variant="warning" size="sm">
                Unsaved changes
              </Badge>
            )}
            <Button variant="ghost" onClick={handleReset}>
              Reset to Defaults
            </Button>
            <Button
              variant="primary"
              onClick={handleSave}
              disabled={!hasChanges}
            >
              Save Changes
            </Button>
          </div>
        </div>

        <div className="flex gap-8">
          {/* Sidebar Navigation */}
          <nav className="w-56 flex-shrink-0">
            <div className="sticky top-8 space-y-1">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-left transition-colors ${
                    activeSection === section.id
                      ? 'bg-cosmic-nebula/10 text-cosmic-nebula'
                      : 'text-fg-secondary hover:bg-bg-secondary hover:text-fg-primary'
                  }`}
                >
                  <span className="w-5 h-5">{section.icon}</span>
                  {section.label}
                </button>
              ))}
            </div>
          </nav>

          {/* Settings Content */}
          <div className="flex-1 space-y-6">
            {/* Appearance Settings */}
            {activeSection === 'appearance' && (
              <SettingsSection
                title="Appearance"
                description="Customize the look and feel of HoloLoom"
              >
                <SettingItem
                  label="Theme"
                  description="Choose your preferred color scheme"
                >
                  <div className="flex gap-3">
                    {(['tufte', 'modern', 'terminal'] as Theme[]).map(
                      (theme) => (
                        <ThemeButton
                          key={theme}
                          theme={theme}
                          isActive={settings.theme === theme}
                          onClick={() => updateSetting('theme', theme)}
                        />
                      )
                    )}
                  </div>
                </SettingItem>

                <SettingItem
                  label="Cosmic Overlay"
                  description="Enable subtle cosmic background effects"
                >
                  <Toggle
                    checked={settings.cosmicOverlay}
                    onChange={(v) => updateSetting('cosmicOverlay', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Compact Mode"
                  description="Use smaller spacing and font sizes"
                >
                  <Toggle
                    checked={settings.compactMode}
                    onChange={(v) => updateSetting('compactMode', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Animations"
                  description="Enable smooth transitions and animations"
                >
                  <Toggle
                    checked={settings.animationsEnabled}
                    onChange={(v) => updateSetting('animationsEnabled', v)}
                  />
                </SettingItem>
              </SettingsSection>
            )}

            {/* API Configuration */}
            {activeSection === 'api' && (
              <SettingsSection
                title="API Configuration"
                description="Configure connection to HoloLoom backend"
              >
                <SettingItem
                  label="API Endpoint"
                  description="Base URL for the HoloLoom API"
                >
                  <input
                    type="text"
                    value={settings.apiEndpoint}
                    onChange={(e) => updateSetting('apiEndpoint', e.target.value)}
                    className="w-full max-w-md px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-fg-primary focus:outline-none focus:border-cosmic-nebula"
                  />
                </SettingItem>

                <SettingItem
                  label="Request Timeout"
                  description="Maximum time to wait for API responses (ms)"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={5000}
                      max={120000}
                      step={1000}
                      value={settings.apiTimeout}
                      onChange={(e) =>
                        updateSetting('apiTimeout', parseInt(e.target.value))
                      }
                      className="w-48"
                    />
                    <span className="text-sm text-fg-secondary w-16">
                      {(settings.apiTimeout / 1000).toFixed(0)}s
                    </span>
                  </div>
                </SettingItem>

                <SettingItem
                  label="Streaming Responses"
                  description="Enable real-time response streaming"
                >
                  <Toggle
                    checked={settings.enableStreaming}
                    onChange={(v) => updateSetting('enableStreaming', v)}
                  />
                </SettingItem>
              </SettingsSection>
            )}

            {/* Memory Settings */}
            {activeSection === 'memory' && (
              <SettingsSection
                title="Memory Settings"
                description="Configure knowledge storage and caching"
              >
                <SettingItem
                  label="Query Caching"
                  description="Cache query results for faster responses"
                >
                  <Toggle
                    checked={settings.cacheEnabled}
                    onChange={(v) => updateSetting('cacheEnabled', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Cache Size"
                  description="Maximum number of cached queries"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={1000}
                      max={50000}
                      step={1000}
                      value={settings.cacheMaxSize}
                      onChange={(e) =>
                        updateSetting('cacheMaxSize', parseInt(e.target.value))
                      }
                      className="w-48"
                    />
                    <span className="text-sm text-fg-secondary w-20">
                      {settings.cacheMaxSize.toLocaleString()}
                    </span>
                  </div>
                </SettingItem>

                <SettingItem
                  label="Memory Retention"
                  description="How long to keep memories before archiving"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={7}
                      max={365}
                      step={1}
                      value={settings.retentionDays}
                      onChange={(e) =>
                        updateSetting('retentionDays', parseInt(e.target.value))
                      }
                      className="w-48"
                    />
                    <span className="text-sm text-fg-secondary w-20">
                      {settings.retentionDays} days
                    </span>
                  </div>
                </SettingItem>

                <SettingItem
                  label="Auto Consolidation"
                  description="Automatically consolidate memories over time"
                >
                  <Toggle
                    checked={settings.autoConsolidate}
                    onChange={(v) => updateSetting('autoConsolidate', v)}
                  />
                </SettingItem>
              </SettingsSection>
            )}

            {/* Safety Settings */}
            {activeSection === 'safety' && (
              <SettingsSection
                title="Safety Settings"
                description="Configure safety guardrails and oversight"
              >
                <SettingItem
                  label="Safety Level"
                  description="Overall strictness of safety guardrails"
                >
                  <div className="flex gap-3">
                    {(['relaxed', 'standard', 'strict'] as SafetyLevel[]).map(
                      (level) => (
                        <button
                          key={level}
                          onClick={() => updateSetting('safetyLevel', level)}
                          className={`px-4 py-2 rounded-lg capitalize transition-colors ${
                            settings.safetyLevel === level
                              ? level === 'relaxed'
                                ? 'bg-safety-caution/20 text-safety-caution border border-safety-caution'
                                : level === 'standard'
                                ? 'bg-safety-safe/20 text-safety-safe border border-safety-safe'
                                : 'bg-safety-risky/20 text-safety-risky border border-safety-risky'
                              : 'bg-bg-secondary text-fg-secondary hover:bg-bg-tertiary'
                          }`}
                        >
                          {level}
                        </button>
                      )
                    )}
                  </div>
                </SettingItem>

                <SettingItem
                  label="Human-in-the-Loop"
                  description="Require human approval for high-risk actions"
                >
                  <Toggle
                    checked={settings.humanInLoopRequired}
                    onChange={(v) => updateSetting('humanInLoopRequired', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Audit Logging"
                  description="Log all decisions for compliance review"
                >
                  <Toggle
                    checked={settings.auditLogging}
                    onChange={(v) => updateSetting('auditLogging', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Risk Tolerance"
                  description="Maximum acceptable risk score for automatic actions"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={settings.maxRiskTolerance}
                      onChange={(e) =>
                        updateSetting(
                          'maxRiskTolerance',
                          parseFloat(e.target.value)
                        )
                      }
                      className="w-48"
                    />
                    <span className="text-sm text-fg-secondary w-16">
                      {(settings.maxRiskTolerance * 100).toFixed(0)}%
                    </span>
                  </div>
                </SettingItem>
              </SettingsSection>
            )}

            {/* Learning Settings */}
            {activeSection === 'learning' && (
              <SettingsSection
                title="Learning Settings"
                description="Configure adaptive learning behavior"
              >
                <SettingItem
                  label="Enable Learning"
                  description="Allow system to learn from interactions"
                >
                  <Toggle
                    checked={settings.enableLearning}
                    onChange={(v) => updateSetting('enableLearning', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Exploration Rate"
                  description="Thompson Sampling exploration parameter"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={0}
                      max={0.5}
                      step={0.01}
                      value={settings.explorationRate}
                      onChange={(e) =>
                        updateSetting(
                          'explorationRate',
                          parseFloat(e.target.value)
                        )
                      }
                      className="w-48"
                    />
                    <span className="text-sm text-fg-secondary w-16">
                      {(settings.explorationRate * 100).toFixed(0)}%
                    </span>
                  </div>
                </SettingItem>

                <SettingItem
                  label="Learning Update Interval"
                  description="How often to update learning models (seconds)"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={10}
                      max={300}
                      step={10}
                      value={settings.learningUpdateInterval}
                      onChange={(e) =>
                        updateSetting(
                          'learningUpdateInterval',
                          parseInt(e.target.value)
                        )
                      }
                      className="w-48"
                    />
                    <span className="text-sm text-fg-secondary w-16">
                      {settings.learningUpdateInterval}s
                    </span>
                  </div>
                </SettingItem>
              </SettingsSection>
            )}

            {/* Notification Settings */}
            {activeSection === 'notifications' && (
              <SettingsSection
                title="Notifications"
                description="Configure alerts and notifications"
              >
                <SettingItem
                  label="High Risk Alerts"
                  description="Notify when high-risk actions are detected"
                >
                  <Toggle
                    checked={settings.notifyOnHighRisk}
                    onChange={(v) => updateSetting('notifyOnHighRisk', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Low Confidence Alerts"
                  description="Notify when confidence drops below threshold"
                >
                  <Toggle
                    checked={settings.notifyOnLowConfidence}
                    onChange={(v) => updateSetting('notifyOnLowConfidence', v)}
                  />
                </SettingItem>

                <SettingItem
                  label="Error Notifications"
                  description="Notify when errors occur"
                >
                  <Toggle
                    checked={settings.notifyOnErrors}
                    onChange={(v) => updateSetting('notifyOnErrors', v)}
                  />
                </SettingItem>
              </SettingsSection>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <div className="p-6">
        <h2 className="text-xl font-semibold text-fg-primary mb-1">{title}</h2>
        <p className="text-sm text-fg-tertiary mb-6">{description}</p>
        <div className="space-y-6">{children}</div>
      </div>
    </Card>
  );
}

function SettingItem({
  label,
  description,
  children,
}: {
  label: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-8">
      <div className="flex-1">
        <h3 className="text-sm font-medium text-fg-primary">{label}</h3>
        <p className="text-sm text-fg-tertiary mt-0.5">{description}</p>
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative w-11 h-6 rounded-full transition-colors ${
        checked ? 'bg-cosmic-nebula' : 'bg-bg-tertiary'
      }`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-0'
        }`}
      />
    </button>
  );
}

function ThemeButton({
  theme,
  isActive,
  onClick,
}: {
  theme: Theme;
  isActive: boolean;
  onClick: () => void;
}) {
  const colors = {
    tufte: { bg: '#FFFEF0', fg: '#333', accent: '#B8860B' },
    modern: { bg: '#0D0D12', fg: '#F5F5F5', accent: '#7C3AED' },
    terminal: { bg: '#0C0C0C', fg: '#00FF00', accent: '#00FF00' },
  };

  return (
    <button
      onClick={onClick}
      className={`relative p-1 rounded-lg transition-all ${
        isActive
          ? 'ring-2 ring-cosmic-nebula ring-offset-2 ring-offset-bg-primary'
          : 'hover:opacity-80'
      }`}
    >
      <div
        className="w-24 h-16 rounded-md flex flex-col items-center justify-center border border-border-primary overflow-hidden"
        style={{ backgroundColor: colors[theme].bg }}
      >
        <div
          className="w-12 h-1 rounded mb-1"
          style={{ backgroundColor: colors[theme].accent }}
        />
        <div
          className="w-16 h-0.5 rounded mb-0.5"
          style={{ backgroundColor: colors[theme].fg, opacity: 0.5 }}
        />
        <div
          className="w-14 h-0.5 rounded"
          style={{ backgroundColor: colors[theme].fg, opacity: 0.3 }}
        />
      </div>
      <span className="block text-xs text-fg-secondary mt-1 capitalize">
        {theme}
      </span>
    </button>
  );
}

// Icons
function PaletteIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="13.5" cy="6.5" r="0.5" fill="currentColor" />
      <circle cx="17.5" cy="10.5" r="0.5" fill="currentColor" />
      <circle cx="8.5" cy="7.5" r="0.5" fill="currentColor" />
      <circle cx="6.5" cy="12.5" r="0.5" fill="currentColor" />
      <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.555C21.965 6.012 17.461 2 12 2z" />
    </svg>
  );
}

function ServerIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="20" height="8" x="2" y="2" rx="2" ry="2" />
      <rect width="20" height="8" x="2" y="14" rx="2" ry="2" />
      <line x1="6" x2="6.01" y1="6" y2="6" />
      <line x1="6" x2="6.01" y1="18" y2="18" />
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5V19A9 3 0 0 0 21 19V5" />
      <path d="M3 12A9 3 0 0 0 21 12" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
      <path d="M19.938 10.5a4 4 0 0 1 .585.396" />
      <path d="M6 18a4 4 0 0 1-1.967-.516" />
      <path d="M19.967 17.484A4 4 0 0 1 18 18" />
    </svg>
  );
}

function BellIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
      <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
    </svg>
  );
}
