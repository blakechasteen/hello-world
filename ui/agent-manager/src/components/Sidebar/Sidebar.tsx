import React from 'react';

/**
 * Sidebar Component
 * Configuration panel for creating new threads with:
 * - Agent Type selector (radio buttons)
 * - Reasoning Mode selector (radio buttons)
 * - Token Budget input (optional)
 * - Priority slider (1-10)
 * - Create Thread button
 *
 * Dark sidebar with subtle borders inspired by VS Code
 */
interface ThreadConfig {
  agentType: string;
  reasoningMode: 'DIRECT' | 'VERIFY' | 'RESEARCH' | 'PLAN_EXECUTE';
  tokenBudget?: number;
  priority: number;
}

export const Sidebar: React.FC = () => {
  const [config, setConfig] = React.useState<ThreadConfig>({
    agentType: 'weaving',
    reasoningMode: 'DIRECT',
    tokenBudget: undefined,
    priority: 5,
  });

  const agentTypes = [
    { value: 'weaving', label: 'Weaving', icon: '◆' },
    { value: 'rag', label: 'RAG', icon: '🔍' },
    { value: 'agentic', label: 'Agentic', icon: '🤖' },
    { value: 'custom', label: 'Custom', icon: '⚙' },
  ];

  const reasoningModes = [
    { value: 'DIRECT' as const, label: 'Direct', description: '~150ms' },
    { value: 'VERIFY' as const, label: 'Verify', description: '~600ms' },
    { value: 'RESEARCH' as const, label: 'Research', description: '~900ms' },
    {
      value: 'PLAN_EXECUTE' as const,
      label: 'Plan & Execute',
      description: '~750ms',
    },
  ];

  const handleCreateThread = () => {
    console.log('Creating thread with config:', config);
    // TODO: Implement thread creation via WebSocket
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800 p-4 gap-6 overflow-y-auto">
      {/* Section: Agent Type */}
      <div className="space-y-3">
        <div className="px-2 py-1.5">
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wide">
            Agent Type
          </label>
        </div>

        <div className="space-y-2">
          {agentTypes.map((type) => (
            <label
              key={type.value}
              className="flex items-center gap-3 p-2.5 rounded-md cursor-pointer hover:bg-slate-800 transition-colors group"
            >
              <input
                type="radio"
                name="agent-type"
                value={type.value}
                checked={config.agentType === type.value}
                onChange={(e) =>
                  setConfig({ ...config, agentType: e.target.value })
                }
                className="w-4 h-4 accent-emerald-600 cursor-pointer"
              />
              <div className="flex-1 flex items-center gap-2">
                <span className="text-lg">{type.icon}</span>
                <span className="text-sm font-medium text-slate-300 group-hover:text-slate-100">
                  {type.label}
                </span>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800" />

      {/* Section: Reasoning Mode */}
      <div className="space-y-3">
        <div className="px-2 py-1.5">
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wide">
            Reasoning Mode
          </label>
        </div>

        <div className="space-y-2">
          {reasoningModes.map((mode) => (
            <label
              key={mode.value}
              className="flex items-center gap-3 p-2.5 rounded-md cursor-pointer hover:bg-slate-800 transition-colors group"
            >
              <input
                type="radio"
                name="reasoning-mode"
                value={mode.value}
                checked={config.reasoningMode === mode.value}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    reasoningMode: e.target.value as ThreadConfig['reasoningMode'],
                  })
                }
                className="w-4 h-4 accent-emerald-600 cursor-pointer"
              />
              <div className="flex-1 flex flex-col gap-0.5">
                <span className="text-sm font-medium text-slate-300 group-hover:text-slate-100">
                  {mode.label}
                </span>
                <span className="text-xs text-slate-500 group-hover:text-slate-400">
                  {mode.description}
                </span>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800" />

      {/* Section: Token Budget */}
      <div className="space-y-3">
        <div className="px-2 py-1.5">
          <label htmlFor="token-budget" className="block text-xs font-semibold text-slate-300 uppercase tracking-wide">
            Token Budget (optional)
          </label>
        </div>

        <input
          id="token-budget"
          type="number"
          min="100"
          step="100"
          placeholder="e.g., 2000"
          value={config.tokenBudget ?? ''}
          onChange={(e) =>
            setConfig({
              ...config,
              tokenBudget: e.target.value ? parseInt(e.target.value) : undefined,
            })
          }
          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors"
        />
        <p className="text-xs text-slate-500 px-2">
          Leave empty for unlimited
        </p>
      </div>

      {/* Divider */}
      <div className="h-px bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800" />

      {/* Section: Priority Slider */}
      <div className="space-y-3">
        <div className="px-2 py-1.5 flex items-center justify-between">
          <label htmlFor="priority" className="block text-xs font-semibold text-slate-300 uppercase tracking-wide">
            Priority
          </label>
          <span className="text-sm font-semibold text-emerald-400">
            {config.priority}
          </span>
        </div>

        <div className="px-1 py-2">
          <input
            id="priority"
            type="range"
            min="1"
            max="10"
            value={config.priority}
            onChange={(e) =>
              setConfig({ ...config, priority: parseInt(e.target.value) })
            }
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-600"
          />
        </div>

        <div className="flex justify-between text-xs text-slate-500 px-2">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Create Thread Button */}
      <button
        onClick={handleCreateThread}
        className="w-full px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-md transition-colors flex items-center justify-center gap-2 text-sm"
      >
        <span>→</span>
        <span>Create Thread</span>
      </button>

      {/* Info Text */}
      <p className="text-xs text-slate-500 text-center px-2">
        WebSocket connection required
      </p>
    </div>
  );
};

export default Sidebar;
