import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import Header from '@components/Layout/Header';
import Sidebar from '@components/Layout/Sidebar';
import Dashboard from '@components/dashboard/Dashboard';
import { useAppStore } from '@stores/appStore';
/**
 * Root App Component
 *
 * Manages the main layout and integration between HoloLoom backend.
 * Implements:
 * - Dark mode by default (enforced at HTML root)
 * - Responsive sidebar layout
 * - Real-time connection to Agent Manager backend (port 8002)
 * - Global state management via Zustand
 *
 * Architecture:
 * Header (navigation, status)
 *   ├─ Sidebar (agent list, controls)
 *   └─ Main (dashboard, metrics, logs)
 */
function App() {
    const [isConnected, setIsConnected] = useState(false);
    const [connectionError, setConnectionError] = useState(null);
    const { setAgents, addLog } = useAppStore();
    useEffect(() => {
        // Establish WebSocket connection to Agent Manager backend
        const connectToBackend = async () => {
            try {
                // Test API connectivity first
                const response = await fetch('/api/health');
                if (!response.ok) {
                    throw new Error('Backend health check failed');
                }
                setIsConnected(true);
                setConnectionError(null);
                addLog('Backend', 'Connected to HoloLoom backend', 'success');
                // Initialize agents
                await fetchAgents();
            }
            catch (error) {
                const message = error instanceof Error ? error.message : 'Failed to connect to backend';
                setConnectionError(message);
                addLog('Backend', `Connection error: ${message}`, 'error');
                setIsConnected(false);
                // Retry connection after 3 seconds
                const timeout = setTimeout(connectToBackend, 3000);
                return () => clearTimeout(timeout);
            }
            return undefined;
        };
        const fetchAgents = async () => {
            try {
                const response = await fetch('/agents/list');
                if (!response.ok)
                    throw new Error('Failed to fetch agents');
                const agents = await response.json();
                setAgents(agents);
            }
            catch (error) {
                const message = error instanceof Error ? error.message : 'Unknown error';
                addLog('Agents', `Failed to load agents: ${message}`, 'warning');
            }
        };
        connectToBackend();
    }, [setAgents, addLog]);
    return (_jsxs("div", { className: "h-screen w-screen flex flex-col bg-surface-primary overflow-hidden dark", children: [!isConnected && connectionError && (_jsx("div", { className: "bg-red-950 border-b border-red-700 px-4 py-3 text-red-200 text-sm", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("span", { children: ["\u26A0\uFE0F ", connectionError] }), _jsx("span", { className: "text-xs", children: "Retrying..." })] }) })), _jsx(Header, { isConnected: isConnected }), _jsxs("div", { className: "flex flex-1 overflow-hidden gap-0", children: [_jsx(Sidebar, {}), _jsx(Dashboard, {})] })] }));
}
export default App;
//# sourceMappingURL=App.js.map