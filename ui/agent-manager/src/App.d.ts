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
declare function App(): import("react/jsx-runtime").JSX.Element;
export default App;
//# sourceMappingURL=App.d.ts.map