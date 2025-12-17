/**
 * Core Types for Agent Manager UI
 *
 * These types correspond to HoloLoom's agentic reasoning system:
 * - Agent: Individual reasoning unit in the swarm
 * - Task: Work unit assigned to agents
 * - Message: Inter-agent communication
 * - Metrics: Performance tracking for agents and system
 */
/** Agent State Enum */
export var AgentState;
(function (AgentState) {
    AgentState["IDLE"] = "idle";
    AgentState["READY"] = "ready";
    AgentState["RUNNING"] = "running";
    AgentState["SUCCESS"] = "success";
    AgentState["WARNING"] = "warning";
    AgentState["ERROR"] = "error";
    AgentState["PAUSED"] = "paused";
})(AgentState || (AgentState = {}));
/** Agent Type */
export var AgentType;
(function (AgentType) {
    AgentType["QUERY"] = "query";
    AgentType["PROCESS"] = "process";
    AgentType["MEMORY"] = "memory";
    AgentType["DECISION"] = "decision";
    AgentType["OUTPUT"] = "output";
    AgentType["CONTROL"] = "control";
})(AgentType || (AgentType = {}));
/** Reasoning Mode for Agents */
export var ReasoningMode;
(function (ReasoningMode) {
    ReasoningMode["DIRECT"] = "direct";
    ReasoningMode["VERIFY"] = "verify";
    ReasoningMode["RESEARCH"] = "research";
    ReasoningMode["PLAN_EXECUTE"] = "plan_execute";
})(ReasoningMode || (ReasoningMode = {}));
/** Dashboard View Type */
export var DashboardView;
(function (DashboardView) {
    DashboardView["OVERVIEW"] = "overview";
    DashboardView["AGENTS"] = "agents";
    DashboardView["TASKS"] = "tasks";
    DashboardView["METRICS"] = "metrics";
    DashboardView["LOGS"] = "logs";
    DashboardView["SETTINGS"] = "settings";
})(DashboardView || (DashboardView = {}));
//# sourceMappingURL=index.js.map