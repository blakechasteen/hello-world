/**
 * Promptly Dashboard - Main Application
 *
 * Real-time dashboard for monitoring and controlling the Promptly Matrix Bot
 * with HoloLoom integration.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import axios from 'axios';
import { WeavingVisualizer } from './components/WeavingVisualizer';
import { KnowledgeGraphExplorer } from './components/KnowledgeGraphExplorer';
import { AuditTrailBrowser } from './components/AuditTrailBrowser';
import { TeamCollaborationUI } from './components/TeamCollaborationUI';
import { WorkflowBuilder } from './components/WorkflowBuilder';
import { WeavingCycle, WeavingStep, StepStatus, WSMessage, KnowledgeGraph, GraphNode } from './types';
import { Activity, Wifi, WifiOff, Network, TrendingUp, FileText, Users, GitBranch } from 'lucide-react';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

type TabType = 'weaving' | 'graph' | 'stats' | 'audit' | 'team' | 'workflow';

function App() {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [currentCycle, setCurrentCycle] = useState<WeavingCycle | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<TabType>('weaving');
  const [knowledgeGraph, setKnowledgeGraph] = useState<KnowledgeGraph | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io(WS_URL, {
      transports: ['websocket'],
      path: '/ws',
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    });

    newSocket.on('connected', (data: any) => {
      console.log('Server message:', data);
    });

    newSocket.on('weaving_start', (data: any) => {
      console.log('Weaving started:', data);
      // Initialize new weaving cycle
      setCurrentCycle({
        query_id: data.query_id,
        query_text: data.query_text,
        timestamp: data.timestamp,
        steps: [],
        total_latency_ms: 0,
        confidence: 0,
      });
    });

    newSocket.on('weaving_update', (data: any) => {
      console.log('Weaving update:', data);
      // Update step status
      setCurrentCycle((prev) => {
        if (!prev || prev.query_id !== data.query_id) return prev;

        const existingStepIndex = prev.steps.findIndex(s => s.step === data.step);

        if (existingStepIndex >= 0) {
          // Update existing step
          const newSteps = [...prev.steps];
          newSteps[existingStepIndex] = {
            step: data.step,
            name: data.name,
            status: data.status as StepStatus,
            latency_ms: data.latency_ms,
            metadata: data.metadata,
          };
          return { ...prev, steps: newSteps };
        } else {
          // Add new step
          return {
            ...prev,
            steps: [
              ...prev.steps,
              {
                step: data.step,
                name: data.name,
                status: data.status as StepStatus,
                latency_ms: data.latency_ms,
                metadata: data.metadata,
              },
            ].sort((a, b) => a.step - b.step),
          };
        }
      });
    });

    newSocket.on('weaving_complete', (data: any) => {
      console.log('Weaving complete:', data);
      setCurrentCycle((prev) => {
        if (!prev || prev.query_id !== data.query_id) return prev;

        return {
          ...prev,
          response: data.response,
          confidence: data.confidence,
          tool_used: data.tool_used,
          total_latency_ms: data.latency_ms,
        };
      });

      // Refresh stats
      fetchStats();
    });

    newSocket.on('weaving_error', (data: any) => {
      console.error('Weaving error:', data);
      setCurrentCycle((prev) => {
        if (!prev || prev.query_id !== data.query_id) return prev;

        // Mark current step as error
        const currentStepIndex = prev.steps.findIndex(s => s.status === StepStatus.IN_PROGRESS);
        if (currentStepIndex >= 0) {
          const newSteps = [...prev.steps];
          newSteps[currentStepIndex].status = StepStatus.ERROR;
          newSteps[currentStepIndex].error = data.error;
          return { ...prev, steps: newSteps };
        }

        return prev;
      });
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  // Fetch stats on mount
  useEffect(() => {
    fetchStats();
    fetchGraph();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const fetchGraph = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/graph`);
      if (response.data.success) {
        setKnowledgeGraph(response.data.data);
      }
    } catch (error) {
      console.error('Failed to fetch graph:', error);
    }
  };

  const handleNodeClick = (node: GraphNode) => {
    console.log('Node clicked:', node);
    // Could expand this to show more details or filter related queries
  };

  const handleQuerySubmit = async (query: string) => {
    try {
      const response = await axios.post(`${API_URL}/api/query`, {
        text: query,
        user_id: '@dashboard:matrix.org',
        complexity: 'FAST',
      });

      console.log('Query response:', response.data);
    } catch (error) {
      console.error('Query failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-blue-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Promptly Dashboard</h1>
                <p className="text-sm text-gray-500">HoloLoom Integration Monitor</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                {connected ? (
                  <>
                    <Wifi className="w-5 h-5 text-green-500" />
                    <span className="text-sm text-green-600 font-medium">Connected</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5 text-red-500" />
                    <span className="text-sm text-red-600 font-medium">Disconnected</span>
                  </>
                )}
              </div>

              {/* Stats */}
              {stats && (
                <div className="text-sm text-gray-600">
                  <span className="font-medium">{stats.query_stats?.total_queries || 0}</span> queries processed
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('weaving')}
              className={`
                flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === 'weaving'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <TrendingUp className="w-5 h-5" />
              Weaving Visualizer
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={`
                flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === 'graph'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <Network className="w-5 h-5" />
              Knowledge Graph
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`
                flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === 'stats'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <Activity className="w-5 h-5" />
              Statistics
            </button>
            <button
              onClick={() => setActiveTab('audit')}
              className={`
                flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === 'audit'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <FileText className="w-5 h-5" />
              Audit Trail
            </button>
            <button
              onClick={() => setActiveTab('team')}
              className={`
                flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === 'team'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <Users className="w-5 h-5" />
              Team
            </button>
            <button
              onClick={() => setActiveTab('workflow')}
              className={`
                flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm
                ${activeTab === 'workflow'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }
              `}
            >
              <GitBranch className="w-5 h-5" />
              Workflows
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="grid grid-cols-1 gap-6">
          {/* Weaving Tab */}
          {activeTab === 'weaving' && (
            <WeavingVisualizer
              weavingCycle={currentCycle || undefined}
              onQuerySubmit={handleQuerySubmit}
            />
          )}

          {/* Knowledge Graph Tab */}
          {activeTab === 'graph' && knowledgeGraph && (
            <KnowledgeGraphExplorer
              graph={knowledgeGraph}
              onNodeClick={handleNodeClick}
            />
          )}

          {/* Stats Tab */}
          {activeTab === 'stats' && stats && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">System Statistics</h2>

              <div className="grid grid-cols-3 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm text-blue-600 mb-1">Total Queries</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {stats.query_stats?.total_queries || 0}
                  </p>
                </div>

                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-sm text-green-600 mb-1">Avg Latency</p>
                  <p className="text-2xl font-bold text-green-900">
                    {(stats.query_stats?.avg_latency_ms || 0).toFixed(1)}ms
                  </p>
                </div>

                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="text-sm text-purple-600 mb-1">Avg Confidence</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {((stats.query_stats?.avg_confidence || 0) * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {stats.bot_stats && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <p className="text-sm text-gray-600">
                    Config Mode: <span className="font-medium">{stats.bot_stats.config_mode}</span>
                  </p>
                  <p className="text-sm text-gray-600 mt-1">
                    Memory Shards: <span className="font-medium">{stats.bot_stats.memory_shards}</span>
                  </p>
                </div>
              )}

              {/* Recent Queries */}
              {stats.query_stats?.recent_queries && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Recent Queries</h3>
                  <div className="space-y-2">
                    {stats.query_stats.recent_queries.slice(0, 5).map((q: any, i: number) => (
                      <div key={i} className="p-3 bg-gray-50 rounded border border-gray-200">
                        <p className="text-sm font-medium text-gray-900 truncate">{q.query_text}</p>
                        <div className="mt-1 flex items-center gap-4 text-xs text-gray-600">
                          <span>Confidence: {(q.confidence * 100).toFixed(0)}%</span>
                          <span>Latency: {q.latency_ms.toFixed(0)}ms</span>
                          <span>Tool: {q.tool_used}</span>
                          {q.cache_hit && <span className="text-green-600">⚡ Cached</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Audit Trail Tab */}
          {activeTab === 'audit' && (
            <AuditTrailBrowser />
          )}

          {/* Team Collaboration Tab */}
          {activeTab === 'team' && (
            <TeamCollaborationUI />
          )}

          {/* Workflow Builder Tab */}
          {activeTab === 'workflow' && (
            <div className="h-[calc(100vh-250px)]">
              <WorkflowBuilder />
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 pb-6 text-center text-sm text-gray-500">
        <p>Promptly Matrix Bot • Powered by HoloLoom • Phase 4 Dashboard</p>
      </footer>
    </div>
  );
}

export default App;
