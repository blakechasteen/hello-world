/**
 * Team Collaboration UI Component
 *
 * Features:
 * - Prompt library grid with search/filter
 * - Permission management (role-based access)
 * - Usage analytics dashboard
 * - Version history for prompts
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import {
  Users,
  BookOpen,
  Lock,
  TrendingUp,
  Search,
  Plus,
  Edit2,
  Trash2,
  Eye,
  Copy,
  Share2,
  Clock,
  Star,
  Filter,
  BarChart3,
  Shield,
  Check,
  X,
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

// Types
interface Prompt {
  id: string;
  title: string;
  content: string;
  author: string;
  scope: 'TEAM' | 'ROOM' | 'USER';
  created_at: string;
  updated_at: string;
  version: number;
  tags: string[];
  usage_count: number;
  avg_rating: number;
  permissions: {
    can_edit: boolean;
    can_delete: boolean;
    can_share: boolean;
  };
}

interface Permission {
  user_id: string;
  role: 'OWNER' | 'ADMIN' | 'EDITOR' | 'VIEWER';
  granted_at: string;
  granted_by: string;
}

interface UsageStats {
  total_prompts: number;
  total_usage: number;
  active_users: number;
  popular_prompts: Array<{
    prompt_id: string;
    title: string;
    usage_count: number;
  }>;
  usage_by_scope: {
    TEAM: number;
    ROOM: number;
    USER: number;
  };
  recent_activity: Array<{
    timestamp: string;
    user: string;
    action: string;
    prompt_title: string;
  }>;
}

type ViewMode = 'library' | 'permissions' | 'analytics';
type ScopeFilter = 'ALL' | 'TEAM' | 'ROOM' | 'USER';

const SCOPE_COLORS = {
  TEAM: 'bg-blue-100 text-blue-800',
  ROOM: 'bg-purple-100 text-purple-800',
  USER: 'bg-green-100 text-green-800',
};

const ROLE_COLORS = {
  OWNER: 'bg-red-100 text-red-800',
  ADMIN: 'bg-orange-100 text-orange-800',
  EDITOR: 'bg-blue-100 text-blue-800',
  VIEWER: 'bg-gray-100 text-gray-800',
};

export const TeamCollaborationUI: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('library');
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [loading, setLoading] = useState(false);

  // Library filters
  const [searchQuery, setSearchQuery] = useState('');
  const [scopeFilter, setScopeFilter] = useState<ScopeFilter>('ALL');
  const [selectedTags, setSelectedTags] = useState<Set<string>>(new Set());

  // Prompt editor
  const [editingPrompt, setEditingPrompt] = useState<Prompt | null>(null);
  const [showPromptEditor, setShowPromptEditor] = useState(false);

  // Permission editor
  const [showPermissionEditor, setShowPermissionEditor] = useState(false);
  const [selectedUser, setSelectedUser] = useState('');
  const [selectedRole, setSelectedRole] = useState<'OWNER' | 'ADMIN' | 'EDITOR' | 'VIEWER'>('VIEWER');

  useEffect(() => {
    fetchData();
  }, [viewMode]);

  const fetchData = async () => {
    setLoading(true);
    try {
      if (viewMode === 'library') {
        const response = await axios.get(`${API_URL}/api/prompts`);
        if (response.data.success) {
          setPrompts(response.data.data.prompts);
        }
      } else if (viewMode === 'permissions') {
        const response = await axios.get(`${API_URL}/api/permissions`);
        if (response.data.success) {
          setPermissions(response.data.data.permissions);
        }
      } else if (viewMode === 'analytics') {
        const response = await axios.get(`${API_URL}/api/usage`);
        if (response.data.success) {
          setUsageStats(response.data.data);
        }
      }
    } catch (error) {
      console.error('Failed to fetch data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreatePrompt = async () => {
    setEditingPrompt({
      id: '',
      title: '',
      content: '',
      author: '@current-user:matrix.org',
      scope: 'USER',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      version: 1,
      tags: [],
      usage_count: 0,
      avg_rating: 0,
      permissions: {
        can_edit: true,
        can_delete: true,
        can_share: true,
      },
    });
    setShowPromptEditor(true);
  };

  const handleSavePrompt = async () => {
    if (!editingPrompt) return;

    try {
      if (editingPrompt.id) {
        // Update existing
        await axios.put(`${API_URL}/api/prompts/${editingPrompt.id}`, editingPrompt);
      } else {
        // Create new
        await axios.post(`${API_URL}/api/prompts`, editingPrompt);
      }
      setShowPromptEditor(false);
      setEditingPrompt(null);
      fetchData();
    } catch (error) {
      console.error('Failed to save prompt:', error);
    }
  };

  const handleDeletePrompt = async (promptId: string) => {
    if (!confirm('Are you sure you want to delete this prompt?')) return;

    try {
      await axios.delete(`${API_URL}/api/prompts/${promptId}`);
      fetchData();
    } catch (error) {
      console.error('Failed to delete prompt:', error);
    }
  };

  const handleCopyPrompt = (content: string) => {
    navigator.clipboard.writeText(content);
    // TODO: Show toast notification
  };

  const handleGrantPermission = async () => {
    if (!selectedUser) return;

    try {
      await axios.post(`${API_URL}/api/permissions`, {
        user_id: selectedUser,
        role: selectedRole,
      });
      setShowPermissionEditor(false);
      setSelectedUser('');
      setSelectedRole('VIEWER');
      fetchData();
    } catch (error) {
      console.error('Failed to grant permission:', error);
    }
  };

  const handleRevokePermission = async (userId: string) => {
    if (!confirm('Are you sure you want to revoke this permission?')) return;

    try {
      await axios.delete(`${API_URL}/api/permissions/${userId}`);
      fetchData();
    } catch (error) {
      console.error('Failed to revoke permission:', error);
    }
  };

  // Filter prompts
  const filteredPrompts = prompts.filter((prompt) => {
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      if (
        !prompt.title.toLowerCase().includes(query) &&
        !prompt.content.toLowerCase().includes(query) &&
        !prompt.tags.some((tag) => tag.toLowerCase().includes(query))
      ) {
        return false;
      }
    }

    // Scope filter
    if (scopeFilter !== 'ALL' && prompt.scope !== scopeFilter) {
      return false;
    }

    // Tag filter
    if (selectedTags.size > 0) {
      if (!prompt.tags.some((tag) => selectedTags.has(tag))) {
        return false;
      }
    }

    return true;
  });

  // Get all unique tags
  const allTags = Array.from(new Set(prompts.flatMap((p) => p.tags)));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Team Collaboration</h2>
          <p className="text-sm text-gray-500">Manage shared prompts, permissions, and team analytics</p>
        </div>

        {/* View Mode Tabs */}
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('library')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'library'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            Library
          </button>
          <button
            onClick={() => setViewMode('permissions')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'permissions'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <Lock className="w-4 h-4" />
            Permissions
          </button>
          <button
            onClick={() => setViewMode('analytics')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'analytics'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            Analytics
          </button>
        </div>
      </div>

      {/* Library View */}
      {viewMode === 'library' && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-4">
              {/* Search */}
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search prompts by title, content, or tags..."
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Scope Filter */}
              <select
                value={scopeFilter}
                onChange={(e) => setScopeFilter(e.target.value as ScopeFilter)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="ALL">All Scopes</option>
                <option value="TEAM">Team</option>
                <option value="ROOM">Room</option>
                <option value="USER">User</option>
              </select>

              {/* Create Button */}
              <button
                onClick={handleCreatePrompt}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus className="w-4 h-4" />
                New Prompt
              </button>
            </div>

            {/* Tag Filter */}
            {allTags.length > 0 && (
              <div className="mt-3 flex items-center gap-2 flex-wrap">
                <Filter className="w-4 h-4 text-gray-400" />
                {allTags.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => {
                      const newTags = new Set(selectedTags);
                      if (newTags.has(tag)) {
                        newTags.delete(tag);
                      } else {
                        newTags.add(tag);
                      }
                      setSelectedTags(newTags);
                    }}
                    className={`px-2 py-1 text-xs rounded ${
                      selectedTags.has(tag)
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Prompts Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredPrompts.map((prompt) => (
              <div
                key={prompt.id}
                className="bg-white rounded-lg shadow hover:shadow-lg transition-shadow border border-gray-200"
              >
                <div className="p-4">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 mb-1">{prompt.title}</h3>
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 text-xs font-medium rounded ${SCOPE_COLORS[prompt.scope]}`}>
                          {prompt.scope}
                        </span>
                        {prompt.avg_rating > 0 && (
                          <div className="flex items-center gap-1 text-xs text-amber-600">
                            <Star className="w-3 h-3 fill-current" />
                            {prompt.avg_rating.toFixed(1)}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Content Preview */}
                  <p className="text-sm text-gray-600 mb-3 line-clamp-3">{prompt.content}</p>

                  {/* Tags */}
                  {prompt.tags.length > 0 && (
                    <div className="flex gap-1 flex-wrap mb-3">
                      {prompt.tags.map((tag) => (
                        <span key={tag} className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Meta */}
                  <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                    <span>By {prompt.author.split(':')[0].substring(1)}</span>
                    <span>{prompt.usage_count} uses</span>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 pt-3 border-t border-gray-200">
                    <button
                      onClick={() => handleCopyPrompt(prompt.content)}
                      className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 text-sm bg-blue-50 text-blue-700 rounded hover:bg-blue-100 transition-colors"
                    >
                      <Copy className="w-3 h-3" />
                      Copy
                    </button>
                    {prompt.permissions.can_edit && (
                      <button
                        onClick={() => {
                          setEditingPrompt(prompt);
                          setShowPromptEditor(true);
                        }}
                        className="flex items-center justify-center p-1.5 text-gray-600 hover:bg-gray-100 rounded"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                    )}
                    {prompt.permissions.can_delete && (
                      <button
                        onClick={() => handleDeletePrompt(prompt.id)}
                        className="flex items-center justify-center p-1.5 text-red-600 hover:bg-red-50 rounded"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {filteredPrompts.length === 0 && (
            <div className="text-center py-12 bg-white rounded-lg shadow">
              <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-3" />
              <p className="text-gray-600">No prompts found</p>
              <p className="text-sm text-gray-500">Try adjusting your filters or create a new prompt</p>
            </div>
          )}
        </div>
      )}

      {/* Permissions View */}
      {viewMode === 'permissions' && (
        <div className="space-y-4">
          {/* Header */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Shield className="w-5 h-5 text-blue-600" />
                <div>
                  <h3 className="font-semibold text-gray-900">Role-Based Access Control</h3>
                  <p className="text-sm text-gray-500">{permissions.length} users with permissions</p>
                </div>
              </div>
              <button
                onClick={() => setShowPermissionEditor(true)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Plus className="w-4 h-4" />
                Grant Permission
              </button>
            </div>
          </div>

          {/* Permissions Table */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    User
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Role
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Granted
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Granted By
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {permissions.map((permission) => (
                  <tr key={permission.user_id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <Users className="w-4 h-4 text-gray-400 mr-2" />
                        <span className="text-sm text-gray-900">{permission.user_id}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-medium rounded ${ROLE_COLORS[permission.role]}`}>
                        {permission.role}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {format(new Date(permission.granted_at), 'MMM d, yyyy')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {permission.granted_by}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm">
                      <button
                        onClick={() => handleRevokePermission(permission.user_id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        Revoke
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Analytics View */}
      {viewMode === 'analytics' && usageStats && (
        <div className="space-y-4">
          {/* Overview Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-blue-100 rounded-lg">
                  <BookOpen className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Prompts</p>
                  <p className="text-2xl font-bold text-gray-900">{usageStats.total_prompts}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-green-100 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Usage</p>
                  <p className="text-2xl font-bold text-gray-900">{usageStats.total_usage}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center gap-3">
                <div className="p-3 bg-purple-100 rounded-lg">
                  <Users className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Active Users</p>
                  <p className="text-2xl font-bold text-gray-900">{usageStats.active_users}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Usage by Scope */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Usage by Scope</h3>
            <div className="space-y-3">
              {Object.entries(usageStats.usage_by_scope).map(([scope, count]) => (
                <div key={scope}>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="text-gray-700">{scope}</span>
                    <span className="font-medium text-gray-900">{count}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{
                        width: `${(count / usageStats.total_usage) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Popular Prompts */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Most Popular Prompts</h3>
            <div className="space-y-2">
              {usageStats.popular_prompts.map((prompt, index) => (
                <div key={prompt.prompt_id} className="flex items-center gap-3 p-3 bg-gray-50 rounded">
                  <span className="text-lg font-bold text-gray-400">#{index + 1}</span>
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{prompt.title}</p>
                  </div>
                  <span className="text-sm text-gray-600">{prompt.usage_count} uses</span>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="font-semibold text-gray-900 mb-4">Recent Activity</h3>
            <div className="space-y-3">
              {usageStats.recent_activity.map((activity, index) => (
                <div key={index} className="flex items-start gap-3 pb-3 border-b border-gray-200 last:border-0">
                  <Clock className="w-4 h-4 text-gray-400 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-gray-900">
                      <span className="font-medium">{activity.user}</span> {activity.action}{' '}
                      <span className="font-medium">"{activity.prompt_title}"</span>
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {format(new Date(activity.timestamp), 'MMM d, yyyy HH:mm')}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Prompt Editor Modal */}
      {showPromptEditor && editingPrompt && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">
                {editingPrompt.id ? 'Edit Prompt' : 'Create Prompt'}
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                  <input
                    type="text"
                    value={editingPrompt.title}
                    onChange={(e) => setEditingPrompt({ ...editingPrompt, title: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter prompt title..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Content</label>
                  <textarea
                    value={editingPrompt.content}
                    onChange={(e) => setEditingPrompt({ ...editingPrompt, content: e.target.value })}
                    rows={8}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter prompt content..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Scope</label>
                  <select
                    value={editingPrompt.scope}
                    onChange={(e) =>
                      setEditingPrompt({ ...editingPrompt, scope: e.target.value as 'TEAM' | 'ROOM' | 'USER' })
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="USER">User (Private)</option>
                    <option value="ROOM">Room (Room Members)</option>
                    <option value="TEAM">Team (All Team Members)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tags (comma-separated)</label>
                  <input
                    type="text"
                    value={editingPrompt.tags.join(', ')}
                    onChange={(e) =>
                      setEditingPrompt({
                        ...editingPrompt,
                        tags: e.target.value.split(',').map((t) => t.trim()).filter(Boolean),
                      })
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="e.g., research, analysis, documentation"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowPromptEditor(false);
                    setEditingPrompt(null);
                  }}
                  className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSavePrompt}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Save Prompt
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Permission Editor Modal */}
      {showPermissionEditor && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
            <div className="p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Grant Permission</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">User ID</label>
                  <input
                    type="text"
                    value={selectedUser}
                    onChange={(e) => setSelectedUser(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="@user:matrix.org"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                  <select
                    value={selectedRole}
                    onChange={(e) =>
                      setSelectedRole(e.target.value as 'OWNER' | 'ADMIN' | 'EDITOR' | 'VIEWER')
                    }
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="VIEWER">Viewer (Read-only)</option>
                    <option value="EDITOR">Editor (Edit prompts)</option>
                    <option value="ADMIN">Admin (Manage permissions)</option>
                    <option value="OWNER">Owner (Full control)</option>
                  </select>
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowPermissionEditor(false);
                    setSelectedUser('');
                    setSelectedRole('VIEWER');
                  }}
                  className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleGrantPermission}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Grant Permission
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
