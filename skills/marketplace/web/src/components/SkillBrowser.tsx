/**
 * Skill Browser Component
 * =======================
 *
 * **Created**: November 22, 2025
 * **Purpose**: Main skill marketplace browser
 *
 * Displays skills in grid layout with search, filtering, and actions.
 */

import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Package, TrendingUp, Star, Download } from 'lucide-react';
import { marketplaceClient, Skill, Stats, WebSocketMessage } from '../api/client';
import { SkillCard } from './SkillCard';
import { SearchBar } from './SearchBar';

export const SkillBrowser: React.FC = () => {
  const queryClient = useQueryClient();
  const [view, setView] = useState<'all' | 'installed'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [category, setCategory] = useState<string | undefined>();
  const [sortBy, setSortBy] = useState('name');

  // Query skills
  const { data: skills, isLoading } = useQuery({
    queryKey: ['skills', view, category, sortBy],
    queryFn: async () => {
      if (view === 'installed') {
        return await marketplaceClient.listInstalled();
      } else {
        return await marketplaceClient.listSkills(category, sortBy);
      }
    },
  });

  // Query stats
  const { data: stats } = useQuery<Stats>({
    queryKey: ['stats'],
    queryFn: () => marketplaceClient.getStats(),
  });

  // Search mutation
  const searchMutation = useMutation({
    mutationFn: async ({
      query,
      category,
    }: {
      query: string;
      category?: string;
    }) => {
      const results = await marketplaceClient.searchSkills({
        query,
        category,
        limit: 50,
      });
      return results.map((r) => r.skill);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(['skills', view, category, sortBy], data);
    },
  });

  // WebSocket connection
  useEffect(() => {
    marketplaceClient.connectWebSocket();

    const unsubscribe = marketplaceClient.onWebSocketMessage(
      '*',
      (message: WebSocketMessage) => {
        console.log('WebSocket message:', message);

        // Invalidate queries on install/upgrade/remove
        if (
          message.type === 'install_complete' ||
          message.type === 'upgrade_complete' ||
          message.type === 'remove_complete'
        ) {
          queryClient.invalidateQueries({ queryKey: ['skills'] });
          queryClient.invalidateQueries({ queryKey: ['stats'] });
        }
      }
    );

    return () => {
      unsubscribe();
      marketplaceClient.disconnectWebSocket();
    };
  }, [queryClient]);

  const handleSearch = (query: string, cat?: string, sort?: string) => {
    if (query) {
      setSearchQuery(query);
      searchMutation.mutate({ query, category: cat });
    } else {
      setSearchQuery('');
      setCategory(cat);
      setSortBy(sort || 'name');
    }
  };

  const installedSkillIds = new Set(
    skills?.filter((s) => view === 'installed').map((s) => s.skill_id) || []
  );

  return (
    <div className="skill-browser">
      {/* Header */}
      <div className="browser-header">
        <h1>
          <Package size={32} />
          HoloLoom Skill Marketplace
        </h1>

        {/* Stats */}
        {stats && (
          <div className="stats-bar">
            <div className="stat">
              <Package size={20} />
              <span>{stats.total_skills} Skills</span>
            </div>
            <div className="stat">
              <Download size={20} />
              <span>{stats.total_installs.toLocaleString()} Installs</span>
            </div>
            <div className="stat">
              <Star size={20} />
              <span>{stats.avg_rating.toFixed(1)} Avg Rating</span>
            </div>
          </div>
        )}
      </div>

      {/* View Tabs */}
      <div className="view-tabs">
        <button
          className={`tab ${view === 'all' ? 'active' : ''}`}
          onClick={() => setView('all')}
        >
          <TrendingUp size={16} />
          All Skills
        </button>
        <button
          className={`tab ${view === 'installed' ? 'active' : ''}`}
          onClick={() => setView('installed')}
        >
          <Package size={16} />
          Installed
        </button>
      </div>

      {/* Search Bar */}
      <SearchBar
        onSearch={handleSearch}
        categories={['meta', 'domain', 'agentic']}
      />

      {/* Skills Grid */}
      <div className="skills-grid">
        {isLoading && <div className="loading">Loading skills...</div>}

        {!isLoading && skills && skills.length === 0 && (
          <div className="empty-state">
            <Package size={64} />
            <p>No skills found</p>
          </div>
        )}

        {!isLoading &&
          skills &&
          skills.map((skill) => (
            <SkillCard
              key={skill.skill_id}
              skill={skill}
              installed={installedSkillIds.has(skill.skill_id)}
              onInstall={() => {
                queryClient.invalidateQueries({ queryKey: ['skills'] });
              }}
              onUpgrade={() => {
                queryClient.invalidateQueries({ queryKey: ['skills'] });
              }}
              onRemove={() => {
                queryClient.invalidateQueries({ queryKey: ['skills'] });
              }}
            />
          ))}
      </div>
    </div>
  );
};
