/**
 * Skill Detail Component
 * ======================
 *
 * **Created**: December 3, 2025
 * **Purpose**: Detailed view for a single skill
 *
 * Shows full skill information, documentation, dependencies, and actions.
 */

import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  ArrowLeft,
  Download,
  Trash2,
  RefreshCw,
  Star,
  ExternalLink,
  GitBranch,
  Clock,
  User,
  Tag,
  Box,
  AlertTriangle,
  CheckCircle,
  FileText,
  Code,
  Settings,
} from 'lucide-react';
import { marketplaceClient, Skill } from '../api/client';

interface SkillDetailProps {
  skillId: string;
  onBack: () => void;
}

type TabType = 'overview' | 'documentation' | 'changelog' | 'dependencies';

export const SkillDetail: React.FC<SkillDetailProps> = ({ skillId, onBack }) => {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [loading, setLoading] = useState(false);
  const [showRating, setShowRating] = useState(false);
  const [rating, setRating] = useState(5);
  const [comment, setComment] = useState('');

  // Query skill details
  const { data: skill, isLoading, error } = useQuery<Skill>({
    queryKey: ['skill', skillId],
    queryFn: () => marketplaceClient.getSkill(skillId),
  });

  // Install mutation
  const installMutation = useMutation({
    mutationFn: () => marketplaceClient.installSkill(skillId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['skill', skillId] });
      queryClient.invalidateQueries({ queryKey: ['skills'] });
    },
  });

  // Upgrade mutation
  const upgradeMutation = useMutation({
    mutationFn: () => marketplaceClient.upgradeSkill(skillId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['skill', skillId] });
      queryClient.invalidateQueries({ queryKey: ['skills'] });
    },
  });

  // Remove mutation
  const removeMutation = useMutation({
    mutationFn: () => marketplaceClient.removeSkill(skillId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['skill', skillId] });
      queryClient.invalidateQueries({ queryKey: ['skills'] });
      onBack();
    },
  });

  // Rate mutation
  const rateMutation = useMutation({
    mutationFn: () => marketplaceClient.rateSkill(skillId, { rating, comment }),
    onSuccess: () => {
      setShowRating(false);
      setComment('');
      queryClient.invalidateQueries({ queryKey: ['skill', skillId] });
    },
  });

  if (isLoading) {
    return (
      <div className="skill-detail">
        <div className="detail-header">
          <button className="btn btn-outline" onClick={onBack}>
            <ArrowLeft size={16} />
            Back
          </button>
        </div>
        <div className="loading">Loading skill details...</div>
      </div>
    );
  }

  if (error || !skill) {
    return (
      <div className="skill-detail">
        <div className="detail-header">
          <button className="btn btn-outline" onClick={onBack}>
            <ArrowLeft size={16} />
            Back
          </button>
        </div>
        <div className="error-state">
          <AlertTriangle size={48} />
          <p>Failed to load skill details</p>
          <button className="btn btn-primary" onClick={() => queryClient.refetchQueries({ queryKey: ['skill', skillId] })}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  const isInstalled = skill.last_used_at !== undefined;

  return (
    <div className="skill-detail">
      {/* Header */}
      <div className="detail-header">
        <button className="btn btn-outline" onClick={onBack}>
          <ArrowLeft size={16} />
          Back
        </button>

        <div className="detail-actions">
          {!isInstalled ? (
            <button
              className="btn btn-primary"
              onClick={() => installMutation.mutate()}
              disabled={installMutation.isPending}
            >
              <Download size={16} />
              {installMutation.isPending ? 'Installing...' : 'Install'}
            </button>
          ) : (
            <>
              <button
                className="btn btn-secondary"
                onClick={() => upgradeMutation.mutate()}
                disabled={upgradeMutation.isPending}
              >
                <RefreshCw size={16} />
                {upgradeMutation.isPending ? 'Upgrading...' : 'Upgrade'}
              </button>
              <button
                className="btn btn-danger"
                onClick={() => {
                  if (confirm(`Remove ${skill.name}?`)) {
                    removeMutation.mutate();
                  }
                }}
                disabled={removeMutation.isPending}
              >
                <Trash2 size={16} />
                Remove
              </button>
            </>
          )}
          <button className="btn btn-outline" onClick={() => setShowRating(!showRating)}>
            <Star size={16} />
            Rate
          </button>
        </div>
      </div>

      {/* Skill Info */}
      <div className="detail-info">
        <div className="detail-title">
          <h1>{skill.name}</h1>
          <span className="skill-version">v{skill.version}</span>
          <span className={`skill-category category-${skill.category}`}>
            {skill.category}
          </span>
          {isInstalled && (
            <span className="installed-badge">
              <CheckCircle size={16} />
              Installed
            </span>
          )}
        </div>

        <p className="detail-description">{skill.description}</p>

        {/* Meta info */}
        <div className="detail-meta">
          {skill.author && (
            <div className="meta-item">
              <User size={16} />
              <span>{skill.author}</span>
            </div>
          )}
          {skill.repository && (
            <a href={skill.repository} target="_blank" rel="noopener noreferrer" className="meta-item meta-link">
              <GitBranch size={16} />
              <span>Repository</span>
              <ExternalLink size={14} />
            </a>
          )}
          {skill.homepage && (
            <a href={skill.homepage} target="_blank" rel="noopener noreferrer" className="meta-item meta-link">
              <ExternalLink size={16} />
              <span>Homepage</span>
            </a>
          )}
          <div className="meta-item">
            <Clock size={16} />
            <span>Updated {new Date(skill.updated_at).toLocaleDateString()}</span>
          </div>
        </div>

        {/* Metrics */}
        <div className="detail-metrics">
          <div className="metric-box">
            <Star size={24} />
            <div className="metric-value">{skill.rating.toFixed(1)}</div>
            <div className="metric-label">{skill.rating_count} reviews</div>
          </div>
          <div className="metric-box">
            <Download size={24} />
            <div className="metric-value">{skill.install_count.toLocaleString()}</div>
            <div className="metric-label">installs</div>
          </div>
          <div className="metric-box">
            <Box size={24} />
            <div className="metric-value">{skill.usage_count.toLocaleString()}</div>
            <div className="metric-label">usages</div>
          </div>
        </div>

        {/* Tags */}
        {skill.tags.length > 0 && (
          <div className="detail-tags">
            <Tag size={16} />
            {skill.tags.map((tag) => (
              <span key={tag} className="skill-tag">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Rating Form */}
      {showRating && (
        <div className="rating-form-detail">
          <h3>Rate this skill</h3>
          <div className="rating-stars">
            {[1, 2, 3, 4, 5].map((n) => (
              <button
                key={n}
                className={`star-btn ${n <= rating ? 'active' : ''}`}
                onClick={() => setRating(n)}
              >
                <Star size={24} fill={n <= rating ? '#f59e0b' : 'none'} />
              </button>
            ))}
          </div>
          <textarea
            placeholder="Leave a comment (optional)"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={3}
          />
          <div className="rating-actions">
            <button className="btn btn-primary" onClick={() => rateMutation.mutate()} disabled={rateMutation.isPending}>
              Submit Review
            </button>
            <button className="btn btn-outline" onClick={() => setShowRating(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="detail-tabs">
        <button
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <Settings size={16} />
          Overview
        </button>
        <button
          className={`tab ${activeTab === 'documentation' ? 'active' : ''}`}
          onClick={() => setActiveTab('documentation')}
        >
          <FileText size={16} />
          Documentation
        </button>
        <button
          className={`tab ${activeTab === 'changelog' ? 'active' : ''}`}
          onClick={() => setActiveTab('changelog')}
        >
          <Clock size={16} />
          Changelog
        </button>
        <button
          className={`tab ${activeTab === 'dependencies' ? 'active' : ''}`}
          onClick={() => setActiveTab('dependencies')}
        >
          <GitBranch size={16} />
          Dependencies
        </button>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && (
          <div className="overview-content">
            {/* Capabilities */}
            <section className="detail-section">
              <h2>Capabilities</h2>
              {skill.capabilities.length > 0 ? (
                <ul className="capability-list">
                  {skill.capabilities.map((cap) => (
                    <li key={cap}>
                      <CheckCircle size={16} />
                      {cap}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="empty-message">No capabilities defined</p>
              )}
            </section>

            {/* Requirements */}
            <section className="detail-section">
              <h2>Requirements</h2>
              <div className="requirements-grid">
                <div className={`requirement-card ${skill.requires_warp_space ? 'required' : ''}`}>
                  <Code size={20} />
                  <span>WarpSpace</span>
                  {skill.requires_warp_space && <CheckCircle size={14} className="check" />}
                </div>
                <div className={`requirement-card ${skill.requires_yarn_graph ? 'required' : ''}`}>
                  <GitBranch size={20} />
                  <span>YarnGraph</span>
                  {skill.requires_yarn_graph && <CheckCircle size={14} className="check" />}
                </div>
                <div className={`requirement-card ${skill.requires_event_bus ? 'required' : ''}`}>
                  <Box size={20} />
                  <span>EventBus</span>
                  {skill.requires_event_bus && <CheckCircle size={14} className="check" />}
                </div>
                <div className={`requirement-card ${skill.requires_mission_control ? 'required' : ''}`}>
                  <Settings size={20} />
                  <span>MissionControl</span>
                  {skill.requires_mission_control && <CheckCircle size={14} className="check" />}
                </div>
              </div>

              {skill.requires_tools.length > 0 && (
                <div className="tools-required">
                  <h3>Required Tools</h3>
                  <div className="tool-badges">
                    {skill.requires_tools.map((tool) => (
                      <span key={tool} className="tool-badge">
                        {tool}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* License */}
            <section className="detail-section">
              <h2>License</h2>
              <p>{skill.license || 'Not specified'}</p>
            </section>
          </div>
        )}

        {activeTab === 'documentation' && (
          <div className="documentation-content">
            <section className="detail-section">
              <h2>Documentation</h2>
              <div className="doc-placeholder">
                <FileText size={48} />
                <p>Documentation for this skill will be loaded from the skill's README.md file.</p>
                <p className="hint">The skill provides the following capabilities:</p>
                <ul>
                  {skill.capabilities.map((cap) => (
                    <li key={cap}>{cap}</li>
                  ))}
                </ul>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'changelog' && (
          <div className="changelog-content">
            <section className="detail-section">
              <h2>Version History</h2>
              <div className="changelog-entry">
                <div className="version-header">
                  <span className="version-number">v{skill.version}</span>
                  <span className="version-date">{new Date(skill.updated_at).toLocaleDateString()}</span>
                  <span className="version-badge current">Current</span>
                </div>
                <p>Current version installed from marketplace.</p>
              </div>
              <div className="changelog-placeholder">
                <Clock size={32} />
                <p>Full changelog will be available when the skill is fetched from the registry.</p>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'dependencies' && (
          <div className="dependencies-content">
            <section className="detail-section">
              <h2>Dependencies</h2>
              {skill.requires_tools.length > 0 || skill.requires_warp_space || skill.requires_yarn_graph ? (
                <div className="dependency-graph">
                  <div className="dep-node skill-node">
                    <Box size={20} />
                    <span>{skill.name}</span>
                  </div>
                  <div className="dep-connections">
                    {skill.requires_warp_space && (
                      <div className="dep-line">
                        <div className="dep-node system-node">
                          <Code size={16} />
                          <span>WarpSpace</span>
                        </div>
                      </div>
                    )}
                    {skill.requires_yarn_graph && (
                      <div className="dep-line">
                        <div className="dep-node system-node">
                          <GitBranch size={16} />
                          <span>YarnGraph</span>
                        </div>
                      </div>
                    )}
                    {skill.requires_event_bus && (
                      <div className="dep-line">
                        <div className="dep-node system-node">
                          <Box size={16} />
                          <span>EventBus</span>
                        </div>
                      </div>
                    )}
                    {skill.requires_tools.map((tool) => (
                      <div key={tool} className="dep-line">
                        <div className="dep-node tool-node">
                          <Settings size={16} />
                          <span>{tool}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="no-deps">
                  <CheckCircle size={32} />
                  <p>This skill has no external dependencies.</p>
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
};
