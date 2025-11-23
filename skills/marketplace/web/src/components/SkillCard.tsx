/**
 * Skill Card Component
 * ====================
 *
 * **Created**: November 22, 2025
 * **Purpose**: Display individual skill in the marketplace
 *
 * Shows skill metadata, requirements, metrics, and actions.
 */

import React, { useState } from 'react';
import { Download, Trash2, RefreshCw, Star, ExternalLink } from 'lucide-react';
import { Skill, marketplaceClient } from '../api/client';

interface SkillCardProps {
  skill: Skill;
  installed?: boolean;
  onInstall?: (skillId: string) => void;
  onUpgrade?: (skillId: string) => void;
  onRemove?: (skillId: string) => void;
}

export const SkillCard: React.FC<SkillCardProps> = ({
  skill,
  installed = false,
  onInstall,
  onUpgrade,
  onRemove,
}) => {
  const [loading, setLoading] = useState(false);
  const [showRating, setShowRating] = useState(false);
  const [rating, setRating] = useState(5);

  const handleInstall = async () => {
    setLoading(true);
    try {
      await marketplaceClient.installSkill(skill.skill_id);
      onInstall?.(skill.skill_id);
    } catch (error) {
      console.error('Install failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpgrade = async () => {
    setLoading(true);
    try {
      await marketplaceClient.upgradeSkill(skill.skill_id);
      onUpgrade?.(skill.skill_id);
    } catch (error) {
      console.error('Upgrade failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async () => {
    if (!confirm(`Remove ${skill.name}?`)) return;

    setLoading(true);
    try {
      await marketplaceClient.removeSkill(skill.skill_id);
      onRemove?.(skill.skill_id);
    } catch (error) {
      console.error('Remove failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRate = async () => {
    try {
      await marketplaceClient.rateSkill(skill.skill_id, { rating });
      setShowRating(false);
    } catch (error) {
      console.error('Rating failed:', error);
    }
  };

  return (
    <div className="skill-card">
      <div className="skill-card-header">
        <div className="skill-title">
          <h3>{skill.name}</h3>
          <span className="skill-version">v{skill.version}</span>
        </div>
        <span className={`skill-category category-${skill.category}`}>
          {skill.category}
        </span>
      </div>

      <p className="skill-description">{skill.description}</p>

      {/* Tags */}
      {skill.tags.length > 0 && (
        <div className="skill-tags">
          {skill.tags.map((tag) => (
            <span key={tag} className="skill-tag">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Capabilities */}
      {skill.capabilities.length > 0 && (
        <div className="skill-capabilities">
          <strong>Capabilities:</strong>
          <ul>
            {skill.capabilities.slice(0, 3).map((cap) => (
              <li key={cap}>{cap}</li>
            ))}
            {skill.capabilities.length > 3 && (
              <li>... and {skill.capabilities.length - 3} more</li>
            )}
          </ul>
        </div>
      )}

      {/* Requirements */}
      <div className="skill-requirements">
        <strong>Requires:</strong>
        <div className="requirement-badges">
          {skill.requires_warp_space && <span className="req-badge">WarpSpace</span>}
          {skill.requires_yarn_graph && <span className="req-badge">YarnGraph</span>}
          {skill.requires_event_bus && <span className="req-badge">EventBus</span>}
          {skill.requires_mission_control && <span className="req-badge">MissionControl</span>}
          {skill.requires_tools.length > 0 && (
            <span className="req-badge">{skill.requires_tools.length} tools</span>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="skill-metrics">
        <div className="metric">
          <Star size={16} />
          <span>
            {skill.rating.toFixed(1)} ({skill.rating_count} reviews)
          </span>
        </div>
        <div className="metric">
          <Download size={16} />
          <span>{skill.install_count.toLocaleString()} installs</span>
        </div>
      </div>

      {/* Author */}
      {skill.author && (
        <div className="skill-author">
          <strong>By:</strong> {skill.author}
          {skill.homepage && (
            <a href={skill.homepage} target="_blank" rel="noopener noreferrer">
              <ExternalLink size={14} />
            </a>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="skill-actions">
        {!installed ? (
          <button
            className="btn btn-primary"
            onClick={handleInstall}
            disabled={loading}
          >
            <Download size={16} />
            {loading ? 'Installing...' : 'Install'}
          </button>
        ) : (
          <>
            <button
              className="btn btn-secondary"
              onClick={handleUpgrade}
              disabled={loading}
            >
              <RefreshCw size={16} />
              {loading ? 'Upgrading...' : 'Upgrade'}
            </button>
            <button
              className="btn btn-danger"
              onClick={handleRemove}
              disabled={loading}
            >
              <Trash2 size={16} />
              Remove
            </button>
          </>
        )}
        <button
          className="btn btn-outline"
          onClick={() => setShowRating(!showRating)}
        >
          <Star size={16} />
          Rate
        </button>
      </div>

      {/* Rating Form */}
      {showRating && (
        <div className="rating-form">
          <label>
            Rating:
            <select value={rating} onChange={(e) => setRating(Number(e.target.value))}>
              {[1, 2, 3, 4, 5].map((n) => (
                <option key={n} value={n}>
                  {n} star{n !== 1 ? 's' : ''}
                </option>
              ))}
            </select>
          </label>
          <button className="btn btn-sm" onClick={handleRate}>
            Submit
          </button>
          <button className="btn btn-sm btn-outline" onClick={() => setShowRating(false)}>
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};
