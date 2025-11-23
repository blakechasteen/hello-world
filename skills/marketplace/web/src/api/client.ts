/**
 * Skill Marketplace API Client
 * =============================
 *
 * **Created**: November 22, 2025
 * **Purpose**: TypeScript client for HoloLoom Skill Marketplace API
 *
 * Provides type-safe access to all marketplace endpoints.
 */

import axios, { AxiosInstance } from 'axios';

// ============================================================================
// Types
// ============================================================================

export interface Skill {
  skill_id: string;
  name: string;
  version: string;
  category: string;
  description: string;
  tags: string[];
  capabilities: string[];

  // Requirements
  requires_warp_space: boolean;
  requires_yarn_graph: boolean;
  requires_event_bus: boolean;
  requires_mission_control: boolean;
  requires_tools: string[];

  // Metadata
  author: string;
  homepage: string;
  repository: string;
  license: string;

  // Metrics
  install_count: number;
  usage_count: number;
  rating: number;
  rating_count: number;

  // Timestamps
  created_at: string;
  updated_at: string;
  last_used_at?: string;
}

export interface SearchResult {
  skill: Skill;
  score: number;
  matched_fields: string[];
}

export interface SearchRequest {
  query: string;
  category?: string;
  tags?: string[];
  limit?: number;
}

export interface InstallRequest {
  force?: boolean;
  skip_tests?: boolean;
  skip_dependencies?: boolean;
}

export interface InstallResponse {
  success: boolean;
  skill_id: string;
  message: string;
  installed_path?: string;
  dependencies_installed: string[];
  warnings: string[];
  errors: string[];
}

export interface UpgradeRequest {
  version?: string;
  skip_tests?: boolean;
}

export interface UpgradeResponse {
  success: boolean;
  skill_id: string;
  message: string;
  old_version?: string;
  new_version?: string;
  warnings: string[];
  errors: string[];
}

export interface RemoveRequest {
  force?: boolean;
}

export interface RemoveResponse {
  success: boolean;
  skill_id: string;
  message: string;
  backup_path?: string;
  warnings: string[];
  errors: string[];
}

export interface RatingRequest {
  rating: number;
  comment?: string;
}

export interface Stats {
  total_skills: number;
  by_category: Record<string, number>;
  total_installs: number;
  total_usage: number;
  avg_rating: number;
}

export interface WebSocketMessage {
  type: string;
  skill_id: string;
  data: Record<string, any>;
  timestamp: string;
}

// ============================================================================
// API Client
// ============================================================================

export class MarketplaceClient {
  private client: AxiosInstance;
  private wsConnection: WebSocket | null = null;
  private wsListeners: Map<string, Set<(msg: WebSocketMessage) => void>> = new Map();

  constructor(baseURL: string = 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // ==========================================================================
  // Health Check
  // ==========================================================================

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // ==========================================================================
  // Catalog
  // ==========================================================================

  async listSkills(
    category?: string,
    sortBy: string = 'name',
    limit?: number
  ): Promise<Skill[]> {
    const response = await this.client.get('/api/skills', {
      params: { category, sort_by: sortBy, limit },
    });
    return response.data;
  }

  async getSkill(skillId: string): Promise<Skill> {
    const response = await this.client.get(`/api/skills/${skillId}`);
    return response.data;
  }

  async searchSkills(request: SearchRequest): Promise<SearchResult[]> {
    const response = await this.client.post('/api/skills/search', request);
    return response.data;
  }

  // ==========================================================================
  // Package Management
  // ==========================================================================

  async installSkill(
    skillId: string,
    request: InstallRequest = {}
  ): Promise<InstallResponse> {
    const response = await this.client.post(
      `/api/skills/${skillId}/install`,
      request
    );
    return response.data;
  }

  async upgradeSkill(
    skillId: string,
    request: UpgradeRequest = {}
  ): Promise<UpgradeResponse> {
    const response = await this.client.post(
      `/api/skills/${skillId}/upgrade`,
      request
    );
    return response.data;
  }

  async removeSkill(
    skillId: string,
    request: RemoveRequest = {}
  ): Promise<RemoveResponse> {
    const response = await this.client.delete(`/api/skills/${skillId}`, {
      data: request,
    });
    return response.data;
  }

  async listInstalled(): Promise<Skill[]> {
    const response = await this.client.get('/api/skills/installed');
    return response.data;
  }

  async checkUpdates(): Promise<{ updates_available: number; updates: Record<string, string> }> {
    const response = await this.client.get('/api/skills/updates');
    return response.data;
  }

  // ==========================================================================
  // Analytics
  // ==========================================================================

  async getStats(): Promise<Stats> {
    const response = await this.client.get('/api/stats');
    return response.data;
  }

  async rateSkill(skillId: string, request: RatingRequest): Promise<void> {
    await this.client.post(`/api/skills/${skillId}/rate`, request);
  }

  // ==========================================================================
  // WebSocket
  // ==========================================================================

  connectWebSocket(url: string = 'ws://localhost:8000/ws'): void {
    if (this.wsConnection) {
      return; // Already connected
    }

    this.wsConnection = new WebSocket(url);

    this.wsConnection.onopen = () => {
      console.log('WebSocket connected');
    };

    this.wsConnection.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.notifyListeners(message);
      } catch (e) {
        console.error('Error parsing WebSocket message:', e);
      }
    };

    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.wsConnection.onclose = () => {
      console.log('WebSocket disconnected');
      this.wsConnection = null;
    };
  }

  disconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  onWebSocketMessage(
    eventType: string,
    listener: (msg: WebSocketMessage) => void
  ): () => void {
    if (!this.wsListeners.has(eventType)) {
      this.wsListeners.set(eventType, new Set());
    }

    this.wsListeners.get(eventType)!.add(listener);

    // Return unsubscribe function
    return () => {
      const listeners = this.wsListeners.get(eventType);
      if (listeners) {
        listeners.delete(listener);
      }
    };
  }

  private notifyListeners(message: WebSocketMessage): void {
    const listeners = this.wsListeners.get(message.type);
    if (listeners) {
      listeners.forEach((listener) => listener(message));
    }

    // Also notify wildcard listeners
    const wildcardListeners = this.wsListeners.get('*');
    if (wildcardListeners) {
      wildcardListeners.forEach((listener) => listener(message));
    }
  }
}

// Export singleton instance
export const marketplaceClient = new MarketplaceClient();
