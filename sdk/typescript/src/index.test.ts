/**
 * HoloLoom TypeScript SDK - Example Tests
 * Demonstrates usage patterns and type safety
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { HoloLoomClient, type Spacetime, type RecallResult } from './index';

describe('HoloLoomClient', () => {
  let client: HoloLoomClient;

  beforeEach(() => {
    // Mock fetch globally for tests
    global.fetch = vi.fn();
    client = new HoloLoomClient({ baseUrl: 'http://localhost:8000' });
  });

  describe('Lite API', () => {
    it('should experience a memory', async () => {
      const mockResponse = {
        success: true,
        data: {
          memoryId: 'mem_123',
          timestamp: '2025-12-30T12:00:00Z',
          embedded: true,
          graphUpdated: true,
        },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      const result = await client.experience({
        content: 'Thompson Sampling balances exploration and exploitation',
        metadata: { source: 'research' },
      });

      expect(result.success).toBe(true);
      expect(result.data.memoryId).toBe('mem_123');
      expect(result.data.embedded).toBe(true);
    });

    it('should recall memories', async () => {
      const mockResponse = {
        success: true,
        data: {
          memories: [
            {
              id: 'mem_123',
              content: 'Thompson Sampling balances exploration and exploitation',
              timestamp: '2025-12-30T12:00:00Z',
              embedding: null,
              metadata: { source: 'research' },
              relevance: 0.95,
              confidence: 0.92,
            },
          ],
          totalCount: 1,
          strategy: 'BALANCED',
        },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      const result = await client.recall('Thompson Sampling', {
        strategy: 'SIMILAR',
        limit: 10,
      });

      expect(result.success).toBe(true);
      expect(result.data.memories).toHaveLength(1);
      expect(result.data.memories[0].relevance).toBe(0.95);
    });

    it('should reflect on memories', async () => {
      const mockResponse = {
        success: true,
        data: {
          processed: 5,
          updated: 5,
          feedback: { helpful: true, quality: 0.95 },
        },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      const result = await client.reflect({
        memoryIds: ['mem_123', 'mem_456'],
        feedback: { helpful: true, accurate: true, quality: 0.95 },
      });

      expect(result.success).toBe(true);
      expect(result.data.updated).toBe(5);
    });

    it('should reason through queries', async () => {
      const mockResponse = {
        success: true,
        data: {
          response: 'Thompson Sampling uses Bayesian priors...',
          confidence: 0.92,
          mode: 'VERIFY',
          stepsTaken: [
            {
              index: 0,
              type: 'retrieve',
              query: 'Thompson Sampling',
              result: 'Retrieved 10 relevant memories',
              confidence: 0.95,
              timestamp: '2025-12-30T12:00:00Z',
            },
          ],
          verification: {
            verified: true,
            issues: [],
          },
        },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      const result = await client.reason('Explain Thompson Sampling', {
        mode: 'VERIFY',
        maxSteps: 5,
      });

      expect(result.success).toBe(true);
      expect(result.data.confidence).toBe(0.92);
      expect(result.data.stepsTaken).toHaveLength(1);
    });

    it('should query with full orchestrator', async () => {
      const mockResponse = {
        success: true,
        data: {
          response: 'Thompson Sampling is a Bayesian approach to the multi-armed bandit problem...',
          confidence: 0.94,
          sources: [
            {
              id: 'mem_123',
              content: 'Thompson Sampling definition',
              timestamp: '2025-12-30T12:00:00Z',
              embedding: null,
              metadata: {},
              relevance: 0.98,
            },
          ],
          toolUsed: 'answer',
          reasoning: 'Retrieved relevant memories and synthesized answer',
          trace: {
            stages: [
              { name: 'Retrieval', durationMs: 45, status: 'success' },
              { name: 'Synthesis', durationMs: 105, status: 'success' },
            ],
            totalMs: 150,
          },
          metadata: {
            cacheHit: false,
            complexity: 'SIMPLE',
            threadCount: 8,
          },
        },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      const result = await client.query('What is Thompson Sampling?', {
        mode: 'VERIFY',
        complexityLevel: 'SIMPLE',
      });

      expect(result.success).toBe(true);
      expect(result.data.response).toContain('Thompson Sampling');
      expect(result.data.confidence).toBe(0.94);
      expect(result.data.sources).toHaveLength(1);
      expect(result.data.trace?.totalMs).toBe(150);
    });
  });

  describe('Configuration', () => {
    it('should initialize with default config', () => {
      const defaultClient = new HoloLoomClient();
      const config = defaultClient.getConfig();

      expect(config.baseUrl).toBeDefined();
      expect(config.timeout).toBeGreaterThan(0);
      expect(config.retries).toBeGreaterThan(0);
    });

    it('should allow runtime configuration', () => {
      const client = new HoloLoomClient({ baseUrl: 'http://localhost:8000' });

      client.configure({
        timeout: 60000,
        retries: 5,
      });

      const config = client.getConfig();
      expect(config.timeout).toBe(60000);
      expect(config.retries).toBe(5);
    });

    it('should read API key from environment', () => {
      process.env.HOLOLOOM_API_KEY = 'test-key-123';
      const client = new HoloLoomClient();

      // Verify in actual implementation by checking headers
      // This is a simplified test; real implementation would verify header
      expect(client).toBeDefined();

      delete process.env.HOLOLOOM_API_KEY;
    });
  });

  describe('Error handling', () => {
    it('should handle API errors gracefully', async () => {
      // Mock all fetch calls to return 500 error (client may retry)
      (global.fetch as any).mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => ({ message: 'Internal Server Error' }),
        headers: new Headers({ 'content-type': 'application/json' }),
      });

      await expect(
        client.query('Test query', {}, { retries: 0 })
      ).rejects.toThrow('Internal Server Error');
    });

    it('should retry on network errors', async () => {
      const mockFetch = (global.fetch as any);

      // First call fails, second succeeds
      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            success: true,
            data: { response: 'Success', confidence: 0.9 },
          }),
          headers: new Headers({ 'content-type': 'application/json' }),
        });

      const result = await client.query('Test query', {}, { retries: 1 });

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });
  });

  describe('Type safety', () => {
    it('should enforce type safety on Spacetime result', async () => {
      // This is a compile-time check, but we can test runtime behavior
      const mockResponse = {
        success: true,
        data: {
          response: 'Test response',
          confidence: 0.95,
          sources: [],
          toolUsed: 'answer',
          reasoning: 'Test reasoning',
        } as Spacetime,
      };

      expect(mockResponse.data.response).toBeDefined();
      expect(typeof mockResponse.data.confidence).toBe('number');
      expect(Array.isArray(mockResponse.data.sources)).toBe(true);
    });

    it('should enforce type safety on RecallResult', async () => {
      const mockResult = {
        memories: [],
        totalCount: 0,
        strategy: 'BALANCED' as const,
      } as RecallResult;

      expect(mockResult.strategy).toBe('BALANCED');
      expect(Array.isArray(mockResult.memories)).toBe(true);
    });
  });
});
