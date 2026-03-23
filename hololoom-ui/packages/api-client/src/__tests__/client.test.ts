import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from './msw-handlers';
import { HoloLoomClient, HoloLoomConnectionError } from '../client';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function createClient() {
  return new HoloLoomClient({
    baseUrl: 'http://localhost:8000',
    enableWebSocket: false,
    timeout: 5000,
  });
}

describe('HoloLoomClient', () => {
  describe('health()', () => {
    it('adapts backend {status: "ok"} to {status: "healthy"}', async () => {
      const client = createClient();
      const health = await client.health();
      expect(health.status).toBe('healthy');
      expect(health.timestamp).toBeTruthy();
      expect(health.components).toEqual([]);
    });
  });

  describe('stats()', () => {
    it('adapts flat backend stats to nested SystemStats', async () => {
      const client = createClient();
      const stats = await client.stats();
      expect(stats.totalQueries).toBe(500);
      expect(stats.avgLatencyMs).toBe(135.5);
      expect(stats.cacheHitRate).toBe(0.78);
      expect(stats.avgConfidence).toBe(0.87);
      expect(stats.memory.totalNodes).toBe(1200); // from memory_shards
      expect(stats.learning.successRate).toBe(0.96); // from success_rate/100
    });
  });

  describe('promptlyChat()', () => {
    it('adapts backend response shape (intent_confidence, model_id, hits[])', async () => {
      const client = createClient();
      const response = await client.promptlyChat({
        text: 'What is Thompson Sampling?',
        room_id: 'test-room',
      });
      expect(response.response).toContain('Echo: What is Thompson Sampling?');
      expect(response.routing?.model).toBe('qwen3.5:9b');
      expect(response.routing?.confidence).toBe(0.85); // adapted from intent_confidence
      expect(response.memory_context?.hits).toBe(3); // array length, not array
      expect(response.memory_context?.latency_ms).toBe(12); // adapted from recall_ms
    });
  });

  describe('getMemoryGraph()', () => {
    it('returns nodes and edges', async () => {
      const client = createClient();
      const graph = await client.getMemoryGraph();
      expect(graph.nodes).toHaveLength(2);
      expect(graph.edges).toHaveLength(1);
      expect(graph.nodes[0].content).toBe('Thompson Sampling');
      expect(graph.stats.coherence).toBe(0.75);
    });
  });

  describe('recall()', () => {
    it('returns matching memories', async () => {
      const client = createClient();
      const result = await client.recall({ query: 'Thompson' });
      expect(result.memories).toHaveLength(1);
      expect(result.memories[0].relevance).toBe(0.92);
    });
  });

  describe('jennyAsk()', () => {
    it('returns generated panels', async () => {
      const client = createClient();
      const result = await client.jennyAsk('Show me confidence');
      expect(result.panels).toHaveLength(1);
      expect(result.panels[0].panel_type).toBe('text');
      expect(result.panels[0].title).toBe('Test Panel');
    });
  });

  describe('retry and error handling', () => {
    it('throws HoloLoomConnectionError on network failure', async () => {
      const client = new HoloLoomClient({
        baseUrl: 'http://localhost:1', // unreachable
        enableWebSocket: false,
        timeout: 500,
        retry: { maxAttempts: 1, baseDelay: 0, maxDelay: 0 },
      });
      await expect(client.health()).rejects.toThrow();
    });
  });
});
