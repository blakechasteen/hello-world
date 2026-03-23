import { http, HttpResponse } from 'msw';

const BASE = 'http://localhost:8000';

/** Mock backend responses for testing */
export const handlers = [
  // Health — matches real backend shape
  http.get(`${BASE}/health`, () =>
    HttpResponse.json({
      status: 'ok',
      service: 'HoloLoom Agentic API',
      version: '1.0.0',
      timestamp: new Date().toISOString(),
    })
  ),

  // Stats — matches real backend flat shape
  http.get(`${BASE}/stats`, () =>
    HttpResponse.json({
      total_queries: 500,
      successful_queries: 480,
      failed_queries: 20,
      avg_latency_ms: 135.5,
      p95_latency_ms: 245.0,
      success_rate: 96.0,
      memory_shards: 1200,
      cache_hit_rate: 0.78,
      avg_confidence: 0.87,
    })
  ),

  // Promptly chat — matches real backend shape
  http.post(`${BASE}/promptly/chat`, async ({ request }) => {
    const body = await request.json() as { text: string; room_id: string };
    return HttpResponse.json({
      response: `Echo: ${body.text}`,
      room_id: body.room_id,
      model: 'qwen3.5:9b',
      routing: { intent: 'general', intent_confidence: 0.85, model_id: 'qwen3.5:9b' },
      memory_context: { hits: ['m1', 'm2', 'm3'], recall_ms: 12, backend: 'inmemory' },
      refinement_info: { triggered: false, complexity_score: 0, max_passes: 4 },
    });
  }),

  // Memory graph — client now uses /api/graph/data
  http.get(`${BASE}/api/graph/data`, () =>
    HttpResponse.json({
      nodes: [
        { id: 'n1', content: 'Thompson Sampling', type: 'concept', state: 'active', activation: 0.9, heat: 0.8, connections: ['n2'], createdAt: '2026-01-01', accessedAt: '2026-03-20' },
        { id: 'n2', content: 'Bayesian', type: 'concept', state: 'active', activation: 0.7, heat: 0.5, connections: ['n1'], createdAt: '2026-01-01', accessedAt: '2026-03-20' },
      ],
      edges: [
        { source: 'n1', target: 'n2', type: 'USES', weight: 0.8 },
      ],
      stats: { totalNodes: 2, totalEdges: 1, activeNodes: 2, avgActivation: 0.8, coherence: 0.75 },
    })
  ),

  // Memory recall
  http.post(`${BASE}/memory/recall`, () =>
    HttpResponse.json({
      memories: [
        { id: 'm1', text: 'Thompson Sampling is a Bayesian approach', relevance: 0.92, state: 'active' },
      ],
      latencyMs: 15,
    })
  ),

  // Jenny
  http.post(`${BASE}/jenny/ask`, () =>
    HttpResponse.json({
      panels: [
        {
          spec_id: 'p1',
          panel_type: 'text',
          title: 'Test Panel',
          content: { text: 'Hello from Jenny' },
          lifecycle: 'stable',
          binding_mode: 'static',
          actions: [],
        },
      ],
    })
  ),
];
