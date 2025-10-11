// engine/packing/pack_builder.ts
// Elastic 128k viewport compiler (skeleton).

export type AnchorKind = 'time' | 'place' | 'quote' | 'shard';

export interface Seeds {
  beats?: string[];
  sensory?: string[];
  quotes?: string[];
  keys?: Record<string, string>;
}

export interface Anchor {
  kind: AnchorKind;
  value: string;   // e.g., "2025-09-11T18:03:22Z", "Mailbox", "“first breath after the storm”"
  refId?: string;  // shard id, etc.
}

export interface KnotDetails {
  id: string;
  s1?: string;
  s2?: string;
  s3?: string;
  seeds?: Seeds;
  anchors?: Anchor[];
  tokenEstimate?: number; // rough token count (if you pre-compute)
}

export interface KnotLite {
  id: string;
  t?: number;
  salience?: number;
  evidence?: number;
}

export interface PackBudget {
  total: number;  // e.g., 128000
  content: number; // e.g., 110000
  instr: number;  // e.g., 6000
  query: number;  // e.g., 6000
  slack: number;  // e.g., 6000
}

export interface PackRequest {
  core: KnotLite[];                        // strict intersection picks
  neighbors: Record<string, KnotLite[]>;   // relax buckets: "drop:actor" => [...]
  budget: PackBudget;
  // optional targeting
  targetAnchorRatio?: number; // e.g., 0.15
  maxCore?: number;           // clamp
  maxNeighbors?: number;      // total neighbor S1s
}

export interface PackArtifact {
  spineKnots: KnotDetails[];               // S2 (+ seeds/anchors)
  neighbors: KnotDetails[];                // S1 only (breadth)
  evidence: Anchor[];                      // pooled highlights
  receipts: any;                           // minimal provenance/meta
  tokenEstimate: number;
}

export interface ContentStore {
  // Fetch S1/S2/S3 + seeds/anchors for ids (batched).
  fetchKnotDetails(ids: string[], opts?: { level: 'S1' | 'S2' | 'S3' | 'all' }): Promise<Record<string, KnotDetails>>;
  // Optionally, fetch short shard/quote snippets to raise anchor ratio.
  fetchEvidence(ids: string[], maxPerKnot: number): Promise<Record<string, Anchor[]>>;
}

function estTokens(s: string | undefined): number {
  if (!s) return 0;
  // cheap proxy: ~4 chars/token
  return Math.ceil(s.length / 4);
}

export async function buildPack(store: ContentStore, req: PackRequest): Promise<PackArtifact> {
  const B = req.budget.content;
  const targetAnchor = req.targetAnchorRatio ?? 0.15;
  const maxCore = req.maxCore ?? Math.min(12, req.core.length);
  const maxNeighbors = req.maxNeighbors ?? 30;

  // 1) Choose spine core (already MMR’d upstream).
  const spineIds = req.core.slice(0, maxCore).map(k => k.id);

  // 2) Choose neighbor ids (round-robin over relax buckets for balance).
  const neighborIds: string[] = [];
  const buckets = Object.values(req.neighbors);
  if (buckets.length) {
    let i = 0;
    while (neighborIds.length < maxNeighbors) {
      const b = buckets[i % buckets.length];
      const pick = b[(neighborIds.length / buckets.length) | 0];
      if (!pick) break;
      neighborIds.push(pick.id);
      i++;
    }
  }

  // 3) Fetch content
  const [spineMap, neighMap] = await Promise.all([
    store.fetchKnotDetails(spineIds, { level: 'S2' }),
    store.fetchKnotDetails(neighborIds, { level: 'S1' })
  ]);

  // 4) Evidence pool for core (quotes/shards) to hit anchor targets
  const evidenceMap = await store.fetchEvidence(spineIds, 2);

  // 5) Assemble with budgeting: ~60% core S2, ~25% neighbor S1, ~10% evidence, ~5% receipts
  const spine: KnotDetails[] = [];
  let tokensUsed = 0;
  const targetCore = Math.floor(B * 0.60);
  for (const id of spineIds) {
    const kd = spineMap[id];
    if (!kd) continue;
    const s2t = estTokens(kd.s2);
    if (tokensUsed + s2t > targetCore) break;
    // attach seeds and anchors
    const anchors = (kd.anchors || []).concat(evidenceMap[id] || []);
    spine.push({ ...kd, anchors });
    tokensUsed += s2t + estTokens(JSON.stringify(kd.seeds || {})) + anchors.length * 12;
  }

  const neighbor: KnotDetails[] = [];
  const targetNeighbor = Math.floor(B * 0.25);
  for (const id of neighborIds) {
    const kd = neighMap[id];
    if (!kd) continue;
    const s1t = estTokens(kd.s1);
    if (tokensUsed + s1t > targetCore + targetNeighbor) break;
    neighbor.push({ id: kd.id, s1: kd.s1 });
    tokensUsed += s1t;
  }

  // Evidence bundle (~10%)
  const evidence: Anchor[] = [];
  const targetEvidence = Math.floor(B * 0.10);
  for (const id of spineIds) {
    const evs = evidenceMap[id] || [];
    for (const a of evs) {
      if (tokensUsed + 16 > targetCore + targetNeighbor + targetEvidence) break;
      evidence.push(a);
      tokensUsed += 16;
    }
    if (tokensUsed >= targetCore + targetNeighbor + targetEvidence) break;
  }

  // Receipts/meta (~5%) — minimal provenance (ids, times)
  const receipts = {
    spineIds,
    neighborIds: neighbor.map(n => n.id),
    anchorTarget: targetAnchor
  };
  const receiptsTokens = 400; // small fixed cushion
  tokensUsed += receiptsTokens;

  const artifact: PackArtifact = {
    spineKnots: spine,
    neighbors: neighbor,
    evidence,
    receipts,
    tokenEstimate: tokensUsed
  };

  return artifact;
}