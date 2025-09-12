// engine/retrieval/retrieval_intersect.ts
// Junction-first retrieval: enter via THREADs → intersect → relax

export type ThreadType = 'time' | 'place' | 'actor' | 'theme' | 'glyph';
export interface Thread {
  id: string;
  type: ThreadType;
  key: string;
}

export interface Knot {
  id: string;
  t: number;             // epoch millis
  salience?: number;     // 0..1
  evidence?: number;     // 0..1 (e.g., shard/quote density)
}

export interface QuerySpec {
  // Required or desired threads per type (text will be resolved to keys by your indexer)
  wants: Partial<Record<ThreadType, string[]>>;
  // Optional filters
  timeRange?: { from?: number; to?: number };
  maxCore?: number;      // top-N core knots
  maxNeighbors?: number; // per relax bucket
}

export interface RetrievalResult {
  core: Knot[];                 // strict intersection, top scored
  neighbors: Record<string, Knot[]>; // keyed by relaxedType (e.g., "drop:actor")
  diagnostics?: any;
}

// ----- Inject your indexers/adapters here -----
interface Index {
  resolveThreadsByText(type: ThreadType, texts: string[]): Promise<Thread[]>;
  knotsCrossingThreads(threadIds: string[]): Promise<Knot[]>; // strict intersection
  knotsCrossingAny(threadIds: string[]): Promise<Knot[]>;     // OR (used in relax)
  recentBoost(now: number, t: number): number;                // 0..1
  contentSim(a: Knot, b: Knot): number;                       // for MMR
}

function scoreKnot(k: Knot, coverage: number, now: number): number {
  const sal = k.salience ?? 0.5;
  const ev  = k.evidence ?? 0.0;
  const rec = Math.exp(-Math.max(0, (now - k.t)) / (1000 * 60 * 60 * 24 * 90)); // ~90d half-life
  const w_cov = 0.45, w_sal = 0.25, w_ev = 0.15, w_rec = 0.15;
  return w_cov * coverage + w_sal * sal + w_ev * ev + w_rec * rec;
}

function mmrSelect(items: Knot[], k: number, sim: (a: Knot,b: Knot)=>number, lambda=0.7): Knot[] {
  const selected: Knot[] = [];
  const pool = [...items];
  while (selected.length < k && pool.length) {
    let bestIdx = 0, bestScore = -Infinity;
    for (let i=0;i<pool.length;i++) {
      const cand = pool[i];
      const rel = (cand as any).__score ?? 0;
      const div = selected.length ? Math.max(...selected.map(s => sim(cand, s))) : 0;
      const mmr = lambda * rel - (1 - lambda) * div;
      if (mmr > bestScore) { bestScore = mmr; bestIdx = i; }
    }
    selected.push(pool.splice(bestIdx,1)[0]);
  }
  return selected;
}

export async function retrievalIntersect(index: Index, q: QuerySpec): Promise<RetrievalResult> {
  const now = Date.now();
  const wants = q.wants || {};
  const requiredTypes = Object.entries(wants)
    .filter(([, texts]) => (texts && texts.length))
    .map(([t]) => t as ThreadType);

  // 1) Resolve threads per type
  const resolved: Record<ThreadType, Thread[]> = {} as any;
  for (const t of requiredTypes) {
    resolved[t] = await index.resolveThreadsByText(t, wants[t] as string[]);
  }

  // 2) Strict intersection candidates
  const threadIdsStrict = requiredTypes.flatMap(t => resolved[t].map(r => r.id));
  // If multiple candidates per type, you can generate combinations; for simplicity, OR within each type, AND across types:
  // i.e., intersect({any of time}, {any of place}, ...)
  // Implement in your adapter; here we assume it handles type groups internally.
  const strict = await index.knotsCrossingThreads(threadIdsStrict);

  // 3) Score strict by full coverage (all required types satisfied)
  for (const k of strict) {
    (k as any).__coverage = requiredTypes.length;
    (k as any).__score = scoreKnot(k, requiredTypes.length, now);
  }

  const maxCore = q.maxCore ?? 12;
  const core = mmrSelect(strict.sort((a,b)=>((b as any).__score - (a as any).__score)), maxCore, index.contentSim);

  // 4) Relaxation: drop one type at a time
  const neighbors: Record<string, Knot[]> = {};
  const maxNeighbors = q.maxNeighbors ?? 12;

  for (const dropType of requiredTypes) {
    const keepTypes = requiredTypes.filter(t => t !== dropType);
    if (!keepTypes.length) continue;

    const keepIds = keepTypes.flatMap(t => resolved[t].map(r => r.id));
    const relaxed = await index.knotsCrossingThreads(keepIds); // still AND across remaining types
    // Exclude already in core
    const coreIds = new Set(core.map(k => k.id));
    const uniq = relaxed.filter(k => !coreIds.has(k.id));

    for (const k of uniq) {
      (k as any).__coverage = keepTypes.length;
      (k as any).__score = scoreKnot(k, keepTypes.length, now);
    }
    neighbors[`drop:${dropType}`] = mmrSelect(
      uniq.sort((a,b)=>((b as any).__score - (a as any).__score)),
      maxNeighbors,
      index.contentSim
    );
  }

  return { core, neighbors, diagnostics: { requiredTypes, resolvedCounts: Object.fromEntries(requiredTypes.map(t => [t, resolved[t].length])) } };
}