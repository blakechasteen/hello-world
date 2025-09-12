// engine/regen/regen.ts
// Constrained regeneration: expands PackArtifact into a grounded recall.

import { PackArtifact, KnotDetails, Anchor } from '../packing/pack_builder';

export interface RegenConfig {
  personaTone: 'reverent' | 'calm' | 'clinical' | string;
  allowEvoked: boolean;
  anchorRatioTarget: number; // e.g., 0.15
}

export interface RegenIO {
  system: string;
  input: any;
}

export interface RegenOutput {
  text: string;
  anchorsUsed: Anchor[];
  anchorRatio: number;
  nextKnots: { id: string; title?: string }[];
  flags?: { missingAnchors?: boolean; lowAnchorRatio?: boolean; unmarkedInterpolation?: boolean };
}

const SYSTEM_TEMPLATE = (tone: string, allowEvoked: boolean) => `
You are a grounded memory-expander speaking with a ${tone} tone.
Rules:
1) Every paragraph must include at least one ANCHOR reference (time, place, quote, shard).
2) Use only provided seeds, quotes, evidence. If you interpolate, mark as [evoked]${allowEvoked ? '' : ' (avoid interpolation where possible)'}.
3) Keep 3–6 sentences for core recall. Offer 1–2 adjacent knots to explore.
4) Do not invent proper nouns, times, or locations not present in anchors/seeds.
5) Match affect implied by the seeds (if present) without exaggeration.
`;

export function buildRegenInput(pack: PackArtifact) {
  // Minimal shape for the LLM to consume.
  return {
    spine: pack.spineKnots.map(k => ({
      id: k.id,
      s2: k.s2,
      seeds: k.seeds,
      anchors: k.anchors
    })),
    neighbors: pack.neighbors.map(n => ({ id: n.id, s1: n.s1 })),
    evidence: pack.evidence,
    receipts: pack.receipts
  };
}

// Very naive token estimation by spaces; replace with a tokenizer lib if you need precision.
function tokenCount(s: string): number {
  return Math.ceil((s || '').split(/\s+/).filter(Boolean).length);
}

export async function generateLLM(io: RegenIO): Promise<string> {
  // Plug your LLM here (OpenAI, local, etc.). For now we just throw to remind you.
  // Example (pseudo):
  // return await llm.chat({ system: io.system, user: JSON.stringify(io.input) });
  throw new Error('generateLLM not wired. Connect your LLM here.');
}

export function verifyRecall(text: string, anchors: Anchor[], targetRatio: number) {
  const words = tokenCount(text);
  // count anchor mentions: crude heuristic — presence of time/place tokens or quotes/shard ids
  const anchorHits = anchors.reduce((acc, a) => {
    const v = String(a.value);
    return acc + (text.includes(v) ? 1 : 0);
  }, 0);
  const anchorTokens = Math.max(1, Math.floor(anchorHits * 8)); // ~8 tokens per hit (very rough)
  const anchorRatio = anchorTokens / Math.max(1, words);

  const missingAnchors = anchorHits === 0;
  const lowAnchorRatio = anchorRatio < targetRatio;
  const unmarkedInterpolation = /\b(?:invented|imagined)\b/i.test(text) && !/\[evoked\]/.test(text);

  return { anchorRatio, missingAnchors, lowAnchorRatio, unmarkedInterpolation };
}

export async function regenFromPack(pack: PackArtifact, cfg: RegenConfig): Promise<RegenOutput> {
  const system = SYSTEM_TEMPLATE(cfg.personaTone, cfg.allowEvoked);
  const input = buildRegenInput(pack);

  // Aggregate anchors (unique by value) to check verification later
  const anchorsUnique: Anchor[] = [];
  const seen = new Set<string>();
  for (const k of pack.spineKnots) {
    for (const a of (k.anchors || [])) {
      const key = `${a.kind}:${a.value}`;
      if (!seen.has(key)) {
        seen.add(key);
        anchorsUnique.push(a);
      }
    }
  }
  for (const a of pack.evidence) {
    const key = `${a.kind}:${a.value}`;
    if (!seen.has(key)) {
      seen.add(key);
      anchorsUnique.push(a);
    }
  }

  const io: RegenIO = { system, input };

  // 1) Generate
  const text = await generateLLM(io); // replace with your LLM call

  // 2) Verify
  const { anchorRatio, missingAnchors, lowAnchorRatio, unmarkedInterpolation } =
    verifyRecall(text, anchorsUnique, cfg.anchorRatioTarget);

  // 3) Offer adjacent (take top 2 neighbors if present)
  const nextKnots = (pack.neighbors || []).slice(0, 2).map(n => ({ id: n.id }));

  return {
    text,
    anchorsUsed: anchorsUnique,
    anchorRatio,
    nextKnots,
    flags: {
      missingAnchors,
      lowAnchorRatio,
      unmarkedInterpolation
    }
  };
}