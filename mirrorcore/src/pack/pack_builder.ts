import { Candidate, marginalUtilityPerToken, Weights } from '../domain/scoring.js';
import { estimateTokens } from './tokenizers.js';
export function buildPack(queryVec:number[], candidates: Candidate[], tokenBudget:number, W:Weights){
  const pool = candidates.map(c => ({ ...c, tokens: c.tokens ?? estimateTokens(c.content ?? '') }));
  const chosen: Candidate[] = [];
  let remain = tokenBudget;
  while (true) {
    const viable = pool.filter(c => c.tokens<=remain && !chosen.find(d=>d.id===c.id));
    if (!viable.length) break;
    let best: Candidate|null = null; let bestScore = -Infinity;
    for (const c of viable) { const score = marginalUtilityPerToken(c, queryVec, chosen, W); if (score>bestScore){ bestScore=score; best=c; } }
    if (!best || bestScore<=0) break;
    chosen.push(best); remain -= best.tokens;
  }
  return { chosen, used: tokenBudget - remain, remain };
}
