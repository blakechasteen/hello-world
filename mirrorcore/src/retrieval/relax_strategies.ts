export type StrictQuery = { requiredKnots: string[]; requiredThreads: string[]; mustTags?: string[]; };
export function relaxAll(q: StrictQuery): StrictQuery[] {
  const out: StrictQuery[] = [q];
  if (q.requiredKnots.length>0) out.push({ ...q, requiredKnots: q.requiredKnots.slice(0,-1) });
  if (q.requiredThreads.length>0) out.push({ ...q, requiredThreads: q.requiredThreads.slice(0,-1) });
  out.push({ ...q, requiredKnots: [], requiredThreads: [] });
  return out;
}
