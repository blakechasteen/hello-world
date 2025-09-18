export type Weights = { w_sym:number; w_txt:number; w_g:number; w_rec:number; w_use:number; w_div:number; lambda:number; tau:number; };
export type Candidate = {
  id: string; tokens: number; symScore: number;
  textVec?: number[]; recencyMs?: number; graphDist?: number; usageBoost?: number; content?: string;
};
export function cosine(a:number[], b:number[]){ let dot=0,na=0,nb=0; for (let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; } return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-9); }
export function utilityBase(c: Candidate, qVec:number[], W: Weights){
  const S_txt = (c.textVec && qVec) ? cosine(c.textVec, qVec) : 0;                  // [1 LA]
  const S_g   = (c.graphDist!=null) ? Math.exp(-W.lambda*(c.graphDist)) : 0;       // [2 Calculus]
  const D_rec = (c.recencyMs!=null) ? Math.exp(-(c.recencyMs)/(W.tau)) : 0;        // [2 Calculus]
  const B_use = c.usageBoost ?? 0;                                                 // [7 Stats prior]
  return W.w_sym*c.symScore + W.w_txt*S_txt + W.w_g*S_g + W.w_rec*D_rec + W.w_use*B_use;
}
export function mmrPenalty(c: Candidate, chosen: Candidate[], W: Weights){          // [8 Info]
  if (chosen.length===0 || !c.textVec) return 0;
  const maxSim = Math.max(...chosen.map(d => (d.textVec? cosine(c.textVec!, d.textVec) : 0)));
  return W.w_div*maxSim;
}
export function marginalUtilityPerToken(c: Candidate, qVec:number[], chosen: Candidate[], W: Weights){
  const u = utilityBase(c, qVec, W) - mmrPenalty(c, chosen, W);
  return u / Math.max(c.tokens, 1);
}
