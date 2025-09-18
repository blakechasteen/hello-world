/** Metric over embeddings: swap cosine for e.g., hyperbolic or Mahalanobis. */
export type Metric = 'cosine'|'euclidean'|'mahalanobis';
export function distance(a:number[], b:number[], metric:Metric='cosine', covInv?:number[][]){
  if (metric==='euclidean'){
    let s=0; for (let i=0;i<a.length;i++) s+=(a[i]-b[i])**2; return Math.sqrt(s);
  }
  if (metric==='mahalanobis' && covInv){
    // Very light Mahalanobis (no checks)
    const diff = a.map((x,i)=>x-b[i]);
    // diff^T Σ^{-1} diff
    const t = covInv.map(row => row.reduce((acc,rij,j)=>acc+rij*diff[j],0));
    const q = diff.reduce((acc,di,i)=>acc+di*t[i],0);
    return Math.sqrt(Math.max(q,0));
  }
  // cosine distance = 1 - cosine similarity
  let dot=0,na=0,nb=0; for (let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return 1 - dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);
}
