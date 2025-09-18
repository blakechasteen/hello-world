/** [1 Linear Algebra] Simple embedding cache + cosine helper for experiments. */
export type EmbeddingProvider = {
  embed(text: string): Promise<number[]>;
};

export class InMemoryEmbeddingProvider implements EmbeddingProvider {
  private store = new Map<string, number[]>();
  constructor(private generator: (text: string) => number[]){ }

  async embed(text: string){
    let vec = this.store.get(text);
    if (!vec){
      vec = this.generator(text);
      this.store.set(text, vec);
    }
    return vec;
  }
}

export function cosineSimilarity(a: number[], b: number[]){
  let dot=0,na=0,nb=0;
  for (let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);
}
