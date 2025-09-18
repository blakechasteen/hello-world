/** [7 Statistics] Basic selectors and smoothing hooks for loom datasets. */
export type NumericSeries = { id: string; value: number }[];

export function zScore(series: NumericSeries){
  if (series.length===0) return new Map<string, number>();
  const mean = series.reduce((acc, s)=>acc+s.value,0)/series.length;
  const variance = series.reduce((acc,s)=>acc+(s.value-mean)**2,0)/series.length;
  const std = Math.sqrt(variance || 1);
  return new Map(series.map(s => [s.id, (s.value-mean)/std]));
}

export function exponentialSmoothing(series: NumericSeries, alpha=0.3){
  let last = series[0]?.value ?? 0;
  const out = new Map<string, number>();
  for (const point of series){
    last = alpha*point.value + (1-alpha)*last;
    out.set(point.id, last);
  }
  return out;
}
