export function timeBucket(epochMs: number, granularity: 'day'|'week'|'month') {
  const d = new Date(epochMs);
  if (granularity==='day') return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  if (granularity==='week') {
    const day = d.getDay(); const offset = (day+6)%7;
    const monday = new Date(d.getFullYear(), d.getMonth(), d.getDate()-offset);
    return new Date(monday.getFullYear(), monday.getMonth(), monday.getDate()).getTime();
  }
  return new Date(d.getFullYear(), d.getMonth(), 1).getTime();
}
