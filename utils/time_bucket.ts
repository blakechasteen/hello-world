// utils/timeBucket.ts
export function timeBucket(date: Date | string | number): string {
  const d = new Date(date);
  if (isNaN(d.getTime())) throw new Error(`Invalid date: ${date}`);
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  const hh = d.getUTCHours();
  let part: 'morning'|'afternoon'|'evening'|'night' = 'morning';
  if (hh >= 12 && hh < 17) part = 'afternoon';
  else if (hh >= 17 && hh < 22) part = 'evening';
  else if (hh >= 22 || hh < 5) part = 'night';
  return `${yyyy}-${mm}-${dd}-${part}`;
}