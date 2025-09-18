import { listDatabasePages } from '../adapters/notion.js';
import { sqlite } from '../adapters/sqlite.js';
import { env } from '../env.js';

function toMinutes(start: string | null, end: string | null){
  if (!start || !end) return 0;
  const s = new Date(start).getTime();
  const e = new Date(end).getTime();
  return Math.max(0, Math.round((e - s)/60000));
}

export async function syncTimeEntries(){
  await sqlite.connect();
  const pages = await listDatabasePages(env.notion.timeDatabase);
  sqlite.run(`DELETE FROM labor`);
  const insert = `INSERT INTO labor (task_id, minutes, worker, date) VALUES (?, ?, ?, ?)`;
  for (const page of pages){
    const props = page.properties ?? {};
    const taskId = props.Task?.relation?.[0]?.id;
    if (!taskId) continue;
    const minutes = toMinutes(props.Start?.date?.start ?? null, props.End?.date?.start ?? null) || (props.Duration?.number ?? 0);
    const worker = props.Tags?.multi_select?.[0]?.name ?? null;
    const date = props.Start?.date?.start ?? null;
    sqlite.run(insert, [taskId, minutes, worker, date]);
  }
}
