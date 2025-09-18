import { listDatabasePages } from '../adapters/notion.js';
import { sqlite } from '../adapters/sqlite.js';
import { env } from '../env.js';
import { Task } from '../domain/types.js';

function extractTask(page: any): Task {
  const props = page.properties ?? {};
  const name = props.Name?.title?.[0]?.plain_text ?? 'Untitled';
  const status = props.Status?.status?.name ?? 'Unknown';
  const tags = (props.Tags?.multi_select ?? []).map((t: any)=>t.name);
  const totalMinutes = Math.round((props['Total Hours']?.number ?? 0) * 60);
  return {
    id: page.id,
    name,
    status,
    tags,
    totalMinutes
  };
}

export async function syncTasks(){
  await sqlite.connect();
  const pages = await listDatabasePages(env.notion.tasksDatabase);
  sqlite.run(`DELETE FROM tasks`);
  const insert = `INSERT INTO tasks (id, name, category, status, total_minutes, created_at) VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')))`;
  for (const page of pages){
    const task = extractTask(page);
    sqlite.run(insert, [task.id, task.name, task.tags[0] ?? null, task.status, task.totalMinutes, null]);
  }
}
