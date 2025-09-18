import { syncTasks } from '../src/notion/sync_tasks.js';
import { syncTimeEntries } from '../src/notion/sync_time_entries.js';

await syncTasks();
await syncTimeEntries();
console.log('Synced Notion tasks and time entries.');
