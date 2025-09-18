import { driver } from './adapters/neo4j.js';
import { notionClient } from './adapters/notion.js';
import { sqlite } from './adapters/sqlite.js';

/** [9 Category] Entry point wiring the adapters as functors between worlds. */
export async function bootstrap(){
  await sqlite.connect();
  await driver.verifyConnectivity();
  console.log('MirrorCore Loom adapters ready.');
  console.log(`Notion client ready: ${Boolean(notionClient)}`);
}

if (import.meta.url === `file://${process.argv[1]}`){
  bootstrap().catch(err => {
    console.error(err);
    process.exitCode = 1;
  }).finally(async ()=>{
    await driver.close();
    await sqlite.close();
  });
}
