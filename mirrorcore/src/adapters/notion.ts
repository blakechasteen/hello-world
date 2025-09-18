import { Client } from '@notionhq/client';
import { env } from '../env.js';

export const notionClient = new Client({ auth: env.notion.token });

export type NotionPage = Awaited<ReturnType<typeof notionClient.pages.retrieve>>;

export async function listDatabasePages(databaseId: string){
  const pages: any[] = [];
  let cursor: string | undefined;
  do {
    const res = await notionClient.databases.query({ database_id: databaseId, start_cursor: cursor });
    pages.push(...res.results);
    cursor = res.has_more ? res.next_cursor ?? undefined : undefined;
  } while (cursor);
  return pages;
}
