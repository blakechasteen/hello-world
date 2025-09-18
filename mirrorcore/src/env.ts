import 'dotenv/config';

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Missing environment variable ${name}`);
  return value;
}

export const env = {
  neo4j: {
    uri: requireEnv('NEO4J_URI'),
    user: requireEnv('NEO4J_USER'),
    password: requireEnv('NEO4J_PASSWORD')
  },
  sqlite: {
    path: requireEnv('SQLITE_PATH')
  },
  notion: {
    token: requireEnv('NOTION_TOKEN'),
    tasksDatabase: requireEnv('NOTION_DATABASE_TASKS'),
    timeDatabase: requireEnv('NOTION_DATABASE_TIME')
  }
};
