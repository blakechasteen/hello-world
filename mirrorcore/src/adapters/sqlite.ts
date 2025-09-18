import Database from 'better-sqlite3';
import { env } from '../env.js';

class SQLiteAdapter {
  private db: Database.Database | null = null;

  async connect(){
    if (!this.db){
      this.db = new Database(env.sqlite.path);
      this.db.pragma('journal_mode = WAL');
    }
  }

  async close(){
    this.db?.close();
    this.db = null;
  }

  run(sql: string, params: any[] = []){
    if (!this.db) throw new Error('Database not connected');
    return this.db.prepare(sql).run(...params);
  }

  all<T = any>(sql: string, params: any[] = []){
    if (!this.db) throw new Error('Database not connected');
    return this.db.prepare(sql).all(...params) as T[];
  }
}

export const sqlite = new SQLiteAdapter();
