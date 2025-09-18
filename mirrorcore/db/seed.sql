-- [0 Bedrock] Base relational schema for tasks, labor, and sales
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT,
  status TEXT,
  total_minutes INTEGER DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS labor (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(id),
  minutes INTEGER NOT NULL,
  worker TEXT,
  date TEXT
);

CREATE TABLE IF NOT EXISTS sales (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  crop TEXT,
  quantity REAL,
  revenue REAL,
  date TEXT
);

CREATE VIEW IF NOT EXISTS task_minutes AS
SELECT t.id AS task_id,
       t.name,
       SUM(l.minutes) AS total_minutes
FROM tasks t
LEFT JOIN labor l ON l.task_id = t.id
GROUP BY t.id, t.name;
