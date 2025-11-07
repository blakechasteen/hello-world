-- HoloLoom COS - Event Stream Database Schema
-- Philosophy: Single source of truth, event-sourced architecture
-- Everything is an event: tasks, sales, purchases, notes, plans

CREATE TABLE IF NOT EXISTS events (
    -- Core identity
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Event classification
    type TEXT NOT NULL CHECK(type IN (
        'task',        -- Work performed (time tracking)
        'sale',        -- Revenue transaction
        'purchase',    -- Expense/purchase
        'inventory',   -- Inventory adjustment
        'note',        -- Reflection, observation, insight
        'plan',        -- Planning (daily/weekly goals)
        'goal',        -- Long-term objective
        'review'       -- Daily/weekly review
    )),

    -- Raw input (complete audit trail)
    raw_input TEXT NOT NULL,           -- Original voice transcript or text
    source TEXT DEFAULT 'manual',      -- 'voice', 'manual', 'import', 'auto'

    -- Structured data (JSON for flexibility)
    parsed_data JSON,                  -- Full structured extraction

    -- Common fields (not all used by every event type)
    amount DECIMAL(10,2),              -- Money (positive = in, negative = out)
    quantity DECIMAL(10,2),            -- Count/volume (positive = in, negative = out)
    unit TEXT,                         -- 'hours', 'loaves', 'lb', 'jars', etc.

    -- Business context
    item TEXT,                         -- Product/task/material name
    category TEXT,                     -- 'labor', 'COGS', 'overhead', 'revenue', etc.
    product_line TEXT,                 -- 'bread', 'honey', 'meal_prep', etc.

    -- Relationships
    related_to INTEGER,                -- Link to other events (FK)
    parent_goal TEXT,                  -- Link to 90-day plan milestone

    -- Metadata
    tags TEXT,                         -- Comma-separated searchable tags
    location TEXT,                     -- Physical location if relevant

    -- Confidence & verification (HITL)
    confidence DECIMAL(3,2),           -- NLP parser confidence (0.0-1.0)
    verified BOOLEAN DEFAULT FALSE,    -- Human verified?
    verification_note TEXT,            -- Human correction/note

    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (related_to) REFERENCES events(id)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_product_line ON events(product_line);
CREATE INDEX IF NOT EXISTS idx_events_item ON events(item);
CREATE INDEX IF NOT EXISTS idx_events_verified ON events(verified);
CREATE INDEX IF NOT EXISTS idx_events_confidence ON events(confidence);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events(type, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_category_timestamp ON events(category, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_product_timestamp ON events(product_line, timestamp);

-- Full-text search on raw input and tags
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    raw_input,
    tags,
    item,
    category,
    content='events',
    content_rowid='id'
);

-- Triggers to keep FTS table in sync
CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
  INSERT INTO events_fts(rowid, raw_input, tags, item, category)
  VALUES (new.id, new.raw_input, new.tags, new.item, new.category);
END;

CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
  DELETE FROM events_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
  UPDATE events_fts SET
    raw_input = new.raw_input,
    tags = new.tags,
    item = new.item,
    category = new.category
  WHERE rowid = new.id;
END;

-- Update timestamp trigger
CREATE TRIGGER IF NOT EXISTS events_update_timestamp
AFTER UPDATE ON events
BEGIN
  UPDATE events SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Views for common queries

-- Daily summary
CREATE VIEW IF NOT EXISTS daily_summary AS
SELECT
    DATE(timestamp) as date,
    type,
    product_line,
    COUNT(*) as event_count,
    SUM(amount) as total_amount,
    SUM(quantity) as total_quantity,
    SUM(CASE WHEN type = 'task' THEN quantity ELSE 0 END) as hours_worked,
    SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) as revenue,
    SUM(CASE WHEN type = 'purchase' AND category LIKE 'COGS%' THEN ABS(amount) ELSE 0 END) as cogs,
    SUM(CASE WHEN type = 'task' THEN ABS(amount) ELSE 0 END) as labor_cost
FROM events
GROUP BY DATE(timestamp), type, product_line;

-- Product line performance
CREATE VIEW IF NOT EXISTS product_performance AS
SELECT
    product_line,
    SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) as revenue,
    SUM(CASE WHEN type = 'purchase' AND category LIKE 'COGS%' THEN ABS(amount) ELSE 0 END) as material_cost,
    SUM(CASE WHEN type = 'task' THEN ABS(amount) ELSE 0 END) as labor_cost,
    SUM(CASE WHEN type = 'task' THEN quantity ELSE 0 END) as hours_worked,
    COUNT(DISTINCT CASE WHEN type = 'sale' THEN DATE(timestamp) END) as days_active
FROM events
WHERE product_line IS NOT NULL
GROUP BY product_line;

-- Inventory current state
CREATE VIEW IF NOT EXISTS inventory_current AS
SELECT
    item,
    unit,
    SUM(quantity) as current_quantity,
    AVG(CASE WHEN type = 'purchase' THEN ABS(amount / NULLIF(quantity, 0)) END) as avg_unit_cost,
    MAX(timestamp) as last_updated
FROM events
WHERE item IS NOT NULL
  AND quantity IS NOT NULL
  AND type IN ('purchase', 'sale', 'inventory', 'task')
GROUP BY item, unit
HAVING SUM(quantity) > 0;

-- Unverified expenditures (HITL queue)
CREATE VIEW IF NOT EXISTS unverified_expenditures AS
SELECT
    id,
    timestamp,
    raw_input,
    amount,
    item,
    category,
    confidence,
    parsed_data
FROM events
WHERE type = 'purchase'
  AND verified = FALSE
  AND (confidence < 0.85 OR confidence IS NULL)
ORDER BY timestamp DESC;

-- Weekly summary for planning
CREATE VIEW IF NOT EXISTS weekly_summary AS
SELECT
    strftime('%Y-W%W', timestamp) as week,
    SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) as revenue,
    SUM(CASE WHEN type = 'purchase' AND category LIKE 'COGS%' THEN ABS(amount) ELSE 0 END) as cogs,
    SUM(CASE WHEN type = 'task' THEN ABS(amount) ELSE 0 END) as labor_cost,
    SUM(CASE WHEN type = 'purchase' AND category = 'overhead' THEN ABS(amount) ELSE 0 END) as overhead,
    SUM(CASE WHEN type = 'task' THEN quantity ELSE 0 END) as hours_worked,
    COUNT(DISTINCT product_line) as active_product_lines
FROM events
GROUP BY strftime('%Y-W%W', timestamp);
