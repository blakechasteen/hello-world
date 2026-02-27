-- HoloLoom SaaS Database Schema
-- Version: 1.0.0
-- Date: December 2025
--
-- This schema supports the Phase 1 API-only SaaS offering with usage-based billing.
-- Customers bring their own backend (BYO) - we only track billing and API access.

-- ============================================================================
-- CUSTOMERS TABLE
-- Core customer data linked to Stripe for billing
-- ============================================================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,              -- Format: cust_xxxxxxxxxxxx
    email TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    company TEXT,
    password_hash TEXT NOT NULL,               -- bcrypt hash
    stripe_customer_id TEXT UNIQUE,            -- Stripe customer ID (cus_xxx)
    status TEXT NOT NULL DEFAULT 'active',     -- active, suspended, deleted
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for email lookups (login)
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
-- Index for Stripe customer lookups (webhooks)
CREATE INDEX IF NOT EXISTS idx_customers_stripe ON customers(stripe_customer_id);

-- ============================================================================
-- SUBSCRIPTIONS TABLE
-- Plan and billing state for each customer
-- ============================================================================
CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id TEXT PRIMARY KEY,          -- Format: sub_xxxxxxxxxxxx
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    plan TEXT NOT NULL DEFAULT 'free',         -- free, payg, pro, enterprise
    stripe_subscription_id TEXT UNIQUE,        -- Stripe subscription ID (sub_xxx)
    stripe_price_id TEXT,                      -- Stripe price ID for the plan
    status TEXT NOT NULL DEFAULT 'active',     -- active, past_due, canceled, trialing
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    queries_included INTEGER NOT NULL DEFAULT 0,    -- Monthly included queries (0 for PAYG)
    price_per_query_cents REAL,                     -- Overage rate in cents (0.1 = $0.001)
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for customer subscription lookups
CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id);
-- Index for Stripe subscription lookups (webhooks)
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe ON subscriptions(stripe_subscription_id);

-- ============================================================================
-- API_KEYS TABLE
-- Customer API credentials for authentication
-- ============================================================================
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,                   -- Format: holo_xxxxxxxxxxxx
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL,                    -- SHA256 hash (never store plain key)
    key_prefix TEXT NOT NULL,                  -- First 8 chars for identification
    name TEXT NOT NULL DEFAULT 'Default',      -- User-friendly name
    status TEXT NOT NULL DEFAULT 'active',     -- active, revoked
    rate_limit_qps REAL NOT NULL DEFAULT 10.0, -- Queries per second limit
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,                      -- NULL means no expiration
    last_used TIMESTAMP,
    request_count INTEGER NOT NULL DEFAULT 0   -- Total requests made with this key
);

-- Index for customer API key lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_customer ON api_keys(customer_id);
-- Index for key prefix lookups (fast key validation)
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
-- Index for active keys only
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(status) WHERE status = 'active';

-- ============================================================================
-- USAGE_RECORDS TABLE
-- Daily usage aggregation for billing
-- ============================================================================
CREATE TABLE IF NOT EXISTS usage_records (
    id TEXT PRIMARY KEY,                       -- Format: usage_xxxxxxxxxxxx
    customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    date DATE NOT NULL,
    queries_count INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,    -- For future token-based billing
    reported_to_stripe BOOLEAN NOT NULL DEFAULT FALSE,
    stripe_usage_record_id TEXT,               -- Stripe usage record ID after reporting
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id, date)
);

-- Index for unreported usage (billing job)
CREATE INDEX IF NOT EXISTS idx_usage_unreported ON usage_records(reported_to_stripe)
    WHERE reported_to_stripe = FALSE;
-- Index for customer usage history
CREATE INDEX IF NOT EXISTS idx_usage_customer_date ON usage_records(customer_id, date DESC);

-- ============================================================================
-- AUDIT_LOG TABLE
-- Security and compliance audit trail
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,                       -- Format: audit_xxxxxxxxxxxx
    customer_id TEXT REFERENCES customers(customer_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,                  -- signup, login, key_created, key_revoked, etc.
    event_data JSONB,                          -- Event-specific data
    ip_address TEXT,
    user_agent TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for customer audit history
CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_log(customer_id, created_at DESC);
-- Index for event type queries
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type, created_at DESC);

-- ============================================================================
-- PLAN CONFIGURATION (Reference Data)
-- ============================================================================
-- Note: Plan configuration is typically stored in code or Stripe,
-- but this table can be used for runtime configuration changes.

CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY,                  -- free, payg, pro, enterprise
    name TEXT NOT NULL,
    description TEXT,
    queries_included INTEGER NOT NULL DEFAULT 0,
    price_per_query_cents REAL NOT NULL,
    rate_limit_qps REAL NOT NULL DEFAULT 10.0,
    features JSONB,                            -- Feature flags for the plan
    stripe_price_id TEXT,                      -- Stripe price ID
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert default plans
INSERT INTO plans (plan_id, name, description, queries_included, price_per_query_cents, rate_limit_qps, features)
VALUES
    ('free', 'Free', 'Free tier with 100 queries/day limit', 0, 0.0, 1.0,
     '{"daily_limit": 100, "support": "community"}'),
    ('payg', 'Pay As You Go', 'Usage-based pricing with no commitment', 0, 0.1, 10.0,
     '{"daily_limit": null, "support": "email"}'),
    ('pro', 'Pro', 'Professional tier with included queries', 50000, 0.05, 50.0,
     '{"daily_limit": null, "support": "priority", "analytics": true}'),
    ('enterprise', 'Enterprise', 'Custom enterprise pricing', 0, 0.0, 100.0,
     '{"daily_limit": null, "support": "dedicated", "sla": true, "custom_pricing": true}')
ON CONFLICT (plan_id) DO NOTHING;
