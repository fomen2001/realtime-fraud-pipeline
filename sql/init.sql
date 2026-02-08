CREATE TABLE IF NOT EXISTS transactions_enriched (
  transaction_id TEXT PRIMARY KEY,
  customer_id TEXT,
  ts TIMESTAMP,
  amount DOUBLE PRECISION,
  currency TEXT,
  merchant TEXT,
  country TEXT,
  category TEXT,
  tx_count_5min INTEGER,
  amount_zscore DOUBLE PRECISION,
  risk_score DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS fraud_alerts (
  alert_id SERIAL PRIMARY KEY,
  transaction_id TEXT,
  ts TIMESTAMP,
  risk_score DOUBLE PRECISION,
  reason TEXT,
  severity TEXT
);
