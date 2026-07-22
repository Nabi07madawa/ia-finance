-- Schema de la base de donnees IA Finance
-- A executer dans le SQL Editor de Supabase

-- Table des prix historiques
CREATE TABLE IF NOT EXISTS prices (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(15, 4),
    high DECIMAL(15, 4),
    low DECIMAL(15, 4),
    close DECIMAL(15, 4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Table du portefeuille
CREATE TABLE IF NOT EXISTS portfolio (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    quantity DECIMAL(15, 6) NOT NULL DEFAULT 0,
    avg_price DECIMAL(15, 4) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker)
);

-- Table des transactions
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    type VARCHAR(4) NOT NULL CHECK (type IN ('BUY', 'SELL')),
    ticker VARCHAR(20) NOT NULL,
    quantity DECIMAL(15, 6) NOT NULL,
    price DECIMAL(15, 4) NOT NULL,
    total DECIMAL(15, 4) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table du cash disponible
CREATE TABLE IF NOT EXISTS cash (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    amount DECIMAL(15, 4) NOT NULL DEFAULT 10000.00,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Inserer le cash initial
INSERT INTO cash (id, amount) VALUES (1, 10000.00)
ON CONFLICT (id) DO NOTHING;

-- Table des alertes
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    condition VARCHAR(20) NOT NULL,
    threshold DECIMAL(15, 4) NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table de l'historique des alertes declenchees
CREATE TABLE IF NOT EXISTS alerts_history (
    id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT REFERENCES alerts(id),
    ticker VARCHAR(20) NOT NULL,
    condition VARCHAR(20) NOT NULL,
    threshold DECIMAL(15, 4) NOT NULL,
    current_value DECIMAL(15, 4) NOT NULL,
    triggered_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table des predictions
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    model VARCHAR(20) NOT NULL,
    predicted_price DECIMAL(15, 4) NOT NULL,
    actual_price DECIMAL(15, 4),
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour accelerer les requetes
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(active) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker, prediction_date DESC);

-- Activer RLS (Row Level Security) - optionnel pour un projet personnel
-- mais bonne pratique
ALTER TABLE prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE cash ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Politique : autoriser tout pour le moment (projet personnel)
CREATE POLICY "Allow all on prices" ON prices FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on portfolio" ON portfolio FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on transactions" ON transactions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on cash" ON cash FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on alerts" ON alerts FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on alerts_history" ON alerts_history FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on predictions" ON predictions FOR ALL USING (true) WITH CHECK (true);
