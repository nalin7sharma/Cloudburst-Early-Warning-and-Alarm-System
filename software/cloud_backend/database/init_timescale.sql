-- TimescaleDB Initialization Script for Cloudburst Early Warning System
-- Creates hypertables and initial schema for time-series data storage

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create database schema
CREATE SCHEMA IF NOT EXISTS cloudburst;

-- Node metadata table
CREATE TABLE cloudburst.nodes (
    node_id VARCHAR(50) PRIMARY KEY,
    location_name VARCHAR(100) NOT NULL,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    altitude DECIMAL(8, 2),
    installation_date TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'active',
    hardware_version VARCHAR(50),
    firmware_version VARCHAR(50),
    battery_level DECIMAL(5, 2) DEFAULT 100.0,
    signal_strength INTEGER DEFAULT 0,
    notes TEXT
);

-- Main sensor data hypertable
CREATE TABLE cloudburst.sensor_data (
    timestamp TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    pressure DECIMAL(7, 2),
    rainfall DECIMAL(6, 2),
    rainfall_hourly DECIMAL(6, 2),
    rainfall_daily DECIMAL(6, 2),
    wind_speed DECIMAL(5, 2),
    wind_direction INTEGER,
    lightning_count INTEGER DEFAULT 0,
    battery_voltage DECIMAL(4, 2),
    solar_voltage DECIMAL(4, 2),
    wind_voltage DECIMAL(4, 2),
    signal_strength INTEGER,
    packet_type VARCHAR(20) DEFAULT 'telemetry'
);

-- Convert to hypertable with 7-day chunks
SELECT create_hypertable(
    'cloudburst.sensor_data', 
    'timestamp',
    chunk_time_interval => INTERVAL '7 days'
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_sensor_data_node_time 
ON cloudburst.sensor_data (node_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp 
ON cloudburst.sensor_data (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_sensor_data_rainfall 
ON cloudburst.sensor_data (rainfall) WHERE rainfall > 0;

-- Risk scores and predictions table
CREATE TABLE cloudburst.risk_predictions (
    timestamp TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    risk_score DECIMAL(3, 2) NOT NULL,
    confidence DECIMAL(3, 2),
    anomaly_type VARCHAR(50),
    severity VARCHAR(20),
    features_used JSONB,
    model_version VARCHAR(50),
    prediction_horizon INTEGER DEFAULT 60, -- minutes
    is_alert_triggered BOOLEAN DEFAULT FALSE
);

-- Convert to hypertable
SELECT create_hypertable(
    'cloudburst.risk_predictions', 
    'timestamp',
    chunk_time_interval => INTERVAL '7 days'
);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_node_risk 
ON cloudburst.risk_predictions (node_id, risk_score DESC, timestamp DESC);

-- Alerts and notifications table
CREATE TABLE cloudburst.alerts (
    alert_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    alert_level VARCHAR(20) NOT NULL,
    risk_score DECIMAL(3, 2),
    message TEXT NOT NULL,
    triggered_by_model VARCHAR(50),
    sent_via_channels JSONB, -- Channels used: SMS, email, push, etc.
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
ON cloudburst.alerts (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_level 
ON cloudburst.alerts (alert_level, timestamp DESC);

-- System events and logs table
CREATE TABLE cloudburst.system_events (
    event_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_source VARCHAR(100),
    severity VARCHAR(20) DEFAULT 'info',
    message TEXT NOT NULL,
    details JSONB,
    node_id VARCHAR(50),
    resolved BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_system_events_timestamp 
ON cloudburst.system_events (timestamp DESC);

-- Gateway communication logs
CREATE TABLE cloudburst.gateway_logs (
    timestamp TIMESTAMPTZ NOT NULL,
    gateway_id VARCHAR(50) NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    packet_type VARCHAR(20),
    packet_size INTEGER,
    signal_strength INTEGER,
    battery_level DECIMAL(5, 2),
    data_quality DECIMAL(3, 2),
    processed_successfully BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

SELECT create_hypertable(
    'cloudburst.gateway_logs', 
    'timestamp',
    chunk_time_interval => INTERVAL '1 day'
);

-- Model training history
CREATE TABLE cloudburst.model_training (
    training_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    training_duration INTERVAL,
    training_samples INTEGER,
    validation_accuracy DECIMAL(5, 4),
    test_accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    hyperparameters JSONB,
    model_path VARCHAR(255),
    is_production BOOLEAN DEFAULT FALSE
);

-- User management for dashboard
CREATE TABLE cloudburst.users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    preferences JSONB
);

-- Alert subscriptions
CREATE TABLE cloudburst.alert_subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES cloudburst.users(user_id),
    node_id VARCHAR(50),
    alert_types JSONB, -- Types of alerts to receive
    channels JSONB, -- Delivery channels: email, sms, push
    min_severity VARCHAR(20) DEFAULT 'medium',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Continuous aggregates for performance
-- Daily rainfall aggregates
CREATE MATERIALIZED VIEW cloudburst.daily_rainfall
WITH (timescaledb.continuous) AS
SELECT 
    node_id,
    time_bucket('1 day', timestamp) AS day,
    SUM(rainfall) as total_rainfall,
    MAX(rainfall) as max_hourly_rainfall,
    AVG(temperature) as avg_temperature,
    AVG(humidity) as avg_humidity,
    COUNT(*) as readings_count
FROM cloudburst.sensor_data
GROUP BY node_id, day
WITH NO DATA;

-- Enable automatic refresh
SELECT add_continuous_aggregate_policy('cloudburst.daily_rainfall',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Hourly risk aggregates
CREATE MATERIALIZED VIEW cloudburst.hourly_risk
WITH (timescaledb.continuous) AS
SELECT 
    node_id,
    time_bucket('1 hour', timestamp) AS hour,
    AVG(risk_score) as avg_risk_score,
    MAX(risk_score) as max_risk_score,
    COUNT(*) as predictions_count,
    SUM(CASE WHEN risk_score > 0.7 THEN 1 ELSE 0 END) as high_risk_count
FROM cloudburst.risk_predictions
GROUP BY node_id, hour
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cloudburst.hourly_risk',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '15 minutes');

-- Node status view
CREATE VIEW cloudburst.node_status AS
SELECT 
    n.node_id,
    n.location_name,
    n.latitude,
    n.longitude,
    n.altitude,
    n.last_seen,
    n.battery_level,
    n.signal_strength,
    sd.temperature as last_temperature,
    sd.humidity as last_humidity,
    sd.rainfall as last_rainfall,
    rp.risk_score as last_risk_score,
    CASE 
        WHEN n.last_seen < NOW() - INTERVAL '1 hour' THEN 'offline'
        WHEN n.battery_level < 20 THEN 'low_battery'
        ELSE 'online'
    END as status,
    (SELECT COUNT(*) FROM cloudburst.alerts a 
     WHERE a.node_id = n.node_id AND a.timestamp > NOW() - INTERVAL '24 hours') as alerts_24h
FROM cloudburst.nodes n
LEFT JOIN cloudburst.sensor_data sd ON (
    n.node_id = sd.node_id 
    AND sd.timestamp = (SELECT MAX(timestamp) FROM cloudburst.sensor_data WHERE node_id = n.node_id)
)
LEFT JOIN cloudburst.risk_predictions rp ON (
    n.node_id = rp.node_id 
    AND rp.timestamp = (SELECT MAX(timestamp) FROM cloudburst.risk_predictions WHERE node_id = n.node_id)
);

-- Retention policies
-- Keep raw sensor data for 1 year
SELECT add_retention_policy('cloudburst.sensor_data', INTERVAL '1 year');

-- Keep risk predictions for 6 months
SELECT add_retention_policy('cloudburst.risk_predictions', INTERVAL '6 months');

-- Keep gateway logs for 3 months
SELECT add_retention_policy('cloudburst.gateway_logs', INTERVAL '3 months');

-- Insert sample node data
INSERT INTO cloudburst.nodes (node_id, location_name, latitude, longitude, altitude, hardware_version) VALUES
('node_001', 'Mountain Peak Station', 28.6139, 77.2090, 1500.0, 'v2.1'),
('node_002', 'Valley Monitoring Post', 28.7041, 77.1025, 800.0, 'v2.1'),
('node_003', 'Ridge Watch Tower', 28.4595, 77.0266, 1200.0, 'v2.0'),
('node_004', 'Forest Outpost', 28.5355, 77.3910, 600.0, 'v2.1');

-- Insert sample user
INSERT INTO cloudburst.users (username, email, password_hash, full_name, role) VALUES
('admin', 'admin@cloudburst.org', 'hashed_password_here', 'System Administrator', 'admin'),
('operator', 'operator@cloudburst.org', 'hashed_password_here', 'System Operator', 'operator');

-- Create functions for common operations

-- Function to get recent sensor data for a node
CREATE OR REPLACE FUNCTION cloudburst.get_recent_sensor_data(
    p_node_id VARCHAR,
    p_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    temperature DECIMAL,
    humidity DECIMAL,
    pressure DECIMAL,
    rainfall DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sd.timestamp,
        sd.temperature,
        sd.humidity,
        sd.pressure,
        sd.rainfall
    FROM cloudburst.sensor_data sd
    WHERE sd.node_id = p_node_id
        AND sd.timestamp >= NOW() - (p_hours || ' hours')::INTERVAL
    ORDER BY sd.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate rainfall statistics
CREATE OR REPLACE FUNCTION cloudburst.get_rainfall_stats(
    p_node_id VARCHAR,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    total_rainfall DECIMAL,
    max_daily_rainfall DECIMAL,
    rainy_days INTEGER,
    current_streak INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH daily_stats AS (
        SELECT 
            DATE(timestamp) as day,
            SUM(rainfall) as daily_rainfall
        FROM cloudburst.sensor_data
        WHERE node_id = p_node_id
            AND timestamp >= NOW() - (p_days || ' days')::INTERVAL
        GROUP BY DATE(timestamp)
    )
    SELECT
        COALESCE(SUM(daily_rainfall), 0) as total_rainfall,
        COALESCE(MAX(daily_rainfall), 0) as max_daily_rainfall,
        COUNT(*) FILTER (WHERE daily_rainfall > 5) as rainy_days,
        (SELECT COUNT(*) 
         FROM (SELECT day, daily_rainfall,
                      day - ROW_NUMBER() OVER (ORDER BY day) * INTERVAL '1 day' as grp
               FROM daily_stats
               WHERE daily_rainfall > 2) t
         WHERE grp = (SELECT MAX(grp) FROM (SELECT day - ROW_NUMBER() OVER (ORDER BY day) * INTERVAL '1 day' as grp
                      FROM daily_stats WHERE daily_rainfall > 2) t2)
        ) as current_streak
    FROM daily_stats;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA cloudburst TO cloudburst_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA cloudburst TO cloudburst_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA cloudburst TO cloudburst_user;

-- Create database user (run this separately as superuser)
-- CREATE USER cloudburst_user WITH PASSWORD 'secure_password';
-- GRANT CONNECT ON DATABASE cloudburst_db TO cloudburst_user;

COMMENT ON SCHEMA cloudburst IS 'Cloudburst Early Warning System Database';
COMMENT ON TABLE cloudburst.sensor_data IS 'Raw sensor data from edge nodes';
COMMENT ON TABLE cloudburst.risk_predictions IS 'ML model risk predictions and scores';
COMMENT ON TABLE cloudburst.alerts IS 'System alerts and notifications';
