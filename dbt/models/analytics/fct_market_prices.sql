SELECT
    timestamp,
    value AS price_eur_mwh,
    AVG(value) OVER (ORDER BY timestamp ROWS BETWEEN 167 PRECEDING AND CURRENT ROW) AS price_7d_rolling_avg,
    LAG(value, 24) OVER (ORDER BY timestamp) AS price_24h_lag,
    LAG(value, 48) OVER (ORDER BY timestamp) AS price_48h_lag,
    LAG(value, 168) OVER (ORDER BY timestamp) AS price_168h_lag

FROM {{ ref('stg_smard_prices')}}
