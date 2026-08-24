-- REDUNDANT: TO BE REMOVED SOON.

-- Historical weather forecasts for ML training (leak-safe).
-- These are the forecasts that were actually available at auction time
-- (12:00 CET on D-1), NOT ERA5 actuals. Used to replace the leaky ERA5 join
-- in fct_ml_features for day-ahead price forecasting.
--
-- issue_timestamp = the UTC timestamp when the forecast was issued.
-- timestamp = the hour the forecast predicts.
-- model = the weather model (icon_seamless for stitched, ecmwf_ifs for single runs).
SELECT
    issue_timestamp :: TIMESTAMP WITH TIME ZONE,
    timestamp :: TIMESTAMP WITH TIME ZONE,
    region,
    signal_type AS signal_name,
    NULLIF(value, 'NaN') AS value,
    unit,
    model,
    fetched_at :: TIMESTAMP WITH TIME ZONE
FROM
    {{ source('raw', 'weather_forecast') }}
WHERE
    value IS NOT NULL
