SELECT 'fct_energy_generation' AS fact_table, timestamp
FROM {{ ref('fct_energy_generation') }}
WHERE timestamp::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_market_prices' AS fact_table, timestamp
FROM {{ ref('fct_market_prices') }}
WHERE timestamp::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_price_spreads' AS fact_table, timestamp
FROM {{ ref('fct_price_spreads') }}
WHERE timestamp::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_weather_features' AS fact_table, timestamp
FROM {{ ref('fct_weather_features') }}
WHERE timestamp::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})

UNION ALL

SELECT 'fct_ml_features' AS fact_table, timestamp
FROM {{ ref('fct_ml_features') }}
WHERE timestamp::date NOT IN (SELECT date_day FROM {{ ref('dim_date') }})