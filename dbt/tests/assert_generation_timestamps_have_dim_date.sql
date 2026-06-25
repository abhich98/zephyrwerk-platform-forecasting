SELECT 
    fact.timestamp
FROM {{ ref('fct_energy_generation') }} AS fact
LEFT JOIN {{ ref('dim_date') }} AS dim 
    ON fact.timestamp::date = dim.date_day
WHERE dim.date_day IS NULL