SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    signal AS signal_name,
    COALESCE(value, 0) AS value,
    unit
FROM
    {{ source('raw', 'smard_generation') }}
WHERE
    signal = 'NUCLEAR'
    AND timestamp <= '{{ var("nuclear_retirement_date") }}'::TIMESTAMP WITH TIME ZONE
    AND (value >= 0 OR value IS NULL)

UNION ALL

SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    signal AS signal_name,
    value,
    unit
FROM
    {{ source('raw', 'smard_generation') }}
WHERE
    signal != 'NUCLEAR' AND value >= 0
