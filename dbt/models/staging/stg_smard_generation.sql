SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    signal AS signal_name,
    COALESCE(value, 0) AS value,
    unit
FROM
    raw.smard_generation
WHERE
    signal = 'NUCLEAR' AND timestamp <= '2024-02-04 22:00:00+00' AND value >= 0

UNION ALL

SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    signal AS signal_name,
    value,
    unit
FROM
    raw.smard_generation
WHERE
    signal != 'NUCLEAR' AND value >= 0
