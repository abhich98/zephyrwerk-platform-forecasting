SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    region,
    signal_type AS signal_name,
    value,
    unit
FROM
    {{ source('raw', 'weather') }}
