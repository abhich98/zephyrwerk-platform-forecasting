SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    signal AS signal_name,
    value,
    unit
FROM
    {{ source('raw', 'smard_prices') }}