SELECT
    timestamp :: TIMESTAMP WITH TIME ZONE,
    signal AS signal_name,
    value,
    unit
FROM
    raw.smard_neighbour_prices