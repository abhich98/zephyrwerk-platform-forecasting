SELECT
    np.timestamp,
    np.signal_name AS neighbour,
    np.value AS neighbour_price_eur_mwh,
    p.value AS de_lu_price_eur_mwh,
    p.value - np.value AS spread_eur_mwh
FROM 
    {{ ref('stg_smard_neighbour_prices') }} AS np
        JOIN
    {{ ref('stg_smard_prices') }} AS p
        ON np.timestamp = p.timestamp
