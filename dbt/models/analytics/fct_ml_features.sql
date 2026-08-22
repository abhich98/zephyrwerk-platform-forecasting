SELECT
    g.timestamp,
    g.resolution,
    g.wind_onshore_mw,
    g.wind_offshore_mw,
    g.solar_mw,
    g.biomass_mw,
    g.hydropower_mw,
    g.pumped_storage_mw,
    g.natural_gas_mw,
    g.hard_coal_mw,
    g.brown_coal_mw,
    g.nuclear_mw,
    g.other_conventional_mw,
    g.other_renewable_mw,
    g.total_consumption_mw,
    g.residual_load_mw,
    mp.price_eur_mwh,
    ps.de_lu_price_eur_mwh,
    ps.austria_price_eur_mwh,
    ps.austria_spread_eur_mwh,
    ps.france_price_eur_mwh,
    ps.france_spread_eur_mwh,
    ps.netherlands_price_eur_mwh,
    ps.netherlands_spread_eur_mwh,
    ps.poland_price_eur_mwh,
    ps.poland_spread_eur_mwh,
    ps.switzerland_price_eur_mwh,
    ps.switzerland_spread_eur_mwh,
    ps.czechia_price_eur_mwh,
    ps.czechia_spread_eur_mwh,
    ps.denmark_1_price_eur_mwh,
    ps.denmark_1_spread_eur_mwh,
    ps.denmark_2_price_eur_mwh,
    ps.denmark_2_spread_eur_mwh,
    w.wind_speed_100m_brandenburg,
    w.wind_direction_100m_brandenburg,
    w.shortwave_radiation_brandenburg,
    w.cloud_cover_brandenburg,
    w.temperature_2m_brandenburg,
    w.wind_speed_100m_schleswig,
    w.wind_direction_100m_schleswig,
    w.shortwave_radiation_schleswig,
    w.cloud_cover_schleswig,
    w.temperature_2m_schleswig,
    w.wind_speed_100m_bavaria,
    w.wind_direction_100m_bavaria,
    w.shortwave_radiation_bavaria,
    w.cloud_cover_bavaria,
    w.temperature_2m_bavaria,
    w.wind_speed_100m_bawue,
    w.wind_direction_100m_bawue,
    w.shortwave_radiation_bawue,
    w.cloud_cover_bawue,
    w.temperature_2m_bawue,
    d.date_day,
    d.year,
    d.quarter,
    d.month,
    d.week_of_year,
    d.day_of_week,
    d.is_weekend,
    d.season,
    d.is_german_public_holiday,
    d.holiday_name,
    d.is_workday

FROM 
    {{ ref('fct_energy_generation') }} AS g
        JOIN
    {{ ref('fct_market_prices') }} AS mp
        ON g.timestamp = mp.timestamp
        AND g.resolution = mp.resolution
        JOIN
    {{ ref('fct_price_spreads') }} AS ps
        ON g.timestamp = ps.timestamp
        AND g.resolution = ps.resolution
        JOIN
    {{ ref('fct_weather_features') }} AS w
        ON g.timestamp = w.timestamp
        JOIN
    {{ ref('dim_date') }} AS d
        ON g.timestamp :: date = d.date_day
