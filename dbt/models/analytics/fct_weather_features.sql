WITH weather_pivot AS(
    SELECT
        timestamp,
        MAX(value) FILTER(WHERE signal_name = 'wind_speed_100m') AS wind_speed_100m,
        MAX(value) FILTER(WHERE signal_name = 'wind_direction_100m') AS wind_direction_100m,
        MAX(value) FILTER(WHERE signal_name = 'shortwave_radiation') AS shortwave_radiation,
        MAX(value) FILTER(WHERE signal_name = 'cloud_cover') AS cloud_cover,
        MAX(value) FILTER(WHERE signal_name = 'temperature_2m') AS temperature_2m

    FROM {{ ref('stg_weather') }}
    GROUP BY timestamp
)

SELECT *
FROM weather_pivot 
ORDER BY timestamp
