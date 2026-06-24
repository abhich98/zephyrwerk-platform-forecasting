WITH date_spine AS (
    {{ dbt_utils.date_spine(
        datepart="day",
        start_date="cast('2019-01-01' as date)",
        end_date="cast('2030-12-31' as date)"
    ) }}
)
SELECT
    date_day,
    EXTRACT(YEAR FROM date_day)::int                AS year,
    EXTRACT(QUARTER FROM date_day)::int             AS quarter,
    EXTRACT(MONTH FROM date_day)::int               AS month,
    EXTRACT(WEEK FROM date_day)::int                AS week_of_year,
    EXTRACT(ISODOW FROM date_day)::int - 1          AS day_of_week,
    EXTRACT(ISODOW FROM date_day)::int >= 6         AS is_weekend,
    CASE
        WHEN EXTRACT(MONTH FROM date_day) IN (12, 1, 2) THEN 'winter'
        WHEN EXTRACT(MONTH FROM date_day) IN (3, 4, 5)  THEN 'spring'
        WHEN EXTRACT(MONTH FROM date_day) IN (6, 7, 8)  THEN 'summer'
        ELSE 'autumn'
    END                                             AS season,
    h.date IS NOT NULL                              AS is_german_public_holiday,
    h.holiday_name,
    NOT (EXTRACT(ISODOW FROM date_day)::int >= 6)
        AND h.date IS NULL                          AS is_workday
FROM
    date_spine
    LEFT JOIN {{ ref('german_public_holidays') }} h
        ON date_spine.date_day = h.date
