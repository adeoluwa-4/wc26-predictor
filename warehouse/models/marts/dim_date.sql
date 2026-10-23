select distinct
    cast(strftime(match_date, '%Y%m%d') as integer) as date_key,
    match_date as full_date,
    year(match_date) as year,
    quarter(match_date) as quarter,
    month(match_date) as month,
    monthname(match_date) as month_name,
    day(match_date) as day_of_month,
    dayofweek(match_date) as day_of_week,
    weekofyear(match_date) as week_of_year
from {{ ref('stg_matches') }}
