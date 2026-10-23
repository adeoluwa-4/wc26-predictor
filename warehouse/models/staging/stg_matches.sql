select
    match_key,
    cast(date as date) as match_date,
    trim(home_team) as home_team,
    trim(away_team) as away_team,
    cast(home_score as integer) as home_score,
    cast(away_score as integer) as away_score,
    trim(tournament) as tournament,
    nullif(trim(city), '') as city,
    nullif(trim(country), '') as country,
    cast(neutral as boolean) as neutral,
    case
        when home_score > away_score then 'home_win'
        when home_score < away_score then 'away_win'
        else 'draw'
    end as outcome,
    source_home_team,
    source_away_team,
    source_tournament,
    source_city,
    source_country,
    source_name,
    snapshot_checksum,
    ingested_at,
    updated_at
from {{ source('warehouse', 'raw_matches') }}
