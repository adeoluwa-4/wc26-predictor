select
    match_key,
    cast(strftime(match_date, '%Y%m%d') as integer) as date_key,
    md5(lower(home_team)) as home_team_key,
    md5(lower(away_team)) as away_team_key,
    md5(lower(tournament)) as tournament_key,
    home_score,
    away_score,
    home_score + away_score as total_goals,
    home_score - away_score as home_goal_difference,
    outcome,
    neutral,
    city,
    country,
    source_name,
    snapshot_checksum,
    ingested_at,
    updated_at
from {{ ref('stg_matches') }}
