with features as (
    select * from {{ source('warehouse', 'raw_model_features') }}
),
matched as (
    select
        s.match_key,
        f.date as snapshot_date,
        f.home_team,
        f.away_team,
        f.home_elo,
        f.away_elo,
        f.home_fifa_rank,
        f.away_fifa_rank,
        f.home_fifa_points,
        f.away_fifa_points
    from features f
    join {{ ref('stg_matches') }} s
      on cast(f.date as date) = s.match_date
     and f.home_team = s.home_team
     and f.away_team = s.away_team
     and f.tournament = s.tournament
     and coalesce(f.match_occurrence, 0) = s.match_occurrence
),
team_rows as (
    select match_key, snapshot_date, home_team as team_name, 'home' as side,
           home_elo as elo, home_fifa_rank as fifa_rank, home_fifa_points as fifa_points
    from matched
    union all
    select match_key, snapshot_date, away_team as team_name, 'away' as side,
           away_elo as elo, away_fifa_rank as fifa_rank, away_fifa_points as fifa_points
    from matched
)
select
    md5(match_key || '|' || side) as team_match_key,
    match_key,
    md5(lower(team_name)) as team_key,
    snapshot_date,
    side,
    elo,
    fifa_rank,
    fifa_points
from team_rows
