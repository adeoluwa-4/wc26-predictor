with features as (
    select * from {{ source('warehouse', 'raw_model_features') }}
),
matched as (
    select s.match_key, f.* exclude (match_id)
    from features f
    join {{ ref('stg_matches') }} s
      on cast(f.date as date) = s.match_date
     and f.home_team = s.home_team
     and f.away_team = s.away_team
     and f.tournament = s.tournament
     and coalesce(f.match_occurrence, 0) = s.match_occurrence
),
team_rows as (
    select
        match_key,
        date as feature_date,
        home_team as team_name,
        'home' as side,
        home_points_last_5 as points_last_5,
        home_points_per_match_last_5 as points_per_match_last_5,
        home_goal_diff_last_5 as goal_diff_last_5,
        home_wins_last_5 as wins_last_5,
        home_points_last_10 as points_last_10,
        home_points_per_match_last_10 as points_per_match_last_10,
        home_goal_diff_last_10 as goal_diff_last_10,
        home_wins_last_10 as wins_last_10
    from matched
    union all
    select
        match_key,
        date as feature_date,
        away_team as team_name,
        'away' as side,
        away_points_last_5 as points_last_5,
        away_points_per_match_last_5 as points_per_match_last_5,
        away_goal_diff_last_5 as goal_diff_last_5,
        away_wins_last_5 as wins_last_5,
        away_points_last_10 as points_last_10,
        away_points_per_match_last_10 as points_per_match_last_10,
        away_goal_diff_last_10 as goal_diff_last_10,
        away_wins_last_10 as wins_last_10
    from matched
)
select
    md5(match_key || '|' || side) as team_match_key,
    match_key,
    md5(lower(team_name)) as team_key,
    feature_date,
    side,
    points_last_5,
    points_per_match_last_5,
    goal_diff_last_5,
    wins_last_5,
    points_last_10,
    points_per_match_last_10,
    goal_diff_last_10,
    wins_last_10
from team_rows
