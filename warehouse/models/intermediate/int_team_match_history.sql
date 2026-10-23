with matches as (
    select * from {{ ref('stg_matches') }}
),
home_rows as (
    select
        match_key,
        match_date,
        home_team as team_name,
        away_team as opponent_name,
        true as is_home,
        home_score as goals_for,
        away_score as goals_against
    from matches
),
away_rows as (
    select
        match_key,
        match_date,
        away_team as team_name,
        home_team as opponent_name,
        false as is_home,
        away_score as goals_for,
        home_score as goals_against
    from matches
)
select
    *,
    goals_for - goals_against as goal_difference,
    case when goals_for > goals_against then 3 when goals_for = goals_against then 1 else 0 end as points,
    case when goals_for > goals_against then 1 else 0 end as win,
    case when goals_for = goals_against then 1 else 0 end as draw,
    case when goals_for < goals_against then 1 else 0 end as loss
from (
    select * from home_rows
    union all
    select * from away_rows
)
