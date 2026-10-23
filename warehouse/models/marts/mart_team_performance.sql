select
    md5(lower(team_name)) as team_key,
    team_name,
    count(*) as matches_played,
    sum(win) as wins,
    sum(draw) as draws,
    sum(loss) as losses,
    sum(points) as points,
    sum(goals_for) as goals_for,
    sum(goals_against) as goals_against,
    sum(goal_difference) as goal_difference,
    round(avg(points), 4) as points_per_match,
    round(avg(goals_for), 4) as goals_for_per_match,
    round(avg(goals_against), 4) as goals_against_per_match,
    min(match_date) as first_match_date,
    max(match_date) as latest_match_date
from {{ ref('int_team_match_history') }}
group by team_name
