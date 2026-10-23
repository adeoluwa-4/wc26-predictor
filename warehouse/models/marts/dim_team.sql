with teams as (
    select home_team as team_name from {{ ref('stg_matches') }}
    union
    select away_team as team_name from {{ ref('stg_matches') }}
)
select
    md5(lower(team_name)) as team_key,
    team_name,
    min(match_date) as first_match_date,
    max(match_date) as latest_match_date
from teams
join {{ ref('int_team_match_history') }} using (team_name)
group by team_name
