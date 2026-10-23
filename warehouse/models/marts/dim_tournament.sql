select
    md5(lower(tournament)) as tournament_key,
    tournament as tournament_name,
    case
        when lower(tournament) = 'fifa world cup' then 'World Cup'
        when lower(tournament) like '%world cup%qualification%' then 'World Cup qualifier'
        when lower(tournament) like '%friendly%' then 'Friendly'
        when lower(tournament) like '%cup%' or lower(tournament) like '%championship%' then 'Tournament'
        else 'Other'
    end as tournament_type,
    count(*) as match_count,
    min(match_date) as first_match_date,
    max(match_date) as latest_match_date
from {{ ref('stg_matches') }}
group by tournament
