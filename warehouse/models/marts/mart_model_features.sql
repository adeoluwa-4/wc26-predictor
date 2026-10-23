select
    *,
    md5(
        coalesce(cast(match_id as varchar), '') || '|' ||
        coalesce(cast(date as varchar), '') || '|' ||
        coalesce(home_team, '') || '|' ||
        coalesce(away_team, '')
    ) as feature_row_checksum
from {{ source('warehouse', 'raw_model_features') }}
where date is not null
  and home_team is not null
  and away_team is not null
  and outcome_label in ('home_win', 'draw', 'away_win')
