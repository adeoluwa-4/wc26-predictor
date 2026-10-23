select
    run_id as model_version,
    finished_at as published_at,
    source,
    rows_loaded,
    rows_rejected,
    duplicates_removed,
    models_built,
    status
from main.pipeline_runs
where status = 'success'
  and models_built > 0
