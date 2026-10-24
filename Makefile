PYTHON ?= python3

.PHONY: refresh backfill rebuild train pipeline test dbt-test dashboard

refresh:
	$(PYTHON) -m src.platform.cli refresh

backfill:
	@test -n "$(START)" -a -n "$(END)" || (echo "Usage: make backfill START=YYYY-MM-DD END=YYYY-MM-DD" && exit 1)
	$(PYTHON) -m src.platform.cli backfill --start "$(START)" --end "$(END)"

rebuild:
	$(PYTHON) -m src.platform.cli rebuild

train:
	$(PYTHON) -m src.platform.cli train

pipeline:
	$(PYTHON) -m src.platform.cli run-all

test:
	$(PYTHON) -m pytest -q

dbt-test:
	WC26_WAREHOUSE_PATH="$(CURDIR)/data/warehouse/wc26.duckdb" dbt build --project-dir warehouse --profiles-dir warehouse --target dev

dashboard:
	$(PYTHON) -m streamlit run streamlit_app.py
