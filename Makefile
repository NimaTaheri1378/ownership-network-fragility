SHELL := /usr/bin/env bash
PROJECT_ROOT := $(CURDIR)
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
PYTHONPATH := src

.PHONY: compile smoke phase0_pilot phase0_validate_pilot phase0_full phase0_validate_full phase0 test snapshot

compile:
	PYTHONPATH=$(PYTHONPATH) python -m compileall -q src scripts tests

smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/001_smoke.py --project-root "$(PROJECT_ROOT)"

phase0_pilot:
	PYTHONPATH=$(PYTHONPATH) python scripts/010_discover_wrds_schema.py \
		--project-root "$(PROJECT_ROOT)" \
		--out-dir "$(PROJECT_ROOT)/artifacts/schema" \
		--mode pilot \
		--run-id "$(RUN_ID)"

phase0_validate_pilot:
	PYTHONPATH=$(PYTHONPATH) python scripts/020_validate_phase0.py \
		--project-root "$(PROJECT_ROOT)" \
		--require-mode pilot

phase0_full:
	PYTHONPATH=$(PYTHONPATH) python scripts/010_discover_wrds_schema.py \
		--project-root "$(PROJECT_ROOT)" \
		--out-dir "$(PROJECT_ROOT)/artifacts/schema" \
		--mode full \
		--run-id "$(RUN_ID)"

phase0_validate_full:
	PYTHONPATH=$(PYTHONPATH) python scripts/020_validate_phase0.py \
		--project-root "$(PROJECT_ROOT)" \
		--require-mode full

phase0: compile smoke phase0_pilot phase0_validate_pilot phase0_full phase0_validate_full test

test:
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s tests -p "test_*.py"

snapshot:
	@echo "Project root: $(PROJECT_ROOT)"
	@echo "Run ID: $(RUN_ID)"
	@find . -maxdepth 3 -type f | sort | sed 's#^./##'

.PHONY: step002_schema step002_validate step002

step002_schema:
	PYTHONPATH=$(PYTHONPATH) python scripts/002_build_schema_contract.py \
		--project-root "$(PROJECT_ROOT)" \
		--run-id "$(RUN_ID)"

step002_validate:
	PYTHONPATH=$(PYTHONPATH) python scripts/022_validate_schema_contract.py \
		--project-root "$(PROJECT_ROOT)"

step002: compile smoke step002_schema step002_validate test

.PHONY: step003
step003: compile smoke
	PYTHONPATH=$(PYTHONPATH) python scripts/003_pilot_extract_and_quality.py 		--project-root "$(PROJECT_ROOT)" 		--run-id "$(RUN_ID)" 		--pilot-start "2019-01-01" 		--pilot-end "2020-12-31" 		--max-rows-13f 75000 		--max-rows-crsp-monthly 60000 		--max-rows-crsp-daily 120000 		--max-rows-reference 50000
	PYTHONPATH=$(PYTHONPATH) python scripts/023_validate_pilot_extract.py 		--project-root "$(PROJECT_ROOT)"
	PYTHONPATH=$(PYTHONPATH) python -m unittest discover -s tests -p "test_*.py"

.PHONY: step004
step004: compile smoke
	PYTHONPATH=src python scripts/004_pilot_panel_and_network.py \
		--project-root "$(PROJECT_ROOT)" \
		--run-id "$(RUN_ID)"
	PYTHONPATH=src python scripts/024_validate_pilot_panel_network.py \
		--project-root "$(PROJECT_ROOT)"
	PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
.PHONY: step005
step005: compile smoke
	PYTHONPATH=src python scripts/005_full_panel_network_scale.py 		--project-root "$(PROJECT_ROOT)" 		--run-id "$(RUN_ID)" 		--start-date "$${ONF_FULL_START_DATE:-2000-01-01}" 		--end-date "$${ONF_FULL_END_DATE:-2025-12-31}" 		--chunk-months "$${ONF_WRDS_CHUNK_MONTHS:-3}" 		--n-jobs "$${ONF_N_JOBS:-32}" 		--network-jobs "$${ONF_NETWORK_JOBS:-8}"
	PYTHONPATH=src python scripts/025_validate_full_panel_network.py 		--project-root "$(PROJECT_ROOT)"
	PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"

.PHONY: step006_run step006_validate step006

step006_run:
	PYTHONPATH=src python scripts/006_baseline_signal_tests.py \
		--project-root "$(PROJECT_ROOT)" \
		--run-id "$(RUN_ID)" \
		--n-jobs "$${ONF_N_JOBS:-32}"

step006_validate:
	PYTHONPATH=src python scripts/026_validate_baseline_signal_tests.py \
		--project-root "$(PROJECT_ROOT)"

step006: compile smoke step006_run step006_validate test

.PHONY: release_audit
release_audit:
	python scripts/011_public_repo_audit.py --project-root .

