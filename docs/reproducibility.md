# Reproducibility

This repository is designed for WRDS-backed research, but it does not redistribute WRDS, CRSP, Compustat, Thomson/Refinitiv, or other vendor data.

## Local setup

```bash
mamba env create -f environment.yml
conda activate ml_core
pip install -e .
make smoke
make test
```

If `ml_core` already exists, activate it and run the editable install and tests.

## Data access

Users must provide their own WRDS/vendor entitlements and local credentials. The code is structured so data access, schema discovery, feature generation, and modeling can be run locally without committing vendor data.

## Public-release policy

The public repository includes code, configuration templates, documentation, tests, aggregate-safe tables, and rendered figures. It intentionally excludes raw vendor data, derived security-level panels, local logs, machine-specific schema contracts, and cluster run wrappers.
