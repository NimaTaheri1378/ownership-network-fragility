# Data access policy

This project is designed for WRDS-backed research, but the public repository must not contain raw WRDS, CRSP, Compustat, Thomson/Refinitiv, SEC bulk extracts under restrictive terms, or other vendor data.

## Credentials

Use local WRDS authentication only, such as `~/.pgpass` or an already configured WRDS environment. Do not commit passwords, API keys, `.pgpass`, or extracted vendor data.

## Public artifacts

Public-safe artifacts include code, documentation, configuration files, aggregate tables, aggregate figures, and sanitized manifests. Local security-level panels remain under ignored `data/` and `artifacts/` folders.

## Rebuild requirements

Users who want to rebuild the full pipeline need their own WRDS entitlements and local credentials.
