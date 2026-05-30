from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit(f"No files found under {directory} matching {pattern}")
    return candidates[0]


def scan_no_bad_prefixes(root: Path) -> list[str]:
    bad: list[str] = []
    suffixes = {".md", ".py", ".toml", ".yaml", ".yml", ".gitignore", ".sh"}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.name != ".gitignore" and path.suffix not in suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(lines, start=1):
            if line.startswith("> "):
                bad.append(f"{path.relative_to(root)}:{i}:{line[:120]}")
    return bad


def check_data_dirs_clean(root: Path) -> list[str]:
    problems: list[str] = []
    for relative in ["data/raw", "data/interim", "data/processed", "data/external"]:
        directory = root / relative
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                problems.append(str(path.relative_to(root)))
    return problems


def role_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {role["role"]: role for role in contract.get("roles", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser()
    contract_path = latest_file(root / "artifacts" / "schema", "002_schema_contract_*.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    roles = role_map(contract)

    required_roles = [
        "13f_holdings",
        "13f_manager_names",
        "crsp_monthly_stock",
        "crsp_daily_stock",
        "crsp_stock_names",
        "ccm_link_history",
        "ff_monthly_factors",
    ]

    problems: list[Any] = []
    warnings: list[Any] = []

    if contract.get("metadata_only") is not True:
        problems.append("contract metadata_only flag is not true")

    if contract.get("policy", {}).get("raw_vendor_rows_extracted") is not False:
        problems.append("contract policy does not explicitly forbid raw vendor row extraction at this phase")

    for role_name in required_roles:
        role = roles.get(role_name)
        if not role:
            problems.append(f"missing required role: {role_name}")
            continue
        if not role.get("selected"):
            problems.append(f"required role has no selected table: {role_name}")
            continue
        if role.get("status") not in {"pass", "warn"}:
            problems.append(f"required role did not pass/warn: {role_name} status={role.get('status')}")
        if role.get("status") == "warn":
            warnings.append(f"required role selected with warnings: {role_name}")

    bad_prefixes = scan_no_bad_prefixes(root)
    if bad_prefixes:
        problems.append({"bad_blockquote_prefixes": bad_prefixes[:100]})

    data_dir_problems = check_data_dirs_clean(root)
    if data_dir_problems:
        problems.append({"unexpected_data_files": data_dir_problems[:100]})

    readme_path = root / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        if "Trading the Production Network" in readme:
            problems.append("stale project title found in README")
        if "Filing-Date-Clean Ownership Network Fragility" not in readme:
            problems.append("correct project title missing from README")

    report = {
        "validated_utc": utc_stamp(),
        "contract_path": str(contract_path),
        "metadata_only": contract.get("metadata_only"),
        "status_counts": contract.get("status_counts", {}),
        "required_roles": required_roles,
        "warnings": warnings,
        "problems": problems,
    }

    out_dir = root / "artifacts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"022_schema_contract_validation_{utc_stamp()}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[022_validate_schema_contract] contract_path={contract_path}")
    print(f"[022_validate_schema_contract] report_path={report_path}")
    print(f"[022_validate_schema_contract] status_counts={report['status_counts']}")
    if warnings:
        print(f"[022_validate_schema_contract] warnings={warnings}")

    if problems:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Step 002 validation failed")

    print("[022_validate_schema_contract] passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
