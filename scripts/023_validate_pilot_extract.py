from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REQUIRED_ROLES = {"13f_holdings", "crsp_monthly_stock", "crsp_daily_stock", "crsp_stock_names"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not candidates:
        raise SystemExit(f"No files found under {directory} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def git_ignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def scan_for_bad_prefixes(root: Path) -> list[str]:
    suffixes = {".md", ".py", ".toml", ".yaml", ".yml", ".gitignore", ".sh"}
    bad: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
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


def check_manifest_has_no_raw_samples(obj: Any, path: str = "manifest") -> list[str]:
    problems: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lower = str(key).lower()
            if any(token in lower for token in ["sample", "preview", "head_records", "records_preview"]):
                problems.append(f"{path}.{key}")
            problems.extend(check_manifest_has_no_raw_samples(value, f"{path}.{key}"))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            problems.extend(check_manifest_has_no_raw_samples(value, f"{path}[{i}]") )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = latest_file(root / "artifacts" / "logs", "003_pilot_extract_manifest_*.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems: list[Any] = []
    if manifest.get("status") != "ok":
        problems.append({"manifest_status": manifest.get("status")})

    by_role = {row.get("role"): row for row in manifest.get("tables", [])}
    for role in REQUIRED_ROLES:
        row = by_role.get(role)
        if not row:
            problems.append(f"missing required pilot role: {role}")
        elif row.get("status") != "pass" or int(row.get("n_rows") or 0) <= 0:
            problems.append({"bad_required_role": role, "row": row})

    ignored_targets = [
        "data/raw/probe.parquet",
        "data/interim/003_pilot/probe.parquet",
        "data/processed/probe.parquet",
        "artifacts/schema/probe.json",
        "artifacts/logs/probe.log",
        "artifacts/figures_static/probe.png",
        "artifacts/figures_interactive/probe.html",
    ]
    not_ignored = [relative for relative in ignored_targets if not git_ignored(root, relative)]
    if not_ignored:
        problems.append({"not_gitignored": not_ignored})

    bad_prefixes = scan_for_bad_prefixes(root)
    if bad_prefixes:
        problems.append({"bad_blockquote_prefixes": bad_prefixes[:50]})

    raw_sample_paths = check_manifest_has_no_raw_samples(manifest)
    if raw_sample_paths:
        problems.append({"manifest_contains_sample_like_keys": raw_sample_paths[:50]})

    report = {
        "validated_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "status": "pass" if not problems else "fail",
        "required_roles": sorted(REQUIRED_ROLES),
        "problems": problems,
    }
    report_path = root / "artifacts" / "logs" / f"023_pilot_extract_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[023_validate_pilot_extract] manifest_path={manifest_path}")
    print(f"[023_validate_pilot_extract] report_path={report_path}")
    if problems:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Step 003 validation failed")
    print("[023_validate_pilot_extract] passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
