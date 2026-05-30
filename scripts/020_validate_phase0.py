from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_latest_schema(root: Path, mode: str) -> Path:
    candidates = sorted(
        (root / "artifacts" / "schema").glob(f"schema_discovery_{mode}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [path for path in candidates if "FAILED" not in path.name]
    if not candidates:
        raise SystemExit(f"No successful schema discovery JSON found for mode={mode}")
    return candidates[0]


def scan_no_bad_prefixes(root: Path) -> list[str]:
    bad: list[str] = []
    suffixes = {".md", ".py", ".toml", ".yaml", ".yml", ".gitignore"}
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
                bad.append(f"{path.relative_to(root)}:{i}:{line[:140]}")
    return bad


def check_data_dirs_clean(root: Path) -> list[str]:
    problems: list[str] = []
    for relative in ["data/raw", "data/interim", "data/processed", "data/external"]:
        directory = root / relative
        for path in directory.rglob("*"):
            if path.is_file() and path.name != ".gitkeep":
                problems.append(str(path.relative_to(root)))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--require-mode", choices=["pilot", "full"], required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    schema_path = find_latest_schema(root, args.require_mode)
    payload = json.loads(schema_path.read_text(encoding="utf-8"))

    checks: dict[str, object] = {
        "validated_utc": utc_stamp(),
        "root": str(root),
        "required_mode": args.require_mode,
        "schema_path": str(schema_path),
        "schema_status": payload.get("status"),
        "metadata_only": payload.get("metadata_only"),
        "candidate_library_count": len(payload.get("candidate_libraries", [])),
        "available_library_count": payload.get("available_library_count", 0),
        "described_table_count": sum(
            len(record.get("described_tables", {}))
            for record in payload.get("libraries", {}).values()
        ),
        "described_column_count": sum(
            int(table_record.get("n_columns", 0))
            for library_record in payload.get("libraries", {}).values()
            for table_record in library_record.get("described_tables", {}).values()
        ),
        "problems": [],
    }

    if payload.get("status") != "ok":
        checks["problems"].append("schema discovery status is not ok")

    if payload.get("metadata_only") is not True:
        checks["problems"].append("schema discovery did not mark metadata_only=true")

    if checks["candidate_library_count"] == 0:
        checks["problems"].append("no candidate libraries discovered")

    if checks["described_table_count"] == 0:
        checks["problems"].append("no tables described")

    readme = (root / "README.md").read_text(encoding="utf-8")
    if "Trading the Production Network" in readme:
        checks["problems"].append("stale project title found in README")
    if "Filing-Date-Clean Ownership Network Fragility" not in readme:
        checks["problems"].append("correct project title missing from README")

    bad_prefixes = scan_no_bad_prefixes(root)
    if bad_prefixes:
        checks["problems"].append({"bad_blockquote_prefixes": bad_prefixes[:100]})

    data_dir_problems = check_data_dirs_clean(root)
    if data_dir_problems:
        checks["problems"].append({"unexpected_data_files": data_dir_problems[:100]})

    out_dir = root / "artifacts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"020_phase0_validation_{args.require_mode}_{utc_stamp()}.json"
    report_path.write_text(json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[020_validate_phase0] schema_path={schema_path}")
    print(f"[020_validate_phase0] report_path={report_path}")
    print(f"[020_validate_phase0] described_table_count={checks['described_table_count']}")
    print(f"[020_validate_phase0] described_column_count={checks['described_column_count']}")

    if checks["problems"]:
        print(json.dumps(checks, indent=2, sort_keys=True))
        raise SystemExit("Phase 0 validation failed")

    print("[020_validate_phase0] passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
