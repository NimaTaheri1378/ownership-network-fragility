from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not candidates:
        raise SystemExit(f"No files found under {directory} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def scan_bad_prefixes(root: Path) -> list[str]:
    bad, suffixes = [], {".md", ".py", ".toml", ".yaml", ".yml", ".gitignore", ".sh"}
    for p in root.rglob("*"):
        if not p.is_file() or ".git" in p.parts: continue
        if p.name != ".gitignore" and p.suffix not in suffixes: continue
        try: lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError: continue
        for i, line in enumerate(lines, 1):
            if line.startswith("> "): bad.append(f"{p.relative_to(root)}:{i}:{line[:120]}")
    return bad


def git_ignored(root: Path, rel: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return r.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", required=True); args = parser.parse_args()
    root = Path(args.project_root).resolve(); manifest_path = latest_file(root / "artifacts" / "logs", "004_pilot_panel_manifest_*.json")
    m = json.loads(manifest_path.read_text(encoding="utf-8")); metrics = m.get("metrics", {})
    problems: list[Any] = []
    if m.get("status") != "ok": problems.append({"manifest_status": m.get("status"), "manifest_problems": m.get("problems", [])})
    for key, minimum in [("mapped_common_rows", 50), ("panel_rows", 50), ("unique_stocks", 5), ("unique_managers", 5)]:
        if int(metrics.get(key, 0)) < minimum: problems.append(f"{key} below {minimum}")
    for key in ["mapping_funnel_png", "monthly_coverage_png", "network_weighted_degree_png", "interactive_html"]:
        fig = Path(m.get("figures", {}).get(key, ""))
        if not fig.exists(): problems.append(f"missing figure {key}")
    for rel in ["data/processed/004_pilot_panel/probe.parquet", "data/interim/004_crosswalk/probe.parquet", "artifacts/logs/probe.log", "artifacts/figures_static/probe.png", "artifacts/figures_interactive/probe.html"]:
        if not git_ignored(root, rel): problems.append({"not_gitignored": rel})
    bad = scan_bad_prefixes(root)
    if bad: problems.append({"bad_blockquote_prefixes": bad[:100]})
    report = {"validated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "manifest_path": str(manifest_path), "status": "passed" if not problems else "failed", "metrics": metrics, "problems": problems}
    report_path = root / "artifacts" / "logs" / f"024_pilot_panel_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[024_validate] manifest_path={manifest_path}"); print(f"[024_validate] report_path={report_path}"); print(f"[024_validate] panel_rows={metrics.get('panel_rows')} mapped_common_rows={metrics.get('mapped_common_rows')}")
    if problems:
        print(json.dumps(report, indent=2, sort_keys=True, default=str)); raise SystemExit("Step 004 validation failed")
    print("[024_validate] passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
