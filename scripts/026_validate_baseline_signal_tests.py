from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def latest_file(directory: Path, pattern: str) -> Path:
    files = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not files:
        raise SystemExit(f"No file found under {directory} matching {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


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
                bad.append(f"{path.relative_to(root)}:{i}:{line[:140]}")
    return bad


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    args = p.parse_args()
    root = Path(args.project_root).resolve()
    manifest_path = latest_file(root / "artifacts" / "logs", "006_baseline_signal_manifest_*.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = manifest.get("metrics", {})
    problems: list[object] = []
    if manifest.get("status") != "ok":
        problems.append(f"manifest status is {manifest.get('status')}")
    if int(metrics.get("panel_rows", 0) or 0) < 100_000:
        problems.append("panel_rows below 100000")
    if int(metrics.get("panel_months", 0) or 0) < 120:
        problems.append("panel_months below 120")
    if int(metrics.get("n_features_tested", 0) or 0) < 5:
        problems.append("fewer than 5 features tested")
    if int(metrics.get("spread_summary_rows", 0) or 0) < 10:
        problems.append("spread_summary_rows below 10")
    if int(metrics.get("rank_ic_summary_rows", 0) or 0) < 10:
        problems.append("rank_ic_summary_rows below 10")
    for label, path_text in manifest.get("tables", {}).items():
        path = Path(path_text)
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"missing or empty table: {label} -> {path}")
    markdown = Path(manifest.get("markdown_report", ""))
    if not markdown.exists():
        problems.append(f"missing markdown report: {markdown}")
    bad_prefixes = scan_no_bad_prefixes(root)
    if bad_prefixes:
        problems.append({"bad_blockquote_prefixes": bad_prefixes[:100]})
    report = {
        "validated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path),
        "metrics": metrics,
        "problems": problems,
        "status": "ok" if not problems else "failed",
    }
    out = root / "artifacts" / "logs" / f"026_baseline_signal_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[026_validate] manifest_path={manifest_path}")
    print(f"[026_validate] report_path={out}")
    print(f"[026_validate] panel_rows={metrics.get('panel_rows')} panel_months={metrics.get('panel_months')} n_features={metrics.get('n_features_tested')}")
    if problems:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        raise SystemExit("Step 006 validation failed")
    print("[026_validate] passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
