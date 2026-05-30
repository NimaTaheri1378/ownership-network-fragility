from __future__ import annotations
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
MIN_COUNTS = {"clean_13f_rows": 100_000, "mapped_common_rows": 25_000, "position_rows": 25_000, "panel_rows": 10_000, "unique_months": 36, "unique_stocks": 500, "unique_managers": 100}

def latest_file(directory: Path, pattern: str) -> Path:
    files = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not files: raise SystemExit(f"No file found under {directory} matching {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)

def git_ignored(root: Path, rel: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", rel], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0

def bad_prefixes(root: Path) -> list[str]:
    out=[]; suffixes={".md", ".py", ".toml", ".yaml", ".yml", ".gitignore", ".sh"}
    for p in root.rglob("*"):
        if not p.is_file() or ".git" in p.parts or (p.name != ".gitignore" and p.suffix not in suffixes): continue
        try: lines=p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError: continue
        for i, line in enumerate(lines, 1):
            if line.startswith("> "): out.append(f"{p.relative_to(root)}:{i}")
    return out

def main() -> int:
    a=argparse.ArgumentParser(); a.add_argument("--project-root", required=True); ns=a.parse_args(); root=Path(ns.project_root).resolve(); manifest_path=latest_file(root/"artifacts"/"logs", "005_full_panel_manifest_*.json"); m=json.loads(manifest_path.read_text(encoding="utf-8")); metrics=m.get("metrics", {}); problems=[]
    if m.get("status") != "ok": problems.append(f"manifest status is {m.get('status')}")
    for k,v in MIN_COUNTS.items():
        if float(metrics.get(k,0) or 0) < v: problems.append(f"{k} below threshold: {metrics.get(k)} < {v}")
    if float(metrics.get("panel_fwd_ret_1m_coverage",0) or 0) < 0.50: problems.append("panel_fwd_ret_1m_coverage below 0.50")
    if bad_prefixes(root): problems.append("copied > prefix contamination found")
    for rel in ["data/interim/005_full_core/probe.parquet", "data/processed/005_full_panel/probe.parquet", "artifacts/logs/probe.log", "artifacts/figures_static/probe.png", "artifacts/figures_interactive/probe.html"]:
        if not git_ignored(root, rel): problems.append(f"not gitignored: {rel}")
    report={"validated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "manifest_path": str(manifest_path), "metrics": metrics, "problems": problems, "status": "ok" if not problems else "failed"}; rp=root/"artifacts"/"logs"/f"025_full_panel_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"; rp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[025_validate] manifest_path={manifest_path}"); print(f"[025_validate] report_path={rp}"); print(f"[025_validate] panel_rows={metrics.get('panel_rows')} unique_months={metrics.get('unique_months')}")
    if problems: print(json.dumps(report, indent=2, sort_keys=True, default=str)); raise SystemExit("Step 005 validation failed")
    print("[025_validate] passed"); return 0
if __name__ == "__main__": sys.exit(main())
