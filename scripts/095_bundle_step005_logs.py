from __future__ import annotations
import argparse
from pathlib import Path
import zipfile

def add(zf: zipfile.ZipFile, path: Path, arc: str) -> None:
    if path.exists():
        if path.is_dir():
            for c in sorted(path.rglob("*")):
                if c.is_file(): zf.write(c, arcname=f"{arc}/{c.relative_to(path)}")
        else: zf.write(path, arcname=arc)

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--project-root", required=True); p.add_argument("--run-id", required=True); p.add_argument("--log-dir", required=True); a=p.parse_args(); root=Path(a.project_root).resolve(); out=root/"artifacts"/"logs"/f"005_full_panel_{a.run_id}_logs_and_manifest.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        add(zf, Path(a.log_dir).resolve(), f"logs/005_full_panel_{a.run_id}")
        for pattern in [f"005_full_panel_manifest_{a.run_id}*.json", "025_full_panel_validation_*.json"]:
            for path in sorted((root/"artifacts"/"logs").glob(pattern)): zf.write(path, arcname=f"artifacts/logs/{path.name}")
        for pattern in [f"005_full_panel_metrics_{a.run_id}.csv", f"005_full_panel_monthly_coverage_{a.run_id}.csv"]:
            for path in sorted((root/"artifacts"/"tables").glob(pattern)): zf.write(path, arcname=f"artifacts/tables/{path.name}")
        for path in sorted((root/"artifacts"/"figures_static").glob(f"005_*_{a.run_id}.png")): zf.write(path, arcname=f"artifacts/figures_static/{path.name}")
        for path in sorted((root/"artifacts"/"figures_interactive").glob(f"005_*_{a.run_id}.html")): zf.write(path, arcname=f"artifacts/figures_interactive/{path.name}")
        add(zf, root/"docs"/"005_full_panel_network_audit.md", "docs/005_full_panel_network_audit.md"); add(zf, root/"configs"/"005_full_panel.yaml", "configs/005_full_panel.yaml")
    print(f"[095_bundle_step005_logs] wrote {out}"); return 0
if __name__ == "__main__": raise SystemExit(main())
