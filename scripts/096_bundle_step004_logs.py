from __future__ import annotations
import argparse
from pathlib import Path
import zipfile


def add(zf: zipfile.ZipFile, path: Path, arc: str) -> None:
    if path.is_dir():
        for c in sorted(path.rglob("*")):
            if c.is_file(): zf.write(c, arcname=f"{arc}/{c.relative_to(path)}")
    elif path.is_file():
        zf.write(path, arcname=arc)


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", required=True); p.add_argument("--run-id", required=True); p.add_argument("--log-dir", required=True); a = p.parse_args()
    root = Path(a.project_root).resolve(); log_dir = Path(a.log_dir).resolve(); out = root / "artifacts" / "logs" / f"004_pilot_panel_{a.run_id}_logs_and_manifest.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        add(zf, log_dir, log_dir.name)
        for pattern in [f"004_pilot_panel_manifest_{a.run_id}*.json", "024_pilot_panel_validation_*.json"]:
            for path in sorted((root / "artifacts" / "logs").glob(pattern)): add(zf, path, f"artifacts/logs/{path.name}")
        for pattern in [f"004_pilot_panel_quality_{a.run_id}.csv", f"004_pilot_panel_monthly_coverage_{a.run_id}.csv"]:
            for path in sorted((root / "artifacts" / "tables").glob(pattern)): add(zf, path, f"artifacts/tables/{path.name}")
        for path in sorted((root / "artifacts" / "figures_static").glob(f"004_*_{a.run_id}.png")): add(zf, path, f"artifacts/figures_static/{path.name}")
        for path in sorted((root / "artifacts" / "figures_interactive").glob(f"004_*_{a.run_id}.html")): add(zf, path, f"artifacts/figures_interactive/{path.name}")
        add(zf, root / "docs" / "004_pilot_panel_network_audit.md", "docs/004_pilot_panel_network_audit.md")
    print(f"[096_bundle_step004_logs] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
