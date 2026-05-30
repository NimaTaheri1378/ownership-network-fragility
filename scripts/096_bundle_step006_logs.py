from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def add_path(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                zf.write(child, arcname=str(Path(arcname) / child.relative_to(path)))
    else:
        zf.write(path, arcname=arcname)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--status", default="0")
    a = p.parse_args()
    root = Path(a.project_root).resolve()
    log_dir = Path(a.log_dir).resolve()
    out = root / "artifacts" / "logs" / f"006_baseline_signals_{a.run_id}_logs_and_results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        add_path(zf, log_dir, f"logs/006_baseline_signals_{a.run_id}")
        for pattern in [f"006_baseline_signal_manifest_{a.run_id}*.json", "026_baseline_signal_validation_*.json"]:
            for path in sorted((root / "artifacts" / "logs").glob(pattern)):
                add_path(zf, path, f"artifacts/logs/{path.name}")
        for pattern in [
            f"006_feature_coverage_{a.run_id}.csv",
            f"006_decile_summary_{a.run_id}.csv",
            f"006_spread_summary_{a.run_id}.csv",
            f"006_rank_ic_summary_{a.run_id}.csv",
            f"006_feature_correlation_{a.run_id}.csv",
            f"006_spread_monthly_{a.run_id}.csv",
            f"006_rank_ic_monthly_{a.run_id}.csv",
        ]:
            for path in sorted((root / "artifacts" / "tables").glob(pattern)):
                add_path(zf, path, f"artifacts/tables/{path.name}")
        for path in sorted((root / "artifacts" / "figures_static").glob(f"006_*_{a.run_id}.png")):
            add_path(zf, path, f"artifacts/figures_static/{path.name}")
        for path in sorted((root / "artifacts" / "figures_interactive").glob(f"006_*_{a.run_id}.html")):
            add_path(zf, path, f"artifacts/figures_interactive/{path.name}")
        add_path(zf, root / "docs" / "006_baseline_signal_results.md", "docs/006_baseline_signal_results.md")
        add_path(zf, root / "configs" / "006_baseline_signals.yaml", "configs/006_baseline_signals.yaml")
    print(f"[096_bundle_step006_logs] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
