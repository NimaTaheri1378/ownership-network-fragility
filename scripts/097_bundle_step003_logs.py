from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def add_file(zipf: zipfile.ZipFile, path: Path, root: Path) -> None:
    if path.exists() and path.is_file():
        zipf.write(path, arcname=str(path.relative_to(root)))


def add_dir_files(zipf: zipfile.ZipFile, directory: Path, root: Path) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            zipf.write(path, arcname=str(path.relative_to(root)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    out_dir = root / "artifacts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"003_pilot_extract_{args.run_id}_logs_and_manifest.zip"

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        add_dir_files(zipf, log_dir, root)
        for path in sorted((root / "artifacts" / "logs").glob(f"003_pilot_extract*{args.run_id}*.json")):
            add_file(zipf, path, root)
        for path in sorted((root / "artifacts" / "logs").glob("023_pilot_extract_validation_*.json")):
            add_file(zipf, path, root)
        for path in sorted((root / "artifacts" / "tables").glob(f"003_pilot_extract_quality_{args.run_id}.csv")):
            add_file(zipf, path, root)
        add_file(zipf, root / "docs" / "003_pilot_extract_audit.md", root)
        for path in sorted((root / "artifacts" / "figures_static").glob(f"003_pilot_extract_*_{args.run_id}.png")):
            add_file(zipf, path, root)
        for path in sorted((root / "artifacts" / "figures_interactive").glob(f"003_pilot_extract_*_{args.run_id}.html")):
            add_file(zipf, path, root)

    print(f"[097_bundle_step003_logs] wrote {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
