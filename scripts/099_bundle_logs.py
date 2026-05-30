from __future__ import annotations

import argparse
from pathlib import Path
import tarfile


def add_if_exists(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    if path.exists():
        tar.add(path, arcname=arcname)


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
    bundle = out_dir / f"001_phase0_{args.run_id}_logs_and_schema.tar.gz"

    with tarfile.open(bundle, "w:gz") as tar:
        add_if_exists(tar, log_dir, f"logs/001_phase0_{args.run_id}")
        schema_dir = root / "artifacts" / "schema"
        for path in schema_dir.glob(f"*{args.run_id}*"):
            tar.add(path, arcname=f"artifacts/schema/{path.name}")
        for path in out_dir.glob("020_phase0_validation_*.json"):
            tar.add(path, arcname=f"artifacts/logs/{path.name}")

    print(f"[099_bundle_logs] wrote {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
