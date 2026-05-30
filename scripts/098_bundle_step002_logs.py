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

    root = Path(args.project_root).expanduser()
    log_dir = Path(args.log_dir).expanduser()
    out_dir = root / "artifacts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"002_schema_contract_{args.run_id}_logs_and_contract.tar.gz"

    with tarfile.open(bundle, "w:gz") as tar:
        add_if_exists(tar, log_dir, f"logs/002_schema_contract_{args.run_id}")
        artifact_log_dir = root / "artifacts" / "logs" / f"002_schema_contract_{args.run_id}"
        add_if_exists(tar, artifact_log_dir, f"artifacts/logs/002_schema_contract_{args.run_id}")
        for pattern in [
            f"002_schema_contract_{args.run_id}*",
            "schema_discovery_full_*.json",
            "schema_discovery_full_*_summary.csv",
        ]:
            for path in (root / "artifacts" / "schema").glob(pattern):
                tar.add(path, arcname=f"artifacts/schema/{path.name}")
        for pattern in ["022_schema_contract_validation_*.json"]:
            for path in out_dir.glob(pattern):
                tar.add(path, arcname=f"artifacts/logs/{path.name}")
        for path in (root / "artifacts" / "figures_static").glob(f"002_schema_contract_coverage_{args.run_id}.*"):
            tar.add(path, arcname=f"artifacts/figures_static/{path.name}")
        for path in (root / "artifacts" / "figures_interactive").glob(f"002_schema_contract_{args.run_id}.*"):
            tar.add(path, arcname=f"artifacts/figures_interactive/{path.name}")
        add_if_exists(tar, root / "configs" / "schema_contract.yaml", "configs/schema_contract.yaml")
        add_if_exists(tar, root / "docs" / "schema_contract.md", "docs/schema_contract.md")

    print(f"[098_bundle_step002_logs] wrote {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
