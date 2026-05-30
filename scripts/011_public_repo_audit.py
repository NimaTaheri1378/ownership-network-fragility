from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, re, subprocess

FORBIDDEN_PREFIXES = [
    "data/raw/", "data/interim/", "data/processed/", "data/external/",
    "artifacts/logs/", "artifacts/schema/", "artifacts/model_runs/", "artifacts/processed/",
    "artifacts/figures_static/", "artifacts/figures_interactive/",
]
FORBIDDEN_EXTS = {".parquet", ".feather", ".duckdb", ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5"}
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".gitignore", ".txt", ".cff"}
SECRET_PATTERNS = [re.compile(r"password\s*=\s*['\"]", re.I), re.compile(r"api[_-]?key\s*=\s*['\"]", re.I), re.compile(r"BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY")]


def git_lines(root: Path, args: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    problems, warnings = [], []

    required = ["README.md", "DATA_ACCESS.md", "CITATION.cff", ".gitignore", "docs/index.md", "docs/results.md", "docs/robustness.md", "docs/reproducibility.md", "docs/model_card.md"]
    for rel in required:
        if not (root / rel).exists():
            problems.append(f"missing required public file: {rel}")

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/") or rel.startswith("site/"):
            continue
        if p.name == ".gitignore" or p.suffix in TEXT_SUFFIXES:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), start=1):
                if line.startswith("> "):
                    problems.append(f"copied prompt/block quote prefix: {rel}:{i}")
                for pat in SECRET_PATTERNS:
                    if pat.search(line):
                        problems.append(f"possible secret pattern: {rel}:{i}")

    for p in root.glob("*.sh"):
        if p.stat().st_size == 0:
            problems.append(f"zero-byte shell script in project root: {p.name}")

    tracked = git_lines(root, ["ls-files"])
    for rel in tracked:
        if any(rel.startswith(x) for x in FORBIDDEN_PREFIXES) and not rel.endswith("/.gitkeep"):
            problems.append(f"forbidden tracked vendor/artifact path: {rel}")
        if Path(rel).suffix.lower() in FORBIDDEN_EXTS:
            problems.append(f"forbidden tracked binary/vendor-like extension: {rel}")

    for p in (root / "docs").rglob("*") if (root / "docs").exists() else []:
        if p.is_file() and p.stat().st_size > 15_000_000:
            warnings.append(f"large docs asset >15MB: {p.relative_to(root)}")

    report = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok" if not problems else "failed", "problems": problems, "warnings": warnings, "tracked_files_checked": len(tracked)}
    out_dir = root / "artifacts" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "011_public_repo_audit_latest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not problems else 1

if __name__ == "__main__":
    raise SystemExit(main())
