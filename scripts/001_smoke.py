from __future__ import annotations

import argparse
import importlib.metadata as metadata
from pathlib import Path
import sys

REQUIRED_DIRS = [
    'configs',
    'scripts',
    'src/ownership_fragility',
    'tests',
    'docs',
    'artifacts/schema',
    'artifacts/logs',
    'data/raw',
    'data/interim',
    'data/processed',
]

KEY_PACKAGES = [
    'pandas',
    'numpy',
    'scipy',
    'wrds',
    'sqlalchemy',
    'psycopg2-binary',
    'pyarrow',
    'duckdb',
    'polars',
    'scikit-learn',
    'statsmodels',
    'linearmodels',
    'lightgbm',
    'xgboost',
]


def package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f'missing_or_unknown: {exc}'


def assert_no_blockquote_contamination(root: Path) -> None:
    checked_suffixes = {'.md', '.py', '.toml', '.yaml', '.yml', '.gitignore'}
    bad_lines: list[str] = []

    skip_parts = {
        '.git',
        'logs',
        'scripts/cluster_runs',
        'data/raw',
        'data/interim',
        'data/processed',
        'data/external',
        'artifacts/logs',
        'artifacts/schema',
        'artifacts/processed',
        'artifacts/model_runs',
    }

    for path in root.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel == part or rel.startswith(part + '/') for part in skip_parts):
            continue
        if path.name == '.gitignore' or path.suffix in checked_suffixes:
            try:
                lines = path.read_text(encoding='utf-8').splitlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines, start=1):
                if line.startswith('> '):
                    bad_lines.append(f'{rel}:{i}:{line[:120]}')

    if bad_lines:
        joined = '\n'.join(bad_lines[:50])
        raise SystemExit(f'Found copied blockquote-prefix contamination:\n{joined}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    print(f'Smoke root: {root}')

    missing = [d for d in REQUIRED_DIRS if not (root / d).exists()]
    if missing:
        raise SystemExit(f'Missing required directories: {missing}')

    readme = (root / 'README.md').read_text(encoding='utf-8')
    if 'Trading the Production Network' in readme:
        raise SystemExit('README contains stale project title from another template.')
    if 'Filing-Date-Clean Ownership Network Fragility' not in readme:
        raise SystemExit('README does not contain the correct project title.')

    assert_no_blockquote_contamination(root)

    print('Package versions:')
    for package in KEY_PACKAGES:
        print(f'  {package}: {package_version(package)}')

    print('Smoke check passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
