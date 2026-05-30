from __future__ import annotations

from pathlib import Path
import os
import subprocess
import unittest


class TestScaffold(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()

    def test_correct_project_title(self) -> None:
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        self.assertIn("Filing-Date-Clean Ownership Network Fragility", readme)
        self.assertNotIn("Trading the Production Network", readme)

    def test_required_directories_exist(self) -> None:
        for relative in [
            "configs",
            "scripts",
            "src/ownership_fragility",
            "docs",
            "tests",
            "artifacts/schema",
            "artifacts/logs",
            "data/raw",
            "data/interim",
            "data/processed",
        ]:
            self.assertTrue((self.root / relative).exists(), relative)

    def test_no_copied_blockquote_prefixes(self) -> None:
        suffixes = {".md", ".py", ".toml", ".yaml", ".yml", ".gitignore", ".sh"}
        bad = []
        for path in self.root.rglob("*"):
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
                    bad.append(f"{path.relative_to(self.root)}:{i}")
        self.assertEqual([], bad)

    def test_local_data_and_heavy_artifacts_are_gitignored(self) -> None:
        targets = [
            "data/raw/probe.parquet",
            "data/interim/003_pilot/probe.parquet",
            "data/processed/probe.parquet",
            "data/external/probe.parquet",
            "artifacts/schema/probe.json",
            "artifacts/processed/probe.parquet",
            "artifacts/model_runs/probe.json",
            "artifacts/figures_static/probe.png",
            "artifacts/figures_interactive/probe.html",
            "artifacts/logs/probe.log",
        ]
        not_ignored = []
        for target in targets:
            result = subprocess.run(
                ["git", "check-ignore", "-q", target],
                cwd=self.root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode != 0:
                not_ignored.append(target)
        self.assertEqual([], not_ignored)


if __name__ == "__main__":
    unittest.main()
