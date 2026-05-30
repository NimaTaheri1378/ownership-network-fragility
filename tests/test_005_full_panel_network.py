from __future__ import annotations
import importlib.util, os, subprocess, unittest
from pathlib import Path
class TestStep005FullPanelNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve(); path=cls.root/"scripts"/"005_full_panel_network_scale.py"; spec=importlib.util.spec_from_file_location("step005", path); assert spec and spec.loader; cls.mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.mod)
    def test_month_windows(self):
        self.assertEqual(self.mod.month_windows("2020-01-15", "2020-07-20", 3)[0], ("2020-01-01", "2020-03-31"))
    def test_cusip8(self):
        import pandas as pd
        self.assertEqual(self.mod.cusip8(pd.Series(["037833100"])).iloc[0], "03783310")
    def test_dirs_gitignored(self):
        for rel in ["data/interim/005_full_core/probe.parquet", "data/processed/005_full_panel/probe.parquet"]:
            self.assertEqual(subprocess.run(["git", "check-ignore", "-q", rel], cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode, 0, rel)
if __name__ == "__main__": unittest.main()
