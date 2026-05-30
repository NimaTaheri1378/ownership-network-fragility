from __future__ import annotations
from pathlib import Path
import importlib.util
import os
import subprocess
import unittest
import pandas as pd


def load_module():
    root = Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()
    path = root / "scripts" / "004_pilot_panel_and_network.py"
    spec = importlib.util.spec_from_file_location("step004", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class TestStep004Helpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()

    def test_cusip8(self):
        got = list(self.m.cusip8(pd.Series(["037833100", " 59491810-4", None, "abc12345x"])).astype("string"))
        self.assertEqual(got[0], "03783310")
        self.assertEqual(got[1], "59491810")
        self.assertEqual(got[3], "ABC12345")

    def test_network(self):
        pos = pd.DataFrame({"month": ["2020-01"]*4, "manager_id": ["a", "a", "b", "b"], "permno": [1, 2, 2, 3], "portfolio_weight": [0.5, 0.5, 0.4, 0.6]})
        stock = pd.DataFrame({"month": ["2020-01"]*3, "permno": [1, 2, 3], "stock_sell_pressure": [0.0, 0.1, 0.2]})
        net, edges, stats = self.m.build_network(pos, stock)
        self.assertGreaterEqual(len(net), 3)
        self.assertGreater(stats["total_graph_nonzeros"], 0)

    def test_processed_data_is_gitignored(self):
        root = Path(os.environ.get("ONF_PROJECT_ROOT", Path.cwd())).resolve()
        result = subprocess.run(["git", "check-ignore", "-q", "data/processed/004_pilot_panel/probe.parquet"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        self.assertEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
