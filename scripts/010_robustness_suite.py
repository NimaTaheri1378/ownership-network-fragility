from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import math
import os
import socket
import traceback
import zipfile

import duckdb
import numpy as np
import pandas as pd


TARGET_ANNUALIZER = {"fwd_ret_1m": 12.0, "fwd_ret_3m": 4.0}
TARGET_SQRT_ANNUALIZER = {"fwd_ret_1m": math.sqrt(12.0), "fwd_ret_3m": math.sqrt(4.0)}
MODELS = ["ridge_all", "lgbm_all", "lgbm_no_network", "lgbm_network_pressure"]
STRATIFIER_SPECS = {
    "mktcap_proxy": ("size", ["small", "mid", "large"]),
    "vol": ("volatility", ["low_vol", "mid_vol", "high_vol"]),
    "fragility_proxy": ("fragility", ["low_fragility", "mid_fragility", "high_fragility"]),
    "network_weighted_degree": ("network_weighted_degree", ["low_network", "mid_network", "high_network"]),
    "stock_sell_pressure": ("sell_pressure", ["low_sell_pressure", "mid_sell_pressure", "high_sell_pressure"]),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def qpath(path: Path) -> str:
    return str(path).replace("'", "''")


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    if p.exists():
        return p

    text = str(p)
    swaps = [
        ("~/", "~/"),
        ("~/", "~/"),
    ]
    for old, new in swaps:
        if text.startswith(old):
            alt = Path(new + text[len(old):])
            if alt.exists():
                return alt
    return p


def latest_json(root: Path, pattern: str) -> Path | None:
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if "FAILED" not in p.name]
    return candidates[0] if candidates else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def newey_west_t(values: pd.Series, lag: int = 6) -> tuple[float, float, float]:
    x = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(x))
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    if n == 1:
        return mean, np.nan, np.nan

    u = (x - mean).to_numpy(dtype="float64")
    gamma0 = float(np.dot(u, u) / n)
    var = gamma0
    L = min(lag, n - 1)
    for ell in range(1, L + 1):
        weight = 1.0 - ell / (L + 1.0)
        gamma = float(np.dot(u[ell:], u[:-ell]) / n)
        var += 2.0 * weight * gamma

    var_mean = max(var / n, 0.0)
    se = math.sqrt(var_mean)
    t = mean / se if se > 0 else np.nan
    return mean, se, t


def normalize_month(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), errors="coerce")
    out = parsed.dt.to_period("M").astype(str)
    missing = parsed.isna()
    if missing.any():
        out.loc[missing] = series.astype(str).str.slice(0, 7).loc[missing]
    return out


def assign_regime(month: str) -> str:
    year = int(str(month)[:4])
    if year <= 2016:
        return "2012_2016_early_oos"
    if year <= 2019:
        return "2017_2019_pre_covid"
    if year == 2020:
        return "2020_covid"
    if year <= 2022:
        return "2021_2022_rate_hike"
    return "2023_2025_recent"


def metric_one_group(group: pd.DataFrame, return_col: str = "actual_return") -> dict:
    target = str(group["target"].iloc[0])
    annualizer = TARGET_ANNUALIZER.get(target, 12.0)
    sqrt_ann = TARGET_SQRT_ANNUALIZER.get(target, math.sqrt(12.0))

    months = []
    ic_values = []
    spread_values = []
    top_counts = []
    bot_counts = []
    all_hits = []

    for month, mdf in group.groupby("month", sort=True):
        use = mdf[["prediction", return_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(use) < 30:
            continue

        months.append(month)

        if use["prediction"].nunique() > 1 and use[return_col].nunique() > 1:
            ic_values.append(float(use["prediction"].corr(use[return_col], method="spearman")))
        else:
            ic_values.append(np.nan)

        rank_pct = use["prediction"].rank(method="first", pct=True)
        top = use.loc[rank_pct >= 0.9, return_col]
        bot = use.loc[rank_pct <= 0.1, return_col]
        if len(top) and len(bot):
            spread_values.append(float(top.mean() - bot.mean()))
            top_counts.append(int(len(top)))
            bot_counts.append(int(len(bot)))
        else:
            spread_values.append(np.nan)
            top_counts.append(0)
            bot_counts.append(0)

        all_hits.append(float(((use["prediction"] > 0) == (use[return_col] > 0)).mean()))

    ic = pd.Series(ic_values, dtype="float64")
    spread = pd.Series(spread_values, dtype="float64")
    hit = pd.Series(all_hits, dtype="float64")

    ic_mean, ic_se, ic_t = newey_west_t(ic)
    spread_mean, spread_se, spread_t = newey_west_t(spread)
    hit_mean, _, hit_t = newey_west_t(hit)

    return {
        "rows": int(len(group)),
        "months": int(len(pd.Series(months).dropna().unique())),
        "rank_ic_mean": ic_mean,
        "rank_ic_t_hac6": ic_t,
        "spread_mean": spread_mean,
        "spread_t_hac6": spread_t,
        "annualized_spread_approx": spread_mean * annualizer if pd.notna(spread_mean) else np.nan,
        "hit_rate_mean": hit_mean,
        "hit_rate_t_hac6": hit_t,
        "avg_top_names": float(np.mean(top_counts)) if top_counts else np.nan,
        "avg_bottom_names": float(np.mean(bot_counts)) if bot_counts else np.nan,
        "annualized_spread_to_vol_proxy": (
            float(spread_mean / spread.std(ddof=1) * sqrt_ann)
            if len(spread.dropna()) > 2 and spread.std(ddof=1) > 0 and pd.notna(spread_mean)
            else np.nan
        ),
    }


def metrics_by(df: pd.DataFrame, group_cols: list[str], return_col: str = "actual_return") -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in df.groupby(group_cols, observed=True, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = {col: val for col, val in zip(group_cols, keys)}
        result.update(metric_one_group(group, return_col=return_col))
        rows.append(result)
    return pd.DataFrame(rows)


def add_monthly_strata(df: pd.DataFrame, col: str, out_col: str, labels: list[str]) -> pd.DataFrame:
    base = df[["month", "permno", col]].drop_duplicates(["month", "permno"]).copy()
    base[out_col] = "missing"

    valid = base[col].replace([np.inf, -np.inf], np.nan).notna()
    if valid.any():
        pct = base.loc[valid].groupby("month", observed=True)[col].rank(method="first", pct=True)
        low = pct <= (1.0 / 3.0)
        high = pct > (2.0 / 3.0)
        base.loc[valid, out_col] = labels[1]
        base.loc[pct.index[low], out_col] = labels[0]
        base.loc[pct.index[high], out_col] = labels[2]

    return df.merge(base[["month", "permno", out_col]], on=["month", "permno"], how="left")


def read_optional_table(path_text: str | None) -> pd.DataFrame:
    path = resolve_path(path_text)
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def make_figures(
    root: Path,
    run_id: str,
    time_metrics: pd.DataFrame,
    stratum_metrics: pd.DataFrame,
    placebo_metrics: pd.DataFrame,
    model_comparison: pd.DataFrame,
    factor_attr: pd.DataFrame,
) -> dict:
    import matplotlib.pyplot as plt

    fig_dir = root / "artifacts" / "figures_static"
    html_dir = root / "artifacts" / "figures_interactive"
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 260,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.titlesize": 14,
        }
    )

    figures: dict[str, str] = {}

    p = time_metrics[
        (time_metrics["target"] == "fwd_ret_1m")
        & (time_metrics["regime"] != "all_oos")
    ].copy()
    if not p.empty:
        pivot = p.pivot_table(
            index="regime",
            columns="model",
            values="annualized_spread_approx",
            aggfunc="mean",
        )
        order = [
            "2012_2016_early_oos",
            "2017_2019_pre_covid",
            "2020_covid",
            "2021_2022_rate_hike",
            "2023_2025_recent",
        ]
        pivot = pivot.reindex([x for x in order if x in pivot.index])
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.axhline(0, linewidth=1)
        ax.set_title("Step 010: OOS 1M portfolio spread by regime")
        ax.set_ylabel("Annualized spread approximation")
        ax.set_xlabel("")
        ax.legend(loc="best")
        fig = ax.get_figure()
        fig.tight_layout()
        path = fig_dir / f"010_regime_spreads_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figures["regime_spreads_1m"] = str(path)

    p = stratum_metrics[
        (stratum_metrics["target"] == "fwd_ret_1m")
        & (stratum_metrics["model"] == "ridge_all")
    ].copy()
    if not p.empty:
        p["label"] = p["stratifier"] + ": " + p["stratum"]
        p = p.sort_values("annualized_spread_approx")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(p["label"], p["annualized_spread_approx"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Step 010: Ridge 1M spread across robustness strata")
        ax.set_xlabel("Annualized spread approximation")
        fig.tight_layout()
        path = fig_dir / f"010_ridge_strata_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figures["ridge_strata_1m"] = str(path)

    if not placebo_metrics.empty:
        p = placebo_metrics[placebo_metrics["target"] == "fwd_ret_1m"].copy()
        if not p.empty:
            p = p.sort_values("actual_rank_ic_mean")
            x = np.arange(len(p))
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.bar(x - 0.18, p["actual_rank_ic_mean"], width=0.36, label="actual")
            ax.bar(x + 0.18, p["placebo_rank_ic_mean"], width=0.36, label="within-month permuted")
            ax.axhline(0, linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(p["model"], rotation=20, ha="right")
            ax.set_title("Step 010: Actual vs permutation-placebo rank IC, 1M")
            ax.set_ylabel("Mean monthly Spearman rank IC")
            ax.legend()
            fig.tight_layout()
            path = fig_dir / f"010_placebo_rank_ic_1m_{run_id}.png"
            fig.savefig(path)
            plt.close(fig)
            figures["placebo_rank_ic_1m"] = str(path)

    p = model_comparison[
        (model_comparison.get("target", pd.Series(dtype=str)) == "fwd_ret_1m")
        & (model_comparison.get("return_series", pd.Series(dtype=str)) == "net_return_25bps")
    ].copy()
    if not p.empty:
        p = p.sort_values("annualized_mean_return")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(p["model"], p["annualized_mean_return"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Step 010: 25 bps net annualized return by 1M OOS model")
        ax.set_xlabel("Annualized mean return")
        fig.tight_layout()
        path = fig_dir / f"010_net25_model_returns_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figures["net25_model_returns_1m"] = str(path)

    p = factor_attr[
        (factor_attr.get("target", pd.Series(dtype=str)) == "fwd_ret_1m")
        & (factor_attr.get("return_series", pd.Series(dtype=str)) == "net_return_25bps")
    ].copy()
    if not p.empty and "alpha_annualized" in p:
        p = p.sort_values("alpha_annualized")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(p["model"], p["alpha_annualized"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Step 010: Factor alpha, 25 bps net, 1M OOS model")
        ax.set_xlabel("Annualized alpha")
        fig.tight_layout()
        path = fig_dir / f"010_factor_alpha_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figures["factor_alpha_1m"] = str(path)

    html_path = html_dir / f"010_robustness_dashboard_{run_id}.html"
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Step 010 robustness dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; max-width: 1250px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; }}
th {{ background: #f4f4f4; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
h1, h2 {{ margin-top: 28px; }}
</style>
</head>
<body>
<h1>Step 010 robustness suite</h1>
<p>Run ID: <code>{run_id}</code></p>
<h2>Model comparison from Step 009</h2>
{model_comparison.to_html(index=False) if not model_comparison.empty else "<p>No model comparison table.</p>"}
<h2>Time-regime robustness</h2>
{time_metrics.to_html(index=False) if not time_metrics.empty else "<p>No time-regime table.</p>"}
<h2>Subsample robustness</h2>
{stratum_metrics.to_html(index=False) if not stratum_metrics.empty else "<p>No stratum table.</p>"}
<h2>Permutation-placebo diagnostics</h2>
{placebo_metrics.to_html(index=False) if not placebo_metrics.empty else "<p>No placebo table.</p>"}
<h2>Factor attribution from Step 009</h2>
{factor_attr.to_html(index=False) if not factor_attr.empty else "<p>No factor attribution table.</p>"}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    figures["interactive_dashboard"] = str(html_path)

    return figures


def bundle_outputs(root: Path, run_id: str, log_dir: Path) -> Path:
    bundle = root / "artifacts" / "logs" / f"010_robustness_{run_id}_logs_and_results.zip"
    if bundle.exists():
        bundle.unlink()

    include_patterns = [
        f"artifacts/logs/010_robustness_manifest_{run_id}.json",
        f"artifacts/tables/010_*_{run_id}.csv",
        f"artifacts/figures_static/010_*_{run_id}.png",
        f"artifacts/figures_interactive/010_*_{run_id}.html",
        "docs/010_robustness_suite.md",
        "scripts/010_robustness_suite.py",
    ]

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if log_dir.exists():
            for p in log_dir.rglob("*"):
                if p.is_file():
                    z.write(p, f"logs/{log_dir.name}/{p.relative_to(log_dir)}")

        for pattern in include_patterns:
            for p in root.glob(pattern):
                if p.is_file():
                    z.write(p, str(p.relative_to(root)))

    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-jobs", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = args.run_id
    log_dir = root / "logs" / f"010_robustness_{run_id}"

    if not socket.gethostname().startswith("compute-node"):
        raise SystemExit("Step 010 is constrained to compute-node only.")

    table_dir = root / "artifacts" / "tables"
    log_dir_art = root / "artifacts" / "logs"
    docs_dir = root / "docs"
    for p in [table_dir, log_dir_art, docs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    step008_manifest_path = latest_json(root, "artifacts/logs/008_oos_ml_manifest_*.json")
    step009_manifest_path = latest_json(root, "artifacts/logs/009_portfolio_costs_manifest_*.json")
    if step008_manifest_path is None:
        raise FileNotFoundError("No successful Step 008 manifest found.")
    if step009_manifest_path is None:
        raise FileNotFoundError("No successful Step 009 manifest found.")

    step008_manifest = load_json(step008_manifest_path)
    step009_manifest = load_json(step009_manifest_path)

    pred_path = resolve_path(step008_manifest.get("local_prediction_path"))
    panel_path = resolve_path(step008_manifest.get("source_step005_panel_local_only"))

    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"Could not resolve Step 008 predictions: {step008_manifest.get('local_prediction_path')}")
    if panel_path is None or not panel_path.exists():
        raise FileNotFoundError(f"Could not resolve Step 005 panel: {step008_manifest.get('source_step005_panel_local_only')}")

    print(f"[010] Step 008 manifest: {step008_manifest_path}")
    print(f"[010] Step 009 manifest: {step009_manifest_path}")
    print(f"[010] Predictions:       {pred_path}")
    print(f"[010] Panel:             {panel_path}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.n_jobs}")

    pred_cols = set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{qpath(pred_path)}')").fetchdf()["column_name"].astype(str)
    )
    panel_cols = set(
        con.execute(f"DESCRIBE SELECT * FROM read_parquet('{qpath(panel_path)}')").fetchdf()["column_name"].astype(str)
    )

    required_pred = {"month", "permno", "model", "target", "prediction", "fwd_ret_1m", "fwd_ret_3m"}
    missing_pred = sorted(required_pred - pred_cols)
    if missing_pred:
        raise RuntimeError(f"Step 008 predictions missing required columns: {missing_pred}")

    stratifier_cols = [c for c in STRATIFIER_SPECS if c in panel_cols]
    if not stratifier_cols:
        raise RuntimeError("No expected Step 005 stratifier columns found in panel.")

    panel_select = ", ".join([f"q.{c}" for c in stratifier_cols])
    if panel_select:
        panel_select = ", " + panel_select

    query = f"""
    SELECT *
    FROM (
      SELECT
        CAST(p.month AS VARCHAR) AS month,
        CAST(p.permno AS BIGINT) AS permno,
        CAST(p.model AS VARCHAR) AS model,
        CAST(p.target AS VARCHAR) AS target,
        CAST(p.prediction AS DOUBLE) AS prediction,
        CAST(CASE WHEN p.target = 'fwd_ret_1m' THEN p.fwd_ret_1m ELSE p.fwd_ret_3m END AS DOUBLE) AS actual_return,
        CAST(p.test_year AS INTEGER) AS test_year
        {panel_select}
      FROM read_parquet('{qpath(pred_path)}') AS p
      LEFT JOIN (
        SELECT
          CAST(month AS VARCHAR) AS month,
          CAST(permno AS BIGINT) AS permno,
          {", ".join(stratifier_cols)}
        FROM read_parquet('{qpath(panel_path)}')
      ) AS q
        ON CAST(p.month AS VARCHAR) = q.month
       AND CAST(p.permno AS BIGINT) = q.permno
      WHERE p.model IN ({", ".join("'" + m + "'" for m in MODELS)})
    ) AS x
    WHERE prediction IS NOT NULL
      AND actual_return IS NOT NULL
    """

    df = con.execute(query).fetchdf()
    con.close()

    print(f"[010] joined rows: {len(df):,}")
    print(f"[010] stratifier columns: {stratifier_cols}")

    if df.empty:
        raise RuntimeError("Joined prediction/panel dataset is empty.")

    df["month"] = normalize_month(df["month"])
    df["year"] = df["month"].str.slice(0, 4).astype(int)
    df["regime"] = df["month"].map(assign_regime)

    for c in ["prediction", "actual_return"] + stratifier_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    all_df = df.copy()
    all_df["regime"] = "all_oos"
    time_df = pd.concat([df, all_df], ignore_index=True)
    time_metrics = metrics_by(time_df, ["model", "target", "regime"])

    stratum_frames = []
    stratified = df.copy()
    for col in stratifier_cols:
        stratifier_name, labels = STRATIFIER_SPECS[col]
        out_col = f"stratum_{stratifier_name}"
        print(f"[010] assigning monthly strata: {stratifier_name} from {col}")
        stratified = add_monthly_strata(stratified, col, out_col, labels)
        tmp = stratified[stratified[out_col].ne("missing")].copy()
        if tmp.empty:
            continue
        metrics = metrics_by(tmp, ["model", "target", out_col])
        metrics = metrics.rename(columns={out_col: "stratum"})
        metrics.insert(0, "source_column", col)
        metrics.insert(0, "stratifier", stratifier_name)
        stratum_frames.append(metrics)

    stratum_metrics = pd.concat(stratum_frames, ignore_index=True) if stratum_frames else pd.DataFrame()

    # One deterministic within-month return permutation placebo.
    print("[010] running deterministic within-month permutation placebo")
    placebo = df[["month", "model", "target", "prediction", "actual_return"]].copy()
    placebo["placebo_return"] = np.nan
    rng = np.random.default_rng(20260529)
    for _, idx in placebo.groupby(["model", "target", "month"], observed=True).groups.items():
        arr = placebo.loc[idx, "actual_return"].to_numpy(dtype="float64", copy=True)
        rng.shuffle(arr)
        placebo.loc[idx, "placebo_return"] = arr

    actual_all = time_metrics[time_metrics["regime"].eq("all_oos")][
        ["model", "target", "rank_ic_mean", "rank_ic_t_hac6", "annualized_spread_approx", "spread_t_hac6"]
    ].copy()
    actual_all = actual_all.rename(
        columns={
            "rank_ic_mean": "actual_rank_ic_mean",
            "rank_ic_t_hac6": "actual_rank_ic_t_hac6",
            "annualized_spread_approx": "actual_annualized_spread_approx",
            "spread_t_hac6": "actual_spread_t_hac6",
        }
    )
    placebo_metric = metrics_by(placebo, ["model", "target"], return_col="placebo_return")
    placebo_metric = placebo_metric.rename(
        columns={
            "rank_ic_mean": "placebo_rank_ic_mean",
            "rank_ic_t_hac6": "placebo_rank_ic_t_hac6",
            "annualized_spread_approx": "placebo_annualized_spread_approx",
            "spread_t_hac6": "placebo_spread_t_hac6",
        }
    )
    placebo_metrics = actual_all.merge(
        placebo_metric[
            [
                "model",
                "target",
                "placebo_rank_ic_mean",
                "placebo_rank_ic_t_hac6",
                "placebo_annualized_spread_approx",
                "placebo_spread_t_hac6",
            ]
        ],
        on=["model", "target"],
        how="left",
    )
    placebo_metrics["rank_ic_gap_actual_minus_placebo"] = (
        placebo_metrics["actual_rank_ic_mean"] - placebo_metrics["placebo_rank_ic_mean"]
    )
    placebo_metrics["spread_gap_actual_minus_placebo"] = (
        placebo_metrics["actual_annualized_spread_approx"] - placebo_metrics["placebo_annualized_spread_approx"]
    )

    # Step 009 aggregate economic results.
    perf = read_optional_table(step009_manifest.get("tables", {}).get("performance_summary"))
    cost = read_optional_table(step009_manifest.get("tables", {}).get("cost_sensitivity"))
    factor_attr = read_optional_table(step009_manifest.get("tables", {}).get("factor_attribution"))

    if not perf.empty:
        model_comparison = perf.copy()
        if "return_series" in model_comparison:
            model_comparison = model_comparison[model_comparison["return_series"].eq("net_return_25bps")].copy()
        for col in ["annualized_mean_return", "sharpe_approx", "nw_t_mean", "cumulative_return", "max_drawdown"]:
            if col in model_comparison:
                model_comparison[col] = pd.to_numeric(model_comparison[col], errors="coerce")
        if {"target", "sharpe_approx"}.issubset(model_comparison.columns):
            model_comparison["rank_by_sharpe_within_target"] = (
                model_comparison.groupby("target")["sharpe_approx"].rank(ascending=False, method="min")
            )
    else:
        model_comparison = pd.DataFrame()

    # Combine into a compact scorecard.
    scorecard_rows = []
    all_time = time_metrics[time_metrics["regime"].eq("all_oos")].copy()
    for _, row in all_time.iterrows():
        model = row["model"]
        target = row["target"]
        entry = {
            "model": model,
            "target": target,
            "oos_rank_ic_mean": row.get("rank_ic_mean"),
            "oos_rank_ic_t_hac6": row.get("rank_ic_t_hac6"),
            "oos_annualized_spread_approx": row.get("annualized_spread_approx"),
            "oos_spread_t_hac6": row.get("spread_t_hac6"),
        }

        if not model_comparison.empty:
            mrow = model_comparison[
                (model_comparison["model"] == model)
                & (model_comparison["target"] == target)
            ]
            if not mrow.empty:
                mrow = mrow.iloc[0]
                for c in ["annualized_mean_return", "sharpe_approx", "nw_t_mean", "cumulative_return", "max_drawdown"]:
                    if c in mrow:
                        entry[f"net25_{c}"] = mrow[c]

        if not factor_attr.empty:
            frow = factor_attr[
                (factor_attr["model"] == model)
                & (factor_attr["target"] == target)
                & (factor_attr["return_series"] == "net_return_25bps")
            ]
            if not frow.empty:
                frow = frow.iloc[0]
                for c in ["alpha_annualized", "alpha_t_hac6", "r2", "beta_mktrf", "beta_smb", "beta_hml"]:
                    if c in frow:
                        entry[f"factor_{c}"] = frow[c]

        prow = placebo_metrics[(placebo_metrics["model"] == model) & (placebo_metrics["target"] == target)]
        if not prow.empty:
            prow = prow.iloc[0]
            for c in [
                "placebo_rank_ic_mean",
                "placebo_rank_ic_t_hac6",
                "placebo_annualized_spread_approx",
                "placebo_spread_t_hac6",
                "rank_ic_gap_actual_minus_placebo",
                "spread_gap_actual_minus_placebo",
            ]:
                entry[c] = prow[c]

        scorecard_rows.append(entry)

    scorecard = pd.DataFrame(scorecard_rows)

    # Network incremental summary: lgbm_all vs lgbm_no_network in economic net25 performance.
    network_delta_rows = []
    if not model_comparison.empty and {"model", "target", "annualized_mean_return"}.issubset(model_comparison.columns):
        for target, g in model_comparison.groupby("target", sort=True):
            lookup = g.set_index("model")
            if "lgbm_all" in lookup.index and "lgbm_no_network" in lookup.index:
                row = {
                    "target": target,
                    "comparison": "lgbm_all_minus_lgbm_no_network",
                    "delta_net25_annualized_mean_return": float(lookup.loc["lgbm_all", "annualized_mean_return"] - lookup.loc["lgbm_no_network", "annualized_mean_return"]),
                    "delta_net25_sharpe": float(lookup.loc["lgbm_all", "sharpe_approx"] - lookup.loc["lgbm_no_network", "sharpe_approx"]) if "sharpe_approx" in lookup else np.nan,
                    "delta_net25_cumulative_return": float(lookup.loc["lgbm_all", "cumulative_return"] - lookup.loc["lgbm_no_network", "cumulative_return"]) if "cumulative_return" in lookup else np.nan,
                }
                network_delta_rows.append(row)
    network_delta = pd.DataFrame(network_delta_rows)

    # Write aggregate tables.
    outputs = {
        "time_regime_metrics": table_dir / f"010_time_regime_metrics_{run_id}.csv",
        "stratum_metrics": table_dir / f"010_stratum_metrics_{run_id}.csv",
        "placebo_metrics": table_dir / f"010_permutation_placebo_{run_id}.csv",
        "model_comparison": table_dir / f"010_model_comparison_net25_{run_id}.csv",
        "network_delta": table_dir / f"010_network_incremental_delta_{run_id}.csv",
        "scorecard": table_dir / f"010_robustness_scorecard_{run_id}.csv",
    }
    time_metrics.to_csv(outputs["time_regime_metrics"], index=False)
    stratum_metrics.to_csv(outputs["stratum_metrics"], index=False)
    placebo_metrics.to_csv(outputs["placebo_metrics"], index=False)
    model_comparison.to_csv(outputs["model_comparison"], index=False)
    network_delta.to_csv(outputs["network_delta"], index=False)
    scorecard.to_csv(outputs["scorecard"], index=False)

    if not cost.empty:
        cost.to_csv(table_dir / f"010_copied_009_cost_sensitivity_{run_id}.csv", index=False)
    if not factor_attr.empty:
        factor_attr.to_csv(table_dir / f"010_copied_009_factor_attribution_{run_id}.csv", index=False)

    figures = make_figures(root, run_id, time_metrics, stratum_metrics, placebo_metrics, model_comparison, factor_attr)

    # Decide status.
    problems: list[str] = []
    if len(df) < 1_000_000:
        problems.append("joined OOS prediction rows are below expected threshold")
    if time_metrics.empty:
        problems.append("time-regime metrics are empty")
    if stratum_metrics.empty:
        problems.append("stratum metrics are empty")
    if placebo_metrics.empty:
        problems.append("placebo metrics are empty")
    if not placebo_metrics.empty:
        bad_placebo = placebo_metrics[
            (placebo_metrics["target"].eq("fwd_ret_1m"))
            & (placebo_metrics["placebo_rank_ic_mean"].abs() > 0.02)
        ]
        if not bad_placebo.empty:
            problems.append("one or more 1M permutation placebo ICs exceed 0.02 in absolute value")
    status = "ok" if not problems else "needs_attention"

    headline = {}
    if not model_comparison.empty:
        one = model_comparison[model_comparison["target"].eq("fwd_ret_1m")].copy()
        if "sharpe_approx" in one and not one.empty:
            best = one.sort_values("sharpe_approx", ascending=False).iloc[0].to_dict()
            headline["best_net25_1m_by_sharpe"] = best

    manifest = {
        "run_id": run_id,
        "status": status,
        "problems": problems,
        "created_utc": now_utc(),
        "host": socket.gethostname(),
        "n_jobs": args.n_jobs,
        "step008_manifest": str(step008_manifest_path),
        "step009_manifest": str(step009_manifest_path),
        "prediction_rows_joined": int(len(df)),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "targets": sorted(df["target"].dropna().unique().tolist()),
        "oos_months": int(df["month"].nunique()),
        "oos_stocks": int(df["permno"].nunique()),
        "stratifier_columns": stratifier_cols,
        "tables": {k: str(v) for k, v in outputs.items()},
        "figures": figures,
        "headline": headline,
        "notes": [
            "All bundled outputs are aggregate tables, figures, logs, documentation, and scripts only.",
            "Local WRDS-derived panels and OOS prediction Parquets remain local-only and gitignored.",
            "Permutation placebo shuffles realized returns within each model-target-month cell.",
        ],
    }

    manifest_path = log_dir_art / f"010_robustness_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    docs_path = docs_dir / "010_robustness_suite.md"
    docs_path.write_text(
        f"""# Step 010 robustness suite

- Run ID: `{run_id}`
- Status: `{status}`
- Host: `{socket.gethostname()}`
- Step 008 manifest: `{step008_manifest_path.name}`
- Step 009 manifest: `{step009_manifest_path.name}`
- Joined OOS prediction rows: `{len(df):,}`
- OOS months: `{df["month"].nunique()}`
- OOS stocks: `{df["permno"].nunique()}`

This step stress-tests the out-of-sample signal using time-regime splits, size/liquidity/network strata, a deterministic within-month permutation placebo, transaction-cost summaries, and factor-attribution summaries.

Vendor-derived panels and name-month predictions remain local-only and are not bundled.
""",
        encoding="utf-8",
    )

    bundle = bundle_outputs(root, run_id, log_dir)

    print(f"[010] status: {status}")
    if problems:
        print("[010] problems:")
        for p in problems:
            print(f"  - {p}")
    print(f"[010] wrote manifest: {manifest_path}")
    print(f"[010] wrote bundle:   {bundle}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        root = Path(".")
        run_id = os.environ.get("ONF_RUN_ID", "unknown")
        fail_path = root / "artifacts" / "logs" / f"010_robustness_manifest_{run_id}_FAILED.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "created_utc": now_utc(),
                    "host": socket.gethostname(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"[010] FAILED. Wrote failure manifest: {fail_path}")
        traceback.print_exc()
        raise
