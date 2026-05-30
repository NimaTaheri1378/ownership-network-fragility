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


COST_BPS = [0, 5, 10, 25, 50]
TARGET_ANNUALIZER = {"fwd_ret_1m": 12.0, "fwd_ret_3m": 4.0}
TARGET_SQRT_ANNUALIZER = {"fwd_ret_1m": math.sqrt(12.0), "fwd_ret_3m": math.sqrt(4.0)}
TOP_MODEL_ORDER = ["ridge_all", "lgbm_all", "lgbm_no_network", "lgbm_network_pressure"]


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


def newey_west_t(values: pd.Series, lag: int = 6) -> tuple[float, float, float]:
    x = pd.Series(values, dtype="float64").dropna()
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
    se = math.sqrt(max(var / n, 0.0))
    t = mean / se if se > 0 else np.nan
    return mean, se, t


def max_drawdown(returns: pd.Series) -> tuple[float, str | None, str | None]:
    r = pd.Series(returns, dtype="float64").fillna(0.0)
    if r.empty:
        return np.nan, None, None
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    trough_idx = drawdown.idxmin()
    trough = str(trough_idx) if trough_idx is not None else None
    if trough_idx is not None:
        peak_window = wealth.loc[:trough_idx]
        peak_idx = peak_window.idxmax() if not peak_window.empty else None
    else:
        peak_idx = None
    return float(drawdown.min()), str(peak_idx) if peak_idx is not None else None, trough


def summarize_return_series(
    df: pd.DataFrame,
    return_col: str,
    label: str,
    target: str,
    model: str,
) -> dict[str, object]:
    use = df[["month", return_col, "turnover_abs_change", "turnover_one_way", "n_long", "n_short"]].dropna().copy()
    annualizer = TARGET_ANNUALIZER.get(target, 12.0)
    sqrt_ann = TARGET_SQRT_ANNUALIZER.get(target, math.sqrt(12.0))
    series = pd.Series(use[return_col].to_numpy(dtype="float64"), index=use["month"].astype(str))
    mean, se, t = newey_west_t(series, lag=6)
    vol = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    ann_mean = annualizer * mean if pd.notna(mean) else np.nan
    ann_vol = sqrt_ann * vol if pd.notna(vol) else np.nan
    sharpe = ann_mean / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
    cumulative = float((1.0 + series.fillna(0.0)).prod() - 1.0) if len(series) else np.nan
    mdd, mdd_peak, mdd_trough = max_drawdown(series)
    return {
        "model": model,
        "target": target,
        "return_series": label,
        "months": int(len(series)),
        "mean_period_return": mean,
        "nw_t_mean": t,
        "annualized_mean_return": ann_mean,
        "annualized_volatility": ann_vol,
        "sharpe_approx": sharpe,
        "hit_rate": float((series > 0).mean()) if len(series) else np.nan,
        "cumulative_return": cumulative,
        "max_drawdown": mdd,
        "max_drawdown_peak_month": mdd_peak,
        "max_drawdown_trough_month": mdd_trough,
        "avg_turnover_abs_change": float(use["turnover_abs_change"].mean()) if len(use) else np.nan,
        "avg_turnover_one_way": float(use["turnover_one_way"].mean()) if len(use) else np.nan,
        "avg_n_long": float(use["n_long"].mean()) if len(use) else np.nan,
        "avg_n_short": float(use["n_short"].mean()) if len(use) else np.nan,
    }


def load_predictions(pred_path: Path, n_jobs: int) -> pd.DataFrame:
    print(f"[009] reading predictions: {pred_path}")
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={n_jobs}")
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{qpath(pred_path)}')").fetchdf()
    cols = set(desc["column_name"].astype(str))
    required = {"month", "permno", "model", "target", "prediction", "test_year"}
    missing = sorted(required - cols)
    if missing:
        raise RuntimeError(f"Step 008 prediction parquet is missing required columns: {missing}")
    targets = [x for x in ["fwd_ret_1m", "fwd_ret_3m"] if x in cols]
    if not targets:
        raise RuntimeError("No fwd_ret_1m/fwd_ret_3m columns found in Step 008 predictions.")
    select_cols = ["month", "permno", "model", "target", "test_year", "prediction"] + targets
    df = con.execute(
        f"SELECT {', '.join(select_cols)} FROM read_parquet('{qpath(pred_path)}')"
    ).fetchdf()
    con.close()
    print(f"[009] prediction rows loaded={len(df):,}")
    print(f"[009] models={sorted(df['model'].dropna().unique().tolist())}")
    print(f"[009] targets={sorted(df['target'].dropna().unique().tolist())}")

    actual = np.full(len(df), np.nan, dtype="float64")
    for target in targets:
        mask = df["target"].astype(str).eq(target)
        actual[mask.to_numpy()] = pd.to_numeric(df.loc[mask, target], errors="coerce").to_numpy(dtype="float64")
    df["actual_return"] = actual
    df = df[["month", "permno", "model", "target", "test_year", "prediction", "actual_return"]].copy()
    df["month"] = df["month"].astype(str)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual_return"] = pd.to_numeric(df["actual_return"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["month", "permno", "model", "target", "prediction", "actual_return"])
    print(f"[009] usable prediction rows={len(df):,}")
    return df


def build_portfolios(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    use = preds.copy()
    group_cols = ["model", "target", "month"]
    use["rank_pct"] = use.groupby(group_cols, observed=True)["prediction"].rank(method="first", pct=True)
    use["side"] = np.where(use["rank_pct"] >= 0.9, 1, np.where(use["rank_pct"] <= 0.1, -1, 0)).astype("int8")
    sel = use.loc[use["side"].ne(0), ["model", "target", "month", "permno", "side", "prediction", "actual_return"]].copy()
    if sel.empty:
        raise RuntimeError("No top/bottom decile names selected from predictions.")

    sel["n_side"] = sel.groupby(["model", "target", "month", "side"], observed=True)["permno"].transform("count")
    sel["weight"] = 0.5 * sel["side"] / sel["n_side"]
    sel["weighted_return"] = sel["weight"] * sel["actual_return"]

    side_ret = (
        sel.groupby(["model", "target", "month", "side"], observed=True)["actual_return"]
        .mean()
        .unstack("side")
        .rename(columns={-1: "short_leg_mean_return", 1: "long_leg_mean_return"})
        .reset_index()
    )
    side_count = (
        sel.groupby(["model", "target", "month", "side"], observed=True)["permno"]
        .count()
        .unstack("side")
        .rename(columns={-1: "n_short", 1: "n_long"})
        .reset_index()
    )
    gross = (
        sel.groupby(["model", "target", "month"], observed=True)["weighted_return"]
        .sum()
        .rename("gross_dollar_neutral_return")
        .reset_index()
    )
    port = gross.merge(side_ret, on=["model", "target", "month"], how="left").merge(
        side_count, on=["model", "target", "month"], how="left"
    )
    port["academic_spread_return"] = port["long_leg_mean_return"] - port["short_leg_mean_return"]

    turnover_rows = []
    sorted_sel = sel.sort_values(["model", "target", "month", "permno"])
    for (model, target), g in sorted_sel.groupby(["model", "target"], observed=True, sort=True):
        prev: dict[int, float] = {}
        for month, m in g.groupby("month", observed=True, sort=True):
            cur = {int(k): float(v) for k, v in zip(m["permno"].astype(int), m["weight"].astype(float))}
            keys = set(prev) | set(cur)
            abs_change = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
            turnover_rows.append(
                {
                    "model": model,
                    "target": target,
                    "month": str(month),
                    "turnover_abs_change": float(abs_change),
                    "turnover_one_way": float(0.5 * abs_change),
                    "gross_exposure": float(sum(abs(x) for x in cur.values())),
                    "net_exposure": float(sum(cur.values())),
                    "n_names": int(len(cur)),
                }
            )
            prev = cur
    turnover = pd.DataFrame(turnover_rows)
    port = port.merge(turnover, on=["model", "target", "month"], how="left")

    for bps in COST_BPS:
        # Conservative: cost is charged on absolute notional changed. Initial portfolio formation is included.
        port[f"net_return_{bps}bps"] = port["gross_dollar_neutral_return"] - (bps / 10000.0) * port["turnover_abs_change"]

    port = port.sort_values(["target", "model", "month"]).reset_index(drop=True)
    return port, sel


def performance_summary(port: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target), g in port.groupby(["model", "target"], observed=True, sort=True):
        for col in ["gross_dollar_neutral_return"] + [f"net_return_{bps}bps" for bps in COST_BPS]:
            label = col
            rows.append(summarize_return_series(g, col, label, target, model))
    out = pd.DataFrame(rows)
    sort_cols = ["target", "return_series", "sharpe_approx"]
    return out.sort_values(sort_cols, ascending=[True, True, False]).reset_index(drop=True)


def cost_sensitivity(port: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target), g in port.groupby(["model", "target"], observed=True, sort=True):
        annualizer = TARGET_ANNUALIZER.get(target, 12.0)
        for bps in COST_BPS:
            col = f"net_return_{bps}bps"
            s = g[col].dropna().astype(float)
            mean, se, t = newey_west_t(s, lag=6)
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "cost_bps": bps,
                    "months": int(len(s)),
                    "annualized_mean_return": annualizer * mean if pd.notna(mean) else np.nan,
                    "nw_t_mean": t,
                    "cumulative_return": float((1.0 + s).prod() - 1.0) if len(s) else np.nan,
                    "avg_turnover_abs_change": float(g["turnover_abs_change"].mean()),
                }
            )
    return pd.DataFrame(rows)


def try_fetch_ff_factors(root: Path, min_month: str, max_month: str) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {"status": "not_attempted", "table": None, "error": None}
    try:
        import wrds
    except Exception as exc:
        info.update({"status": "skipped_no_wrds", "error": repr(exc)})
        return pd.DataFrame(), info

    username = os.environ.get("WRDS_USERNAME") or os.environ.get("USER")
    try:
        db = wrds.Connection(wrds_username=username)
    except Exception as exc:
        info.update({"status": "connection_failed", "error": repr(exc)})
        return pd.DataFrame(), info

    try:
        cols = db.raw_sql(
            """
            select table_schema, table_name, column_name
            from information_schema.columns
            where table_schema in ('ff_all', 'ff')
            """
        )
        if cols.empty:
            info.update({"status": "no_ff_schema_rows"})
            return pd.DataFrame(), info
        cols["lc"] = cols["column_name"].str.lower()
        candidates = []
        for (schema, table), g in cols.groupby(["table_schema", "table_name"], sort=True):
            lc = set(g["lc"].tolist())
            date_candidates = [c for c in ["date", "mcaldt", "caldt", "month"] if c in lc]
            mkt_candidates = [c for c in ["mktrf", "mkt_rf", "mkt-rf", "mkt"] if c in lc]
            score = 0
            for needed in ["smb", "hml", "rf"]:
                if needed in lc:
                    score += 1
            if date_candidates and mkt_candidates and score >= 2:
                optional_score = sum(1 for c in ["umd", "mom", "rmw", "cma"] if c in lc)
                candidates.append((score + optional_score, schema, table, sorted(lc), date_candidates[0], mkt_candidates[0]))
        if not candidates:
            info.update({"status": "no_factor_table_candidate"})
            return pd.DataFrame(), info
        candidates.sort(reverse=True)
        _, schema, table, lc_cols, date_col, mkt_col = candidates[0]
        selected = [date_col, mkt_col]
        for c in ["smb", "hml", "rf", "umd", "mom", "rmw", "cma"]:
            if c in lc_cols and c not in selected:
                selected.append(c)
        sql_cols = ", ".join(selected)
        sql = f"select {sql_cols} from {schema}.{table} order by {date_col}"
        fac = db.raw_sql(sql)
        fac.columns = [c.lower().replace("-", "_") for c in fac.columns]
        if "mkt_rf" in fac.columns and "mktrf" not in fac.columns:
            fac = fac.rename(columns={"mkt_rf": "mktrf"})
        if "mom" in fac.columns and "umd" not in fac.columns:
            fac = fac.rename(columns={"mom": "umd"})
        date_name = date_col.lower().replace("-", "_")
        if date_name not in fac.columns:
            date_name = fac.columns[0]
        fac["month"] = pd.to_datetime(fac[date_name], errors="coerce").dt.strftime("%Y-%m")
        fac = fac.dropna(subset=["month"]).drop_duplicates("month", keep="last")
        fac = fac.loc[fac["month"].between(min_month, max_month)].copy()
        factor_cols = [c for c in ["mktrf", "smb", "hml", "rf", "umd", "rmw", "cma"] if c in fac.columns]
        for c in factor_cols:
            fac[c] = pd.to_numeric(fac[c], errors="coerce")
        # WRDS/Ken French factors are often percent returns. Convert to decimals if scale indicates percentages.
        for c in factor_cols:
            med_abs = fac[c].abs().median(skipna=True)
            if pd.notna(med_abs) and med_abs > 0.5:
                fac[c] = fac[c] / 100.0
        fac = fac[["month"] + factor_cols].dropna(subset=["mktrf", "smb", "hml"], how="any")
        info.update({"status": "ok", "table": f"{schema}.{table}", "columns": factor_cols, "rows": int(len(fac))})
        return fac, info
    except Exception as exc:
        info.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(limit=8)})
        return pd.DataFrame(), info
    finally:
        try:
            db.close()
        except Exception:
            pass


def factor_attribution(port: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    if factors.empty:
        return pd.DataFrame()
    import statsmodels.api as sm

    factor_cols = [c for c in ["mktrf", "smb", "hml", "umd", "rmw", "cma"] if c in factors.columns]
    if len(factor_cols) < 3:
        return pd.DataFrame()
    rows = []
    merged = port.merge(factors, on="month", how="inner")
    for (model, target), g in merged.groupby(["model", "target"], observed=True, sort=True):
        # Standard factor attribution is cleanest for the 1-month portfolio. Still compute 3m separately for diagnostics.
        ann = TARGET_ANNUALIZER.get(target, 12.0)
        for ret_col in ["gross_dollar_neutral_return", "net_return_25bps"]:
            use = g[[ret_col] + factor_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(use) < 36:
                continue
            y = use[ret_col].astype(float)
            X = sm.add_constant(use[factor_cols].astype(float), has_constant="add")
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            row = {
                "model": model,
                "target": target,
                "return_series": ret_col,
                "n_months": int(len(use)),
                "alpha_monthly": float(fit.params.get("const", np.nan)),
                "alpha_annualized": float(ann * fit.params.get("const", np.nan)),
                "alpha_t_hac6": float(fit.tvalues.get("const", np.nan)),
                "r2": float(fit.rsquared),
            }
            for fc in factor_cols:
                row[f"beta_{fc}"] = float(fit.params.get(fc, np.nan))
                row[f"t_{fc}"] = float(fit.tvalues.get(fc, np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def make_figures(root: Path, run_id: str, port: pd.DataFrame, summary: pd.DataFrame, costs: pd.DataFrame, factor: pd.DataFrame) -> dict[str, str]:
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
            "font.size": 10,
        }
    )

    figure_paths: dict[str, str] = {}

    p1 = port.loc[port["target"].eq("fwd_ret_1m")].copy()
    if not p1.empty:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        for model in TOP_MODEL_ORDER:
            g = p1.loc[p1["model"].eq(model)].sort_values("month")
            if g.empty:
                continue
            wealth = (1.0 + g["net_return_25bps"].fillna(0.0)).cumprod() - 1.0
            ax.plot(pd.to_datetime(g["month"]), wealth, label=model)
        ax.axhline(0, linewidth=1)
        ax.set_title("Step 009: cumulative OOS dollar-neutral return after 25 bps trading cost, 1M target")
        ax.set_xlabel("Month")
        ax.set_ylabel("Cumulative return")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        path = fig_dir / f"009_cumulative_net25_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figure_paths["cumulative_net25_1m"] = str(path)

        fig, ax = plt.subplots(figsize=(12, 6.5))
        for model in TOP_MODEL_ORDER:
            g = p1.loc[p1["model"].eq(model)].sort_values("month")
            if g.empty:
                continue
            wealth = (1.0 + g["net_return_25bps"].fillna(0.0)).cumprod()
            dd = wealth / wealth.cummax() - 1.0
            ax.plot(pd.to_datetime(g["month"]), dd, label=model)
        ax.axhline(0, linewidth=1)
        ax.set_title("Step 009: OOS drawdowns after 25 bps trading cost, 1M target")
        ax.set_xlabel("Month")
        ax.set_ylabel("Drawdown")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        path = fig_dir / f"009_drawdown_net25_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figure_paths["drawdown_net25_1m"] = str(path)

    s1 = summary.loc[summary["target"].eq("fwd_ret_1m") & summary["return_series"].eq("net_return_25bps")].copy()
    if not s1.empty:
        s1 = s1.sort_values("annualized_mean_return")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(s1["model"], s1["annualized_mean_return"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Step 009: annualized net OOS return after 25 bps cost, 1M target")
        ax.set_xlabel("Annualized mean return")
        fig.tight_layout()
        path = fig_dir / f"009_model_net25_bar_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figure_paths["model_net25_bar_1m"] = str(path)

    c1 = costs.loc[costs["target"].eq("fwd_ret_1m")].copy()
    if not c1.empty:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        for model in TOP_MODEL_ORDER:
            g = c1.loc[c1["model"].eq(model)].sort_values("cost_bps")
            if g.empty:
                continue
            ax.plot(g["cost_bps"], g["annualized_mean_return"], marker="o", label=model)
        ax.axhline(0, linewidth=1)
        ax.set_title("Step 009: cost sensitivity of OOS portfolios, 1M target")
        ax.set_xlabel("One-way trading cost assumption, bps")
        ax.set_ylabel("Annualized mean return")
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        path = fig_dir / f"009_cost_sensitivity_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        figure_paths["cost_sensitivity_1m"] = str(path)

    html = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\">
<title>Step 009 portfolio costs and factor attribution</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; max-width: 1280px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px 0; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
th {{ background: #f4f4f4; }}
td:first-child, th:first-child {{ text-align: left; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Step 009 portfolio costs and factor attribution</h1>
<p>Run ID: <code>{run_id}</code></p>
<h2>Performance summary</h2>
{summary.to_html(index=False)}
<h2>Cost sensitivity</h2>
{costs.to_html(index=False)}
<h2>Factor attribution</h2>
{factor.to_html(index=False) if not factor.empty else '<p>Factor attribution unavailable or skipped.</p>'}
</body>
</html>
"""
    html_path = html_dir / f"009_portfolio_costs_dashboard_{run_id}.html"
    html_path.write_text(html, encoding="utf-8")
    figure_paths["interactive_dashboard"] = str(html_path)
    return figure_paths


def bundle_outputs(root: Path, run_id: str, log_dir: Path) -> Path:
    bundle = root / "artifacts" / "logs" / f"009_portfolio_costs_{run_id}_logs_and_results.zip"
    if bundle.exists():
        bundle.unlink()
    patterns = [
        f"artifacts/logs/009_portfolio_costs_manifest_{run_id}*.json",
        f"artifacts/tables/009_*_{run_id}.csv",
        f"artifacts/figures_static/009_*_{run_id}.png",
        f"artifacts/figures_interactive/009_*_{run_id}.html",
        "docs/009_portfolio_costs_and_factor_attribution.md",
        "scripts/009_portfolio_costs_and_factor_attribution.py",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if log_dir.exists():
            for path in log_dir.rglob("*"):
                if path.is_file():
                    z.write(path, f"logs/{log_dir.name}/{path.relative_to(log_dir)}")
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    z.write(path, str(path.relative_to(root)))
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-jobs", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = args.run_id
    log_dir = root / "logs" / f"009_portfolio_costs_{run_id}"
    table_dir = root / "artifacts" / "tables"
    artifact_log_dir = root / "artifacts" / "logs"
    doc_dir = root / "docs"
    for directory in [table_dir, artifact_log_dir, doc_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    if not host.lower().startswith("compute-node"):
        raise RuntimeError(f"Step 009 must run on compute-node only; current host={host}")

    step008_manifest_path = latest_json(root, "artifacts/logs/008_oos_ml_manifest_*.json")
    if step008_manifest_path is None:
        raise FileNotFoundError("No Step 008 OOS ML manifest found under artifacts/logs.")
    step008_manifest = json.loads(step008_manifest_path.read_text(encoding="utf-8"))
    if step008_manifest.get("status") != "ok":
        raise RuntimeError(f"Latest Step 008 manifest status is not ok: {step008_manifest_path}")
    pred_path = resolve_path(step008_manifest.get("local_prediction_path"))
    if pred_path is None or not pred_path.exists():
        raise FileNotFoundError(f"Could not resolve Step 008 prediction parquet: {step008_manifest.get('local_prediction_path')}")

    print(f"[009] Step 008 manifest: {step008_manifest_path}")
    print(f"[009] Step 008 predictions: {pred_path}")

    preds = load_predictions(pred_path, args.n_jobs)
    port, holdings = build_portfolios(preds)
    summary = performance_summary(port)
    costs = cost_sensitivity(port)

    min_month = str(port["month"].min())
    max_month = str(port["month"].max())
    factors, factor_info = try_fetch_ff_factors(root, min_month, max_month)
    print(f"[009] FF factor fetch info: {json.dumps(factor_info, sort_keys=True, default=str)}")
    factor = factor_attribution(port, factors)

    monthly_path = table_dir / f"009_portfolio_monthly_returns_{run_id}.csv"
    turnover_path = table_dir / f"009_portfolio_turnover_{run_id}.csv"
    summary_path = table_dir / f"009_portfolio_performance_summary_{run_id}.csv"
    costs_path = table_dir / f"009_cost_sensitivity_{run_id}.csv"
    factor_path = table_dir / f"009_factor_attribution_{run_id}.csv"
    holdings_counts_path = table_dir / f"009_portfolio_holding_counts_{run_id}.csv"

    port.to_csv(monthly_path, index=False)
    port[["model", "target", "month", "turnover_abs_change", "turnover_one_way", "gross_exposure", "net_exposure", "n_names"]].to_csv(turnover_path, index=False)
    summary.to_csv(summary_path, index=False)
    costs.to_csv(costs_path, index=False)
    factor.to_csv(factor_path, index=False)
    (
        holdings.groupby(["model", "target", "month", "side"], observed=True)["permno"]
        .count()
        .reset_index(name="n_names")
        .to_csv(holdings_counts_path, index=False)
    )

    figure_paths = make_figures(root, run_id, port, summary, costs, factor)

    # Pick a clean headline model using 25 bps net 1M return first, then fallback to all rows.
    headline = summary.loc[
        summary["target"].eq("fwd_ret_1m") & summary["return_series"].eq("net_return_25bps")
    ].copy()
    if headline.empty:
        headline = summary.copy()
    headline = headline.sort_values(["sharpe_approx", "annualized_mean_return"], ascending=False)
    top = headline.head(1).to_dict("records")[0] if not headline.empty else {}

    manifest = {
        "run_id": run_id,
        "status": "ok",
        "created_utc": now_utc(),
        "host": host,
        "n_jobs": args.n_jobs,
        "step008_manifest": str(step008_manifest_path),
        "step008_prediction_path_local_only": str(pred_path),
        "prediction_rows_loaded": int(len(preds)),
        "portfolio_month_rows": int(len(port)),
        "models": sorted(port["model"].dropna().unique().tolist()),
        "targets": sorted(port["target"].dropna().unique().tolist()),
        "cost_bps_grid": COST_BPS,
        "factor_fetch_info": factor_info,
        "headline_net25_1m_or_fallback": top,
        "tables": {
            "monthly_returns": str(monthly_path),
            "turnover": str(turnover_path),
            "performance_summary": str(summary_path),
            "cost_sensitivity": str(costs_path),
            "factor_attribution": str(factor_path),
            "holding_counts": str(holdings_counts_path),
        },
        "figures": figure_paths,
        "notes": [
            "Portfolio weights are dollar-neutral with +0.5 long top decile and -0.5 short bottom decile.",
            "Transaction costs are charged against absolute notional changed; this is intentionally conservative.",
            "Name-month holdings and Step 008 predictions remain local-only and are not bundled.",
        ],
    }

    manifest_path = artifact_log_dir / f"009_portfolio_costs_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    doc_path = doc_dir / "009_portfolio_costs_and_factor_attribution.md"
    doc_path.write_text(
        f"""# Step 009 portfolio costs and factor attribution

- Run ID: `{run_id}`
- Status: `ok`
- Host: `{host}`
- Input Step 008 manifest: `{step008_manifest_path.name}`
- Prediction rows loaded: `{len(preds):,}`
- Portfolio month rows: `{len(port):,}`
- Factor status: `{factor_info.get('status')}`
- Factor table: `{factor_info.get('table')}`

This step converts Step 008 OOS predictions into dollar-neutral top-minus-bottom decile portfolios. It reports gross and transaction-cost-adjusted returns, turnover, drawdowns, cost sensitivity, and optional Fama-French factor attribution. Vendor-derived name-month predictions and holdings remain local-only and gitignored.

## Headline 25 bps net 1M result or fallback

```text
{pd.DataFrame([top]).to_string(index=False) if top else 'No headline row available.'}
```
""",
        encoding="utf-8",
    )

    bundle = bundle_outputs(root, run_id, log_dir)
    print(f"[009] wrote manifest: {manifest_path}")
    print(f"[009] wrote bundle:   {bundle}")
    print("[009] headline:")
    print(pd.DataFrame([top]).to_string(index=False) if top else "No headline row available.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        root = Path(".")
        run_id = os.environ.get("ONF_RUN_ID", "unknown")
        fail_path = root / "artifacts" / "logs" / f"009_portfolio_costs_manifest_{run_id}_FAILED.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "created_utc": now_utc(),
                    "host": socket.gethostname(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"[009] FAILED. Wrote failure manifest: {fail_path}")
        traceback.print_exc()
        raise
