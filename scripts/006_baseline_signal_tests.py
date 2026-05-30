from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import sys
import traceback
from typing import Any

import duckdb
import numpy as np
import pandas as pd

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TARGETS = ["fwd_ret_1m", "fwd_ret_3m"]
CORE_FEATURES = [
    "owner_count",
    "owner_hhi",
    "top_owner_share",
    "fragility_proxy",
    "avg_manager_breadth",
    "stock_sell_pressure",
    "network_degree",
    "network_weighted_degree",
    "network_peer_sell_pressure",
]
CONTROL_CONTEXT_FEATURES = ["mktcap_proxy", "vol"]
NEGATIVE_HIGH_MINUS_LOW_EXPECTED = {
    "owner_hhi",
    "top_owner_share",
    "fragility_proxy",
    "stock_sell_pressure",
    "network_degree",
    "network_weighted_degree",
    "network_peer_sell_pressure",
}
MIN_ROWS = 100_000
MIN_MONTHS = 120


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def qident(x: str) -> str:
    if not IDENT_RE.match(x or ""):
        raise ValueError(f"Unsafe SQL identifier: {x!r}")
    return '"' + x.replace('"', '""') + '"'


def qpath(path: str | Path) -> str:
    return str(path).replace("'", "''")


def latest_file(directory: Path, pattern: str) -> Path:
    files = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not files:
        raise SystemExit(f"No successful file found under {directory} matching {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_local_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    text = str(path)
    replacements = [
        ("~/", "~/"),
        ("~/", "~/"),
    ]
    for old, new in replacements:
        if text.startswith(old):
            alt = Path(new + text[len(old):])
            if alt.exists():
                return alt
    if not path.is_absolute():
        alt = root / path
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Could not resolve local path: {value}")


def safe_float(x: Any) -> float | None:
    try:
        if x is None or pd.isna(x):
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def t_stat(series: pd.Series) -> float | None:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return None
    sd = float(x.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return None
    return float(x.mean() / (sd / np.sqrt(len(x))))


def describe_panel(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    base = con.execute(
        """
        SELECT
          COUNT(*) AS rows,
          COUNT(DISTINCT month) AS months,
          COUNT(DISTINCT permno) AS stocks,
          MIN(month) AS min_month,
          MAX(month) AS max_month,
          AVG(CASE WHEN fwd_ret_1m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS fwd_ret_1m_coverage,
          AVG(CASE WHEN fwd_ret_3m IS NOT NULL THEN 1.0 ELSE 0.0 END) AS fwd_ret_3m_coverage
        FROM panel
        """
    ).fetchdf().iloc[0].to_dict()
    return {k: (int(v) if k in {"rows", "months", "stocks"} else safe_float(v) if isinstance(v, float) else v) for k, v in base.items()}


def feature_coverage(con: duckdb.DuckDBPyConnection, features: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total = int(con.execute("SELECT COUNT(*) FROM panel").fetchone()[0])
    for feature in features:
        c = qident(feature)
        row = con.execute(
            f"""
            SELECT
              COUNT({c}) AS non_null_rows,
              COUNT(DISTINCT month) FILTER (WHERE {c} IS NOT NULL) AS months_non_null,
              AVG(CAST({c} AS DOUBLE)) AS mean_value,
              MEDIAN(CAST({c} AS DOUBLE)) AS median_value,
              MIN(CAST({c} AS DOUBLE)) AS min_value,
              MAX(CAST({c} AS DOUBLE)) AS max_value
            FROM panel
            """
        ).fetchdf().iloc[0].to_dict()
        records.append(
            {
                "feature": feature,
                "is_core_signal": feature in CORE_FEATURES,
                "expected_high_minus_low_sign": "negative" if feature in NEGATIVE_HIGH_MINUS_LOW_EXPECTED else "not_prespecified",
                "non_null_rows": int(row["non_null_rows"] or 0),
                "coverage": int(row["non_null_rows"] or 0) / max(total, 1),
                "months_non_null": int(row["months_non_null"] or 0),
                "mean_value": safe_float(row.get("mean_value")),
                "median_value": safe_float(row.get("median_value")),
                "min_value": safe_float(row.get("min_value")),
                "max_value": safe_float(row.get("max_value")),
            }
        )
    return pd.DataFrame(records)


def decile_monthly(con: duckdb.DuckDBPyConnection, feature: str, target: str) -> pd.DataFrame:
    f = qident(feature)
    y = qident(target)
    sql = f"""
        WITH base AS (
            SELECT month,
                   TRY_CAST({f} AS DOUBLE) AS x,
                   TRY_CAST({y} AS DOUBLE) AS y
            FROM panel
            WHERE {f} IS NOT NULL AND {y} IS NOT NULL
        ), filtered AS (
            SELECT * FROM base
            WHERE x IS NOT NULL AND y IS NOT NULL
              AND isfinite(x) AND isfinite(y)
        ), ranked AS (
            SELECT month, y,
                   NTILE(10) OVER (PARTITION BY month ORDER BY x ASC) AS decile
            FROM filtered
        )
        SELECT month, decile, COUNT(*) AS n_obs,
               AVG(y) AS mean_return,
               MEDIAN(y) AS median_return
        FROM ranked
        GROUP BY month, decile
        HAVING COUNT(*) >= 5
        ORDER BY month, decile
    """
    out = con.execute(sql).fetchdf()
    out["feature"] = feature
    out["target"] = target
    return out[["feature", "target", "month", "decile", "n_obs", "mean_return", "median_return"]]


def ic_monthly(con: duckdb.DuckDBPyConnection, feature: str, target: str) -> pd.DataFrame:
    f = qident(feature)
    y = qident(target)
    sql = f"""
        WITH base AS (
            SELECT month,
                   TRY_CAST({f} AS DOUBLE) AS x,
                   TRY_CAST({y} AS DOUBLE) AS y
            FROM panel
            WHERE {f} IS NOT NULL AND {y} IS NOT NULL
        ), filtered AS (
            SELECT * FROM base
            WHERE x IS NOT NULL AND y IS NOT NULL
              AND isfinite(x) AND isfinite(y)
        ), ranked AS (
            SELECT month,
                   PERCENT_RANK() OVER (PARTITION BY month ORDER BY x ASC) AS rx,
                   PERCENT_RANK() OVER (PARTITION BY month ORDER BY y ASC) AS ry
            FROM filtered
        )
        SELECT month, COUNT(*) AS n_obs, CORR(rx, ry) AS rank_ic
        FROM ranked
        GROUP BY month
        HAVING COUNT(*) >= 50
        ORDER BY month
    """
    out = con.execute(sql).fetchdf()
    out["feature"] = feature
    out["target"] = target
    return out[["feature", "target", "month", "n_obs", "rank_ic"]]


def summarize_deciles(deciles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if deciles.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    decile_summary = (
        deciles.groupby(["feature", "target", "decile"], observed=True)
        .agg(
            months=("month", "nunique"),
            avg_n_obs=("n_obs", "mean"),
            ew_mean_return=("mean_return", "mean"),
            ew_median_return=("median_return", "mean"),
        )
        .reset_index()
    )
    piv = deciles.pivot_table(index=["feature", "target", "month"], columns="decile", values="mean_return", aggfunc="mean")
    spreads = piv.reset_index()
    if 1 in spreads.columns and 10 in spreads.columns:
        spreads["high_minus_low"] = spreads[10] - spreads[1]
        spreads["low_minus_high"] = spreads[1] - spreads[10]
    else:
        spreads["high_minus_low"] = np.nan
        spreads["low_minus_high"] = np.nan
    spread_records: list[dict[str, Any]] = []
    for (feature, target), grp in spreads.groupby(["feature", "target"], observed=True):
        x = pd.to_numeric(grp["high_minus_low"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        directional = -x if feature in NEGATIVE_HIGH_MINUS_LOW_EXPECTED else x
        spread_records.append(
            {
                "feature": feature,
                "target": target,
                "expected_high_minus_low_sign": "negative" if feature in NEGATIVE_HIGH_MINUS_LOW_EXPECTED else "not_prespecified",
                "n_months": int(len(x)),
                "mean_high_minus_low": safe_float(x.mean() if len(x) else np.nan),
                "annualized_high_minus_low": safe_float(12.0 * x.mean() if len(x) else np.nan),
                "t_high_minus_low": t_stat(x),
                "positive_month_rate_high_minus_low": safe_float((x > 0).mean() if len(x) else np.nan),
                "mean_directional_spread": safe_float(directional.mean() if len(directional) else np.nan),
                "annualized_directional_spread": safe_float(12.0 * directional.mean() if len(directional) else np.nan),
                "t_directional_spread": t_stat(directional),
            }
        )
    spread_summary = pd.DataFrame(spread_records).sort_values(
        ["target", "t_directional_spread"], key=lambda s: s.abs() if s.name == "t_directional_spread" else s, ascending=[True, False]
    )
    return decile_summary, spreads, spread_summary


def summarize_ic(ic: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (feature, target), grp in ic.groupby(["feature", "target"], observed=True):
        x = pd.to_numeric(grp["rank_ic"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        directional = -x if feature in NEGATIVE_HIGH_MINUS_LOW_EXPECTED else x
        records.append(
            {
                "feature": feature,
                "target": target,
                "expected_high_minus_low_sign": "negative" if feature in NEGATIVE_HIGH_MINUS_LOW_EXPECTED else "not_prespecified",
                "n_months": int(len(x)),
                "mean_rank_ic": safe_float(x.mean() if len(x) else np.nan),
                "t_rank_ic": t_stat(x),
                "positive_month_rate_rank_ic": safe_float((x > 0).mean() if len(x) else np.nan),
                "mean_directional_ic": safe_float(directional.mean() if len(directional) else np.nan),
                "t_directional_ic": t_stat(directional),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(
        ["target", "t_directional_ic"], key=lambda s: s.abs() if s.name == "t_directional_ic" else s, ascending=[True, False]
    )


def feature_correlations(con: duckdb.DuckDBPyConnection, features: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for a in features:
        for b in features:
            ca, cb = qident(a), qident(b)
            val = con.execute(
                f"""
                SELECT CORR(TRY_CAST({ca} AS DOUBLE), TRY_CAST({cb} AS DOUBLE)) AS corr_value
                FROM panel
                WHERE {ca} IS NOT NULL AND {cb} IS NOT NULL
                """
            ).fetchone()[0]
            records.append({"feature_x": a, "feature_y": b, "corr": safe_float(val)})
    return pd.DataFrame(records)


def write_figures(root: Path, run_id: str, spread_summary: pd.DataFrame, ic_summary: pd.DataFrame, spreads: pd.DataFrame, corr: pd.DataFrame) -> dict[str, str]:
    figs = root / "artifacts" / "figures_static"
    htmls = root / "artifacts" / "figures_interactive"
    figs.mkdir(parents=True, exist_ok=True)
    htmls.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt

        for target in TARGETS:
            ss = spread_summary[spread_summary["target"] == target].copy()
            if not ss.empty:
                ss = ss.sort_values("annualized_directional_spread")
                fig, ax = plt.subplots(figsize=(10.5, 5.8))
                ax.barh(ss["feature"], ss["annualized_directional_spread"])
                ax.axvline(0, linewidth=1)
                ax.set_title(f"Step 006 directional high-minus-low spreads: {target}")
                ax.set_xlabel("Annualized directional spread")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                path = figs / f"006_directional_spreads_{target}_{run_id}.png"
                fig.savefig(path, dpi=240)
                plt.close(fig)
                out[f"static_spreads_{target}"] = str(path)

            ii = ic_summary[ic_summary["target"] == target].copy()
            if not ii.empty:
                ii = ii.sort_values("mean_directional_ic")
                fig, ax = plt.subplots(figsize=(10.5, 5.8))
                ax.barh(ii["feature"], ii["mean_directional_ic"])
                ax.axvline(0, linewidth=1)
                ax.set_title(f"Step 006 directional monthly rank IC: {target}")
                ax.set_xlabel("Mean directional rank IC")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                path = figs / f"006_directional_ic_{target}_{run_id}.png"
                fig.savefig(path, dpi=240)
                plt.close(fig)
                out[f"static_ic_{target}"] = str(path)

        if not corr.empty:
            mat = corr.pivot(index="feature_x", columns="feature_y", values="corr").reindex(index=CORE_FEATURES, columns=CORE_FEATURES)
            fig, ax = plt.subplots(figsize=(9.8, 8.5))
            im = ax.imshow(mat.to_numpy(dtype=float), vmin=-1, vmax=1)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_yticks(range(len(mat.index)))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right")
            ax.set_yticklabels(mat.index)
            ax.set_title("Step 006 core feature correlation audit")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            path = figs / f"006_feature_correlation_{run_id}.png"
            fig.savefig(path, dpi=240)
            plt.close(fig)
            out["static_correlation"] = str(path)
    except Exception as exc:
        (figs / f"006_matplotlib_failed_{run_id}.txt").write_text(repr(exc), encoding="utf-8")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "scatter"}, {"type": "table"}]],
            subplot_titles=("Directional spread t-stat", "Directional IC t-stat", "Cumulative selected spreads", "Top rows"),
        )
        top_spread = spread_summary.copy()
        if not top_spread.empty:
            top_spread["abs_t"] = pd.to_numeric(top_spread["t_directional_spread"], errors="coerce").abs()
            top_spread = top_spread.sort_values("abs_t", ascending=False).head(14)
            fig.add_trace(go.Bar(x=top_spread["t_directional_spread"], y=top_spread["feature"] + " / " + top_spread["target"], orientation="h", name="spread t"), row=1, col=1)
        top_ic = ic_summary.copy()
        if not top_ic.empty:
            top_ic["abs_t"] = pd.to_numeric(top_ic["t_directional_ic"], errors="coerce").abs()
            top_ic = top_ic.sort_values("abs_t", ascending=False).head(14)
            fig.add_trace(go.Bar(x=top_ic["t_directional_ic"], y=top_ic["feature"] + " / " + top_ic["target"], orientation="h", name="IC t"), row=1, col=2)
        if not top_spread.empty and not spreads.empty:
            selected = set(zip(top_spread["feature"], top_spread["target"]))
            shown = 0
            for feature, target in selected:
                if shown >= 6:
                    break
                part = spreads[(spreads["feature"] == feature) & (spreads["target"] == target)].copy()
                if part.empty:
                    continue
                x = pd.to_numeric(part["high_minus_low"], errors="coerce")
                if feature in NEGATIVE_HIGH_MINUS_LOW_EXPECTED:
                    x = -x
                part["cum_directional_spread"] = x.fillna(0).cumsum()
                fig.add_trace(go.Scatter(x=part["month"], y=part["cum_directional_spread"], mode="lines", name=f"{feature} {target}"), row=2, col=1)
                shown += 1
        table_df = spread_summary.head(12).copy()
        if table_df.empty:
            table_df = pd.DataFrame({"feature": [], "target": [], "annualized_directional_spread": [], "t_directional_spread": []})
        fig.add_trace(
            go.Table(
                header={"values": ["feature", "target", "ann dir spread", "t"]},
                cells={"values": [table_df.get("feature", []), table_df.get("target", []), table_df.get("annualized_directional_spread", []), table_df.get("t_directional_spread", [])]},
            ),
            row=2,
            col=2,
        )
        fig.update_layout(title="Step 006 full-panel baseline signal tests", template="plotly_white", height=950)
        path = htmls / f"006_baseline_signal_dashboard_{run_id}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        out["interactive_dashboard"] = str(path)
    except Exception as exc:
        path = htmls / f"006_baseline_signal_dashboard_{run_id}.html"
        path.write_text(f"Plotly failed: {exc}\n", encoding="utf-8")
        out["interactive_dashboard"] = str(path)
    return out


def write_markdown(root: Path, manifest: dict[str, Any], spread_summary: pd.DataFrame, ic_summary: pd.DataFrame, coverage: pd.DataFrame) -> str:
    path = root / "docs" / "006_baseline_signal_results.md"
    metrics = manifest.get("metrics", {})
    lines = [
        "# 006 Baseline signal tests",
        "",
        "This step evaluates the full Step 005 filing-date-clean ownership-network panel using aggregate-only baseline tests. Local stock-level Parquet inputs remain under ignored `data/` directories and are not bundled.",
        "",
        "## Panel input",
        "",
        f"- Step 005 manifest: `{manifest.get('source_manifest')}`",
        f"- Local panel path: `{manifest.get('local_panel_path')}`",
        f"- Rows: `{metrics.get('panel_rows')}`",
        f"- Months: `{metrics.get('panel_months')}`",
        f"- Stocks: `{metrics.get('panel_stocks')}`",
        f"- Month range: `{metrics.get('min_month')}` to `{metrics.get('max_month')}`",
        "",
        "## Best directional spreads",
        "",
        "| Feature | Horizon | Annualized directional spread | t-stat | Months |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in spread_summary.head(12).iterrows():
        lines.append(
            f"| {row['feature']} | {row['target']} | {row['annualized_directional_spread']:.6f} | {row['t_directional_spread']:.3f} | {int(row['n_months'])} |"
        )
    lines += [
        "",
        "## Best directional rank IC",
        "",
        "| Feature | Horizon | Mean directional IC | t-stat | Months |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in ic_summary.head(12).iterrows():
        lines.append(
            f"| {row['feature']} | {row['target']} | {row['mean_directional_ic']:.6f} | {row['t_directional_ic']:.3f} | {int(row['n_months'])} |"
        )
    lines += [
        "",
        "## Feature coverage",
        "",
        "| Feature | Coverage | Months non-null | Median |",
        "|---|---:|---:|---:|",
    ]
    for _, row in coverage.iterrows():
        med = row.get("median_value")
        med_text = "" if pd.isna(med) else f"{float(med):.6g}"
        lines.append(f"| {row['feature']} | {row['coverage']:.4f} | {int(row['months_non_null'])} | {med_text} |")
    if manifest.get("problems"):
        lines += ["", "## Validation problems", ""] + [f"- {p}" for p in manifest["problems"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def run(root: Path, run_id: str, n_jobs: int) -> int:
    logs = root / "artifacts" / "logs"
    tables = root / "artifacts" / "tables"
    local_out = root / "data" / "processed" / "006_baseline_signals" / run_id
    for d in [logs, tables, local_out]:
        d.mkdir(parents=True, exist_ok=True)

    source_manifest_path = latest_file(logs, "005_full_panel_manifest_*.json")
    source_manifest = read_json(source_manifest_path)
    panel_path_text = source_manifest.get("network_info", {}).get("final_panel_path")
    if not panel_path_text:
        raise RuntimeError(f"Step 005 manifest lacks network_info.final_panel_path: {source_manifest_path}")
    panel_path = resolve_local_path(root, panel_path_text)
    print(f"[006] source_manifest={source_manifest_path}")
    print(f"[006] local_panel_path={panel_path}")

    con = duckdb.connect(str(local_out / "006_baseline_signals.duckdb"))
    con.execute(f"PRAGMA threads={int(n_jobs)}")
    con.execute("PRAGMA memory_limit='180GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{qpath(panel_path)}')")

    schema = con.execute("DESCRIBE SELECT * FROM panel").fetchdf()
    available_cols = set(schema["column_name"].astype(str))
    features = [f for f in CORE_FEATURES + CONTROL_CONTEXT_FEATURES if f in available_cols]
    targets = [t for t in TARGETS if t in available_cols]
    missing = [x for x in CORE_FEATURES + TARGETS if x not in available_cols]
    if not features or not targets:
        raise RuntimeError(f"No usable features/targets. missing={missing}; available={sorted(available_cols)[:80]}")
    print(f"[006] features={features}")
    print(f"[006] targets={targets}")

    panel_desc = describe_panel(con)
    coverage = feature_coverage(con, features)
    coverage_path = tables / f"006_feature_coverage_{run_id}.csv"
    coverage.to_csv(coverage_path, index=False)

    all_deciles = []
    all_ics = []
    for target in targets:
        for feature in features:
            print(f"[006][signal] feature={feature} target={target}")
            all_deciles.append(decile_monthly(con, feature, target))
            all_ics.append(ic_monthly(con, feature, target))
    deciles = pd.concat(all_deciles, ignore_index=True) if all_deciles else pd.DataFrame()
    ic = pd.concat(all_ics, ignore_index=True) if all_ics else pd.DataFrame()
    decile_summary, spreads, spread_summary = summarize_deciles(deciles)
    ic_summary = summarize_ic(ic)
    corr = feature_correlations(con, [f for f in CORE_FEATURES if f in features])
    con.close()

    table_paths = {
        "feature_coverage_csv": str(coverage_path),
        "decile_monthly_csv": str(tables / f"006_decile_monthly_{run_id}.csv"),
        "decile_summary_csv": str(tables / f"006_decile_summary_{run_id}.csv"),
        "spread_monthly_csv": str(tables / f"006_spread_monthly_{run_id}.csv"),
        "spread_summary_csv": str(tables / f"006_spread_summary_{run_id}.csv"),
        "rank_ic_monthly_csv": str(tables / f"006_rank_ic_monthly_{run_id}.csv"),
        "rank_ic_summary_csv": str(tables / f"006_rank_ic_summary_{run_id}.csv"),
        "feature_correlation_csv": str(tables / f"006_feature_correlation_{run_id}.csv"),
    }
    deciles.to_csv(table_paths["decile_monthly_csv"], index=False)
    decile_summary.to_csv(table_paths["decile_summary_csv"], index=False)
    spreads.to_csv(table_paths["spread_monthly_csv"], index=False)
    spread_summary.to_csv(table_paths["spread_summary_csv"], index=False)
    ic.to_csv(table_paths["rank_ic_monthly_csv"], index=False)
    ic_summary.to_csv(table_paths["rank_ic_summary_csv"], index=False)
    corr.to_csv(table_paths["feature_correlation_csv"], index=False)

    fig_paths = write_figures(root, run_id, spread_summary, ic_summary, spreads, corr)

    metrics: dict[str, Any] = {
        "panel_rows": int(panel_desc.get("rows") or 0),
        "panel_months": int(panel_desc.get("months") or 0),
        "panel_stocks": int(panel_desc.get("stocks") or 0),
        "min_month": panel_desc.get("min_month"),
        "max_month": panel_desc.get("max_month"),
        "fwd_ret_1m_coverage": panel_desc.get("fwd_ret_1m_coverage"),
        "fwd_ret_3m_coverage": panel_desc.get("fwd_ret_3m_coverage"),
        "n_features_tested": len(features),
        "n_targets_tested": len(targets),
        "decile_summary_rows": int(len(decile_summary)),
        "spread_summary_rows": int(len(spread_summary)),
        "rank_ic_summary_rows": int(len(ic_summary)),
    }
    if not spread_summary.empty:
        best = spread_summary.iloc[0]
        metrics.update(
            {
                "top_spread_feature": str(best["feature"]),
                "top_spread_target": str(best["target"]),
                "top_spread_annualized_directional": safe_float(best["annualized_directional_spread"]),
                "top_spread_t_directional": safe_float(best["t_directional_spread"]),
            }
        )
    if not ic_summary.empty:
        best_ic = ic_summary.iloc[0]
        metrics.update(
            {
                "top_ic_feature": str(best_ic["feature"]),
                "top_ic_target": str(best_ic["target"]),
                "top_ic_mean_directional": safe_float(best_ic["mean_directional_ic"]),
                "top_ic_t_directional": safe_float(best_ic["t_directional_ic"]),
            }
        )

    problems: list[str] = []
    if metrics["panel_rows"] < MIN_ROWS:
        problems.append(f"panel_rows below threshold: {metrics['panel_rows']} < {MIN_ROWS}")
    if metrics["panel_months"] < MIN_MONTHS:
        problems.append(f"panel_months below threshold: {metrics['panel_months']} < {MIN_MONTHS}")
    if metrics["n_features_tested"] < 5:
        problems.append(f"too few features tested: {metrics['n_features_tested']}")
    if metrics["spread_summary_rows"] < 10:
        problems.append(f"too few spread summary rows: {metrics['spread_summary_rows']}")
    if metrics["rank_ic_summary_rows"] < 10:
        problems.append(f"too few rank IC summary rows: {metrics['rank_ic_summary_rows']}")

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": utc_now(),
        "host": socket.gethostname(),
        "python": sys.version,
        "project_root": str(root),
        "source_manifest": str(source_manifest_path),
        "source_step005_run_id": source_manifest.get("run_id"),
        "local_panel_path": str(panel_path),
        "local_derived_dir": str(local_out),
        "features": features,
        "targets": targets,
        "missing_expected_columns": missing,
        "metrics": metrics,
        "tables": table_paths,
        "figures": fig_paths,
        "problems": problems,
        "status": "ok" if not problems else "needs_attention",
    }
    markdown = write_markdown(root, manifest, spread_summary, ic_summary, coverage)
    manifest["markdown_report"] = markdown
    manifest_path = logs / f"006_baseline_signal_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[006] manifest={manifest_path}")
    for key, value in metrics.items():
        print(f"[006] {key}={value}")
    print(f"[006] status={manifest['status']}")
    for p in problems:
        print(f"[006][problem] {p}")
    return 0 if not problems else 20


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--n-jobs", type=int, default=32)
    a = p.parse_args()
    try:
        return run(Path(a.project_root).resolve(), a.run_id, a.n_jobs)
    except Exception as exc:
        root = Path(a.project_root).resolve()
        logs = root / "artifacts" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        fail = logs / f"006_baseline_signal_manifest_{a.run_id}_FAILED.json"
        fail.write_text(json.dumps({"run_id": a.run_id, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "created_utc": utc_now()}, indent=2), encoding="utf-8")
        print(f"[006] FAILED wrote {fail}")
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
