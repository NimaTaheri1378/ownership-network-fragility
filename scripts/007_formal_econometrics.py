from __future__ import annotations

from dataclasses import dataclass
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


DEFAULT_SIGNAL_FEATURES = [
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

DEFAULT_CONTROLS = ["mktcap_proxy", "vol"]
TARGETS = ["fwd_ret_1m", "fwd_ret_3m"]


@dataclass(frozen=True)
class Spec:
    name: str
    features: list[str]
    controls: list[str]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def qpath(path: Path) -> str:
    return str(path).replace("'", "''")


def resolve_path(text: str | None) -> Path | None:
    if not text:
        return None
    p = Path(text)
    if p.exists():
        return p

    s = str(p)
    swaps = [
        ("~/", "~/"),
        ("~/", "~/"),
    ]
    for old, new in swaps:
        if s.startswith(old):
            alt = Path(new + s[len(old):])
            if alt.exists():
                return alt

    return p


def latest_manifest(root: Path, pattern: str, required_status: str = "ok") -> Path:
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [p for p in candidates if "FAILED" not in p.name]
    good: list[Path] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == required_status:
                good.append(path)
        except Exception:
            continue
    if not good:
        raise FileNotFoundError(f"No {required_status!r} manifest found for pattern: {pattern}")
    return good[0]


def find_step006_manifest(root: Path) -> Path:
    patterns = [
        "artifacts/logs/006_baseline_signal_manifest_*.json",
        "artifacts/logs/006_baseline_signals_manifest_*.json",
        "artifacts/logs/*006*manifest*.json",
    ]
    for pattern in patterns:
        try:
            return latest_manifest(root, pattern, required_status="ok")
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No successful Step 006 manifest found under artifacts/logs.")


def panel_path_from_manifest(root: Path, step006: dict) -> Path:
    candidates = [
        step006.get("local_panel_path"),
        step006.get("input_panel_path_local_only"),
        step006.get("source_panel_path"),
    ]

    source_manifest = resolve_path(step006.get("source_manifest"))
    if source_manifest and source_manifest.exists():
        try:
            step005 = json.loads(source_manifest.read_text(encoding="utf-8"))
            candidates.extend(
                [
                    step005.get("network_info", {}).get("final_panel_path"),
                    step005.get("panel_info", {}).get("final_panel_path"),
                    step005.get("final_panel_path"),
                ]
            )
        except Exception:
            pass

    candidates.extend(
        [
            str(root / "data/processed/005_full_panel/20260529T185448Z/stock_month_network_panel.parquet"),
            str(root / "data/processed/005_full_panel/*/stock_month_network_panel.parquet"),
        ]
    )

    for item in candidates:
        if not item:
            continue
        if "*" in item:
            matches = sorted(root.glob(str(Path(item).relative_to(root)) if str(item).startswith(str(root)) else item))
            for m in reversed(matches):
                if m.exists():
                    return m
            continue

        p = resolve_path(item)
        if p and p.exists():
            return p

    raise FileNotFoundError("Could not resolve the Step 005 full panel path.")


def newey_west_for_mean(x: pd.Series, max_lag: int = 6) -> dict[str, float | int | None]:
    s = pd.Series(x, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(s))
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "t": None, "std": None}
    mean = float(s.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "se": None, "t": None, "std": None}

    d = (s - mean).to_numpy(dtype=float)
    lag = min(max_lag, n - 1)
    gamma0 = float(np.dot(d, d) / n)
    var = gamma0
    for L in range(1, lag + 1):
        weight = 1.0 - L / (lag + 1.0)
        gamma = float(np.dot(d[L:], d[:-L]) / n)
        var += 2.0 * weight * gamma

    se = math.sqrt(max(var, 0.0) / n)
    t = mean / se if se > 0 else None
    return {"n": n, "mean": mean, "se": se, "t": t, "std": float(s.std(ddof=1))}


def winsorize_series(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    qlo, qhi = s.quantile([lo, hi])
    return s.clip(qlo, qhi)


def within_month_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        g = out.groupby("month", observed=True)[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        out[f"z_{col}"] = (out[col] - mean) / std
    return out


def add_interactions(df: pd.DataFrame, possible: list[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    created: list[str] = []
    out = df.copy()
    for a, b in possible:
        za, zb = f"z_{a}", f"z_{b}"
        if za in out.columns and zb in out.columns:
            name = f"z_{a}_x_{b}"
            out[name] = out[za] * out[zb]
            created.append(name)
    return out, created


def ols_one_month(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.dot(resid, resid))
    y_centered = y - y.mean()
    tss = float(np.dot(y_centered, y_centered))
    r2 = 1.0 - sse / tss if tss > 0 else np.nan
    n, k = X.shape
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1) if np.isfinite(r2) else np.nan
    return beta, adj


def run_fama_macbeth(
    df: pd.DataFrame,
    target: str,
    spec: Spec,
    nw_lag: int,
    min_obs_multiplier: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regressors = [f"z_{x}" for x in spec.features if f"z_{x}" in df.columns]
    regressors += [f"z_{x}" for x in spec.controls if f"z_{x}" in df.columns]
    regressors = list(dict.fromkeys(regressors))

    if not regressors:
        return pd.DataFrame(), pd.DataFrame()

    month_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []

    min_obs = max(50, (len(regressors) + 1) * min_obs_multiplier)

    for month, g in df[["month", target] + regressors].dropna().groupby("month", sort=True, observed=True):
        if len(g) < min_obs:
            continue
        y = g[target].to_numpy(dtype=float)
        X0 = g[regressors].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), X0])

        try:
            beta, adj = ols_one_month(y, X)
        except np.linalg.LinAlgError:
            continue

        month_record = {
            "spec": spec.name,
            "target": target,
            "month": month,
            "n_obs": int(len(g)),
            "adj_r2": float(adj) if np.isfinite(adj) else np.nan,
        }
        month_rows.append(month_record)

        for name, value in zip(["intercept"] + regressors, beta):
            coef_rows.append(
                {
                    "spec": spec.name,
                    "target": target,
                    "month": month,
                    "variable": name.replace("z_", "", 1),
                    "coef": float(value),
                }
            )

    monthly = pd.DataFrame(month_rows)
    coefs = pd.DataFrame(coef_rows)

    if coefs.empty:
        return monthly, pd.DataFrame()

    summary_rows = []
    for (variable), g in coefs.groupby("variable", observed=True):
        nw = newey_west_for_mean(g["coef"], max_lag=nw_lag)
        summary_rows.append(
            {
                "spec": spec.name,
                "target": target,
                "variable": variable,
                "months": nw["n"],
                "mean_coef": nw["mean"],
                "nw_se": nw["se"],
                "nw_t": nw["t"],
                "coef_std": nw["std"],
                "positive_month_share": float((g["coef"] > 0).mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return monthly, summary


def build_specs(signal_features: list[str], controls: list[str], interactions: list[str]) -> list[Spec]:
    direct = [x for x in ["owner_count", "owner_hhi", "top_owner_share", "fragility_proxy", "avg_manager_breadth", "stock_sell_pressure"] if x in signal_features]
    network = [x for x in ["network_degree", "network_weighted_degree", "network_peer_sell_pressure"] if x in signal_features]
    mechanism = [x for x in ["fragility_proxy", "stock_sell_pressure", "network_peer_sell_pressure"] if x in signal_features]

    specs: list[Spec] = []

    for f in signal_features:
        specs.append(Spec(name=f"univariate__{f}", features=[f], controls=[]))

    if direct:
        specs.append(Spec(name="direct_ownership_core", features=direct, controls=controls))
    if network:
        specs.append(Spec(name="network_core", features=network, controls=controls))
    if mechanism:
        specs.append(Spec(name="sell_pressure_mechanism", features=mechanism, controls=controls))

    full_features = direct + network
    if full_features:
        specs.append(Spec(name="full_core_no_controls", features=full_features, controls=[]))
        specs.append(Spec(name="full_core_with_controls", features=full_features, controls=controls))

    if interactions:
        # Interaction names are already z-prefixed in the dataframe; store pseudo-feature names without z_.
        pseudo = [x.replace("z_", "", 1) for x in interactions]
        specs.append(Spec(name="interaction_mechanism", features=mechanism + pseudo, controls=controls))

    # Deduplicate while keeping order.
    seen = set()
    unique: list[Spec] = []
    for spec in specs:
        key = (spec.name, tuple(spec.features), tuple(spec.controls))
        if key not in seen:
            unique.append(spec)
            seen.add(key)
    return unique


def make_figures(
    root: Path,
    run_id: str,
    coef_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    corr: pd.DataFrame,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig_dir = root / "artifacts/figures_static"
    html_dir = root / "artifacts/figures_interactive"
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
            "axes.titlesize": 12,
        }
    )

    outputs: dict[str, str] = {}

    core = coef_summary[
        (coef_summary["spec"] == "full_core_with_controls")
        & (coef_summary["target"] == "fwd_ret_1m")
        & (coef_summary["variable"] != "intercept")
    ].copy()
    if not core.empty:
        core["abs_t"] = core["nw_t"].abs()
        core = core.sort_values("abs_t").tail(16)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(core["variable"], core["nw_t"])
        ax.axvline(0, linewidth=1)
        ax.set_title("Step 007: Fama-MacBeth Newey-West t-statistics, full core, 1M")
        ax.set_xlabel("Newey-West t-statistic")
        fig.tight_layout()
        path = fig_dir / f"007_fmb_tstats_full_core_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["fmb_tstats_full_core_1m"] = str(path)

    if not model_summary.empty:
        p = model_summary[model_summary["target"] == "fwd_ret_1m"].copy()
        p = p.sort_values("avg_adj_r2").tail(15)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(p["spec"], p["avg_adj_r2"])
        ax.set_title("Step 007: average monthly adjusted R², 1M target")
        ax.set_xlabel("Average adjusted R²")
        fig.tight_layout()
        path = fig_dir / f"007_model_adj_r2_1m_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["model_adj_r2_1m"] = str(path)

    if not corr.empty:
        labels = list(corr.columns)
        fig, ax = plt.subplots(figsize=(9.5, 8))
        im = ax.imshow(corr.to_numpy(dtype=float), aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title("Step 007: Spearman correlation among regressors and targets")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = fig_dir / f"007_regressor_target_correlation_{run_id}.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["regressor_target_correlation"] = str(path)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Step 007 Formal Econometrics</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; max-width: 1250px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; font-size: 12px; }}
th {{ background: #f4f4f4; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Step 007 Formal Econometrics</h1>
<p>Run ID: <code>{run_id}</code></p>
<h2>Model summary</h2>
{model_summary.to_html(index=False) if not model_summary.empty else "<p>No model summary.</p>"}
<h2>Coefficient summary</h2>
{coef_summary.to_html(index=False) if not coef_summary.empty else "<p>No coefficient summary.</p>"}
</body>
</html>
"""
    html_path = html_dir / f"007_formal_econometrics_dashboard_{run_id}.html"
    html_path.write_text(html, encoding="utf-8")
    outputs["interactive_dashboard"] = str(html_path)

    return outputs


def bundle_outputs(root: Path, run_id: str, log_dir: Path) -> Path:
    bundle = root / "artifacts/logs" / f"007_formal_econometrics_{run_id}_logs_and_results.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)

    patterns = [
        f"artifacts/logs/007_formal_econometrics_manifest_{run_id}.json",
        f"artifacts/tables/007_*_{run_id}.csv",
        f"artifacts/figures_static/007_*_{run_id}.png",
        f"artifacts/figures_interactive/007_*_{run_id}.html",
        "docs/007_formal_econometrics_results.md",
        "scripts/007_formal_econometrics.py",
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
    parser.add_argument("--nw-lag", type=int, default=6)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = args.run_id
    log_dir = root / "logs" / f"007_formal_econometrics_{run_id}"

    table_dir = root / "artifacts/tables"
    log_art_dir = root / "artifacts/logs"
    docs_dir = root / "docs"
    out_dir = root / "data/processed/007_formal_econometrics" / run_id
    for path in [table_dir, log_art_dir, docs_dir, out_dir]:
        path.mkdir(parents=True, exist_ok=True)

    step006_path = find_step006_manifest(root)
    step006 = json.loads(step006_path.read_text(encoding="utf-8"))
    panel = panel_path_from_manifest(root, step006)

    print(f"[007] Step 006 manifest: {step006_path}")
    print(f"[007] Step 005 panel:     {panel}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.n_jobs}")

    cols_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{qpath(panel)}')").fetchdf()
    cols = set(cols_df["column_name"].astype(str).tolist())

    manifest_features = [x for x in step006.get("features", []) if isinstance(x, str)]
    signal_features = [x for x in DEFAULT_SIGNAL_FEATURES if x in cols]
    for f in manifest_features:
        if f in DEFAULT_SIGNAL_FEATURES and f in cols and f not in signal_features:
            signal_features.append(f)

    controls = [x for x in DEFAULT_CONTROLS if x in cols]
    targets = [x for x in TARGETS if x in cols]

    missing = {
        "signal_features_missing": [x for x in DEFAULT_SIGNAL_FEATURES if x not in cols],
        "controls_missing": [x for x in DEFAULT_CONTROLS if x not in cols],
        "targets_missing": [x for x in TARGETS if x not in cols],
    }

    if not signal_features:
        raise RuntimeError(f"No Step 007 signal features found. Missing summary: {missing}")
    if not targets:
        raise RuntimeError(f"No Step 007 targets found. Missing summary: {missing}")
    if "month" not in cols or "permno" not in cols:
        raise RuntimeError("Panel must include month and permno.")

    read_cols = ["month", "permno"] + signal_features + controls + targets
    read_cols = list(dict.fromkeys(read_cols))

    print(f"[007] columns selected: {read_cols}")
    df = con.execute(
        f"SELECT {', '.join(read_cols)} FROM read_parquet('{qpath(panel)}')"
    ).fetchdf()
    con.close()

    print(f"[007] loaded panel rows: {len(df):,}")

    df["month"] = df["month"].astype(str)
    for col in signal_features + controls + targets:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Light winsorization by month reduces influence of bad return/mcap outliers in cross-sectional OLS.
        df[col] = df.groupby("month", observed=True)[col].transform(
            lambda s: winsorize_series(s) if s.notna().sum() >= 50 else s
        )

    df = within_month_zscore(df, signal_features + controls)

    df, interactions = add_interactions(
        df,
        [
            ("fragility_proxy", "stock_sell_pressure"),
            ("fragility_proxy", "network_peer_sell_pressure"),
            ("owner_hhi", "stock_sell_pressure"),
            ("network_weighted_degree", "network_peer_sell_pressure"),
        ],
    )

    specs = build_specs(signal_features, controls, interactions)
    print(f"[007] signal features: {signal_features}")
    print(f"[007] controls:        {controls}")
    print(f"[007] targets:         {targets}")
    print(f"[007] interactions:    {interactions}")
    print(f"[007] n_specs:         {len(specs)}")

    monthly_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []

    for target in targets:
        for spec in specs:
            print(f"[007] Fama-MacBeth target={target} spec={spec.name}")
            monthly, summary = run_fama_macbeth(df, target, spec, nw_lag=args.nw_lag)
            if not monthly.empty:
                monthly_parts.append(monthly)
            if not summary.empty:
                summary_parts.append(summary)

    monthly_df = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    coef_summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    if monthly_df.empty or coef_summary.empty:
        raise RuntimeError("Fama-MacBeth produced empty outputs.")

    model_summary = (
        monthly_df.groupby(["spec", "target"], observed=True)
        .agg(
            months=("month", "nunique"),
            avg_n_obs=("n_obs", "mean"),
            min_n_obs=("n_obs", "min"),
            avg_adj_r2=("adj_r2", "mean"),
            median_adj_r2=("adj_r2", "median"),
        )
        .reset_index()
    )

    corr_cols = [f"z_{x}" for x in signal_features + controls if f"z_{x}" in df.columns] + targets
    corr = df[corr_cols].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    corr.index = [x.replace("z_", "", 1) for x in corr.index]
    corr.columns = [x.replace("z_", "", 1) for x in corr.columns]

    paths = {
        "monthly_coefficients": table_dir / f"007_fmb_monthly_coefficients_{run_id}.csv",
        "coef_summary": table_dir / f"007_fmb_coefficient_summary_{run_id}.csv",
        "model_summary": table_dir / f"007_fmb_model_summary_{run_id}.csv",
        "correlation": table_dir / f"007_regressor_target_correlation_{run_id}.csv",
    }

    monthly_df.to_csv(paths["monthly_coefficients"], index=False)
    coef_summary.to_csv(paths["coef_summary"], index=False)
    model_summary.to_csv(paths["model_summary"], index=False)
    corr.to_csv(paths["correlation"])

    figures = make_figures(root, run_id, coef_summary, model_summary, corr)

    # Pick headline results without overfitting the narrative.
    signal_only = coef_summary[
        (~coef_summary["variable"].eq("intercept"))
        & (~coef_summary["variable"].isin(controls))
        & (coef_summary["target"].eq("fwd_ret_1m"))
    ].copy()
    signal_only["abs_t"] = signal_only["nw_t"].abs()
    top_rows = signal_only.sort_values("abs_t", ascending=False).head(10)

    headline = top_rows[
        ["spec", "target", "variable", "mean_coef", "nw_t", "months", "positive_month_share"]
    ].to_dict("records")

    manifest = {
        "run_id": run_id,
        "status": "ok",
        "created_utc": now_utc(),
        "host": socket.gethostname(),
        "project_root": str(root),
        "source_step006_manifest": str(step006_path),
        "source_step005_panel_local_only": str(panel),
        "n_jobs": args.n_jobs,
        "nw_lag": args.nw_lag,
        "rows_loaded": int(len(df)),
        "months": int(df["month"].nunique()),
        "stocks": int(df["permno"].nunique()),
        "targets": targets,
        "signal_features": signal_features,
        "controls": controls,
        "interactions": interactions,
        "missing": missing,
        "n_specs": len(specs),
        "coef_summary_rows": int(len(coef_summary)),
        "model_summary_rows": int(len(model_summary)),
        "headline_top_abs_t_fwd_ret_1m": headline,
        "tables": {k: str(v) for k, v in paths.items()},
        "figures": figures,
    }

    manifest_path = log_art_dir / f"007_formal_econometrics_manifest_{run_id}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    report = docs_dir / "007_formal_econometrics_results.md"
    top_md = top_rows[
        ["spec", "target", "variable", "mean_coef", "nw_t", "months", "positive_month_share"]
    ].to_markdown(index=False) if not top_rows.empty else "No headline coefficient rows."

    report.write_text(
        f"""# Step 007 formal econometrics results

- Run ID: `{run_id}`
- Source Step 006 manifest: `{step006_path.name}`
- Panel rows loaded: `{len(df):,}`
- Months: `{df["month"].nunique()}`
- Stocks: `{df["permno"].nunique()}`
- Targets: `{", ".join(targets)}`
- Signal features: `{", ".join(signal_features)}`
- Controls: `{", ".join(controls) if controls else "none"}`
- Newey-West lag: `{args.nw_lag}`

## Top absolute Newey-West t-statistics for 1-month forward return

{top_md}

This step estimates monthly cross-sectional Fama-MacBeth regressions on the full filing-date-clean ownership-network panel. Outputs are aggregate coefficient, model, correlation, and figure artifacts only; vendor-derived panels remain local and gitignored.
""",
        encoding="utf-8",
    )

    bundle = bundle_outputs(root, run_id, log_dir)

    print("[007] top headline rows:")
    if top_rows.empty:
        print("  none")
    else:
        print(top_rows[["spec", "target", "variable", "mean_coef", "nw_t", "months"]].to_string(index=False))

    print(f"[007] wrote manifest: {manifest_path}")
    print(f"[007] wrote bundle:   {bundle}")
    print("[007] status: ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        root = Path(os.environ.get("ROOT", "."))
        run_id = os.environ.get("RUN_ID", "unknown")
        fail_path = root / "artifacts/logs" / f"007_formal_econometrics_manifest_{run_id}_FAILED.json"
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
        print(f"[007] FAILED. Wrote failure manifest: {fail_path}")
        traceback.print_exc()
        raise
