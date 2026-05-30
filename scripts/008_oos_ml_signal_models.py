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


RAW_SIGNAL_FEATURES = [
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
CONTROL_FEATURES = ["mktcap_proxy", "vol"]
TARGETS = ["fwd_ret_1m", "fwd_ret_3m"]

INTERACTION_PAIRS = [
    ("fragility_proxy", "stock_sell_pressure"),
    ("fragility_proxy", "network_peer_sell_pressure"),
    ("owner_hhi", "stock_sell_pressure"),
    ("network_weighted_degree", "network_peer_sell_pressure"),
]


@dataclass(frozen=True)
class Split:
    target: str
    test_year: int
    train_months: list[str]
    valid_months: list[str]
    test_months: list[str]


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
    candidates = sorted(
        root.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    candidates = [p for p in candidates if "FAILED" not in p.name]
    return candidates[0] if candidates else None


def find_step005_panel(root: Path) -> tuple[Path, Path]:
    manifest_path = latest_json(root, "artifacts/logs/005_full_panel_manifest_*.json")
    if manifest_path is None:
        raise FileNotFoundError("No successful Step 005 manifest found under artifacts/logs.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = [
        manifest.get("network_info", {}).get("final_panel_path"),
        manifest.get("panel_info", {}).get("final_panel_path"),
        manifest.get("final_panel_path"),
    ]
    for item in candidates:
        path = resolve_path(item)
        if path is not None and path.exists():
            return manifest_path, path

    fallback = sorted(root.glob("data/processed/005_full_panel/*/stock_month_network_panel.parquet"))
    if fallback:
        return manifest_path, fallback[-1]

    raise FileNotFoundError("Could not resolve stock_month_network_panel.parquet from Step 005 manifest.")


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


def month_string(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m")


def cross_sectional_z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("month", observed=True)[col]
    med = g.transform("median")
    std = g.transform("std").replace(0, np.nan)
    z = (df[col] - med) / std
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5).astype("float32")


def winsor_train_y(data: pd.DataFrame, target: str) -> pd.Series:
    g = data.groupby("month", observed=True)[target]
    lo = g.transform(lambda s: s.quantile(0.01))
    hi = g.transform(lambda s: s.quantile(0.99))
    return data[target].clip(lower=lo, upper=hi).astype("float32")


def build_splits(months: list[str], target: str, target_nonnull_by_month: set[str]) -> list[Split]:
    horizon = 3 if target.endswith("3m") else 1
    embargo = horizon
    validation_months = 24
    min_train_months = 96

    out: list[Split] = []
    years = sorted({int(m[:4]) for m in months})
    for year in years:
        test_months = [m for m in months if int(m[:4]) == year and m in target_nonnull_by_month]
        if not test_months:
            continue
        first_idx = months.index(test_months[0])
        valid_end = first_idx - embargo
        valid_start = valid_end - validation_months
        train_end = valid_start - embargo
        if train_end < min_train_months:
            continue
        train_months = months[:train_end]
        valid_months = months[max(0, valid_start):valid_end]
        if len(train_months) < min_train_months or len(valid_months) < 6:
            continue
        out.append(Split(target, year, train_months, valid_months, test_months))
    return out


def spearman_by_month(df: pd.DataFrame, pred_col: str, target: str) -> pd.DataFrame:
    rows = []
    for month, g in df[["month", pred_col, target]].dropna().groupby("month", observed=True, sort=True):
        if len(g) < 20 or g[pred_col].nunique() < 2 or g[target].nunique() < 2:
            continue
        rows.append({"month": month, "rank_ic": g[pred_col].corr(g[target], method="spearman")})
    return pd.DataFrame(rows)


def spread_by_month(df: pd.DataFrame, pred_col: str, target: str) -> pd.DataFrame:
    use = df[["month", pred_col, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if use.empty:
        return pd.DataFrame(columns=["month", "low", "high", "high_minus_low", "n"])
    use["decile"] = np.ceil(
        use.groupby("month", observed=True)[pred_col].rank(method="first", pct=True) * 10
    ).clip(1, 10).astype(int)
    dec = use.groupby(["month", "decile"], observed=True)[target].mean().reset_index()
    wide = dec.pivot(index="month", columns="decile", values=target)
    if 1 not in wide.columns or 10 not in wide.columns:
        return pd.DataFrame(columns=["month", "low", "high", "high_minus_low", "n"])
    counts = use.groupby("month", observed=True).size().rename("n")
    out = pd.DataFrame(
        {
            "month": wide.index.astype(str),
            "low": wide[1].to_numpy(dtype="float64"),
            "high": wide[10].to_numpy(dtype="float64"),
            "high_minus_low": (wide[10] - wide[1]).to_numpy(dtype="float64"),
        }
    )
    out = out.merge(counts.reset_index(), on="month", how="left")
    return out


def fit_predict_ridge(X_train, y_train, X_valid, y_valid, X_test, n_jobs: int):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    alphas = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_alpha = alphas[0]
    best_loss = np.inf
    best_model = None
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=1729)
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        loss = mean_squared_error(y_valid, pred_valid)
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_model = model
    pred = best_model.predict(X_test)
    importance = pd.Series(np.abs(best_model.coef_), index=X_train.columns, dtype="float64")
    return pred, {"alpha": best_alpha, "valid_mse": float(best_loss)}, importance


def fit_predict_lgbm(X_train, y_train, X_valid, y_valid, X_test, n_jobs: int):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=650,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=2.0,
        random_state=1729,
        n_jobs=n_jobs,
        verbosity=-1,
        force_col_wise=True,
    )
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_valid, y_valid)],
        "eval_metric": "l2",
    }
    try:
        model.fit(**fit_kwargs, callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])
    except TypeError:
        model.fit(**fit_kwargs, early_stopping_rounds=60, verbose=False)
    pred = model.predict(X_test, num_iteration=getattr(model, "best_iteration_", None))
    importance = pd.Series(model.feature_importances_, index=X_train.columns, dtype="float64")
    params = {
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "n_estimators": 650,
        "valid_score": float(getattr(model, "best_score_", {}).get("valid_0", {}).get("l2", np.nan)),
    }
    return pred, params, importance


def evaluate_predictions(preds: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    spread_frames = []
    for model_name, g in preds.groupby("model", observed=True, sort=True):
        pred_col = "prediction"
        ic_m = spearman_by_month(g, pred_col, target)
        sp_m = spread_by_month(g, pred_col, target)
        if not ic_m.empty:
            ic_m.insert(0, "model", model_name)
            ic_m.insert(1, "target", target)
            monthly_rows.append(ic_m)
        if not sp_m.empty:
            sp_m.insert(0, "model", model_name)
            sp_m.insert(1, "target", target)
            spread_frames.append(sp_m)

    rank_ic_monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    spread_monthly = pd.concat(spread_frames, ignore_index=True) if spread_frames else pd.DataFrame()

    summary_rows = []
    for model_name, g in preds.groupby("model", observed=True, sort=True):
        ic = rank_ic_monthly.loc[rank_ic_monthly["model"].eq(model_name), "rank_ic"] if not rank_ic_monthly.empty else pd.Series(dtype=float)
        sp = spread_monthly.loc[spread_monthly["model"].eq(model_name), "high_minus_low"] if not spread_monthly.empty else pd.Series(dtype=float)
        ic_mean, ic_se, ic_t = newey_west_t(ic, lag=6)
        sp_mean, sp_se, sp_t = newey_west_t(sp, lag=6)
        annualizer = 4.0 if target.endswith("3m") else 12.0
        y = g[target].astype(float)
        p = g["prediction"].astype(float)
        denom = float(np.sum(np.square(y)))
        r2_zero = 1.0 - float(np.sum(np.square(y - p))) / denom if denom > 0 else np.nan
        hit = float((np.sign(y) == np.sign(p)).mean()) if len(g) else np.nan
        summary_rows.append(
            {
                "model": model_name,
                "target": target,
                "oos_rows": int(len(g)),
                "months": int(g["month"].nunique()),
                "rank_ic_mean": ic_mean,
                "rank_ic_nw_t": ic_t,
                "spread_mean": sp_mean,
                "spread_nw_t": sp_t,
                "annualized_spread_approx": annualizer * sp_mean if pd.notna(sp_mean) else np.nan,
                "r2_vs_zero": r2_zero,
                "sign_hit_rate": hit,
            }
        )
    return pd.DataFrame(summary_rows), rank_ic_monthly, spread_monthly


def make_figures(root: Path, run_id: str, summary: pd.DataFrame, ic_monthly: pd.DataFrame, spread_monthly: pd.DataFrame, importance: pd.DataFrame) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig_dir = root / "artifacts" / "figures_static"
    html_dir = root / "artifacts" / "figures_interactive"
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

    paths: dict[str, str] = {}

    if not summary.empty:
        plot = summary[summary["target"].eq("fwd_ret_1m")].sort_values("rank_ic_mean")
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.barh(plot["model"], plot["rank_ic_mean"])
            ax.axvline(0, linewidth=1)
            ax.set_title("Step 008 OOS mean monthly rank IC, 1M target")
            ax.set_xlabel("Mean Spearman rank IC")
            fig.tight_layout()
            p = fig_dir / f"008_oos_rank_ic_1m_{run_id}.png"
            fig.savefig(p)
            plt.close(fig)
            paths["rank_ic_1m"] = str(p)

        plot = summary[summary["target"].eq("fwd_ret_1m")].sort_values("annualized_spread_approx")
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.barh(plot["model"], plot["annualized_spread_approx"])
            ax.axvline(0, linewidth=1)
            ax.set_title("Step 008 OOS annualized high-minus-low spread, 1M target")
            ax.set_xlabel("Annualized spread approximation")
            fig.tight_layout()
            p = fig_dir / f"008_oos_spread_1m_{run_id}.png"
            fig.savefig(p)
            plt.close(fig)
            paths["spread_1m"] = str(p)

    if not spread_monthly.empty:
        plot = spread_monthly[spread_monthly["target"].eq("fwd_ret_1m")].copy()
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(11, 6))
            for model_name, g in plot.groupby("model", sort=True):
                g = g.sort_values("month")
                curve = (1.0 + g["high_minus_low"].fillna(0.0)).cumprod() - 1.0
                ax.plot(pd.to_datetime(g["month"]), curve, label=model_name, linewidth=1.4)
            ax.axhline(0, linewidth=1)
            ax.set_title("Step 008 OOS cumulative high-minus-low spread, 1M target")
            ax.set_ylabel("Cumulative spread")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            p = fig_dir / f"008_oos_cumulative_spread_1m_{run_id}.png"
            fig.savefig(p)
            plt.close(fig)
            paths["cumulative_spread_1m"] = str(p)

    if not importance.empty:
        plot = (
            importance[importance["model"].str.contains("lgbm")]
            .groupby(["target", "feature"], observed=True)["importance"]
            .mean()
            .reset_index()
        )
        plot = plot[plot["target"].eq("fwd_ret_1m")].sort_values("importance").tail(18)
        if not plot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(plot["feature"], plot["importance"])
            ax.set_title("Step 008 average LightGBM feature importance, 1M target")
            ax.set_xlabel("Average split importance")
            fig.tight_layout()
            p = fig_dir / f"008_lgbm_feature_importance_1m_{run_id}.png"
            fig.savefig(p)
            plt.close(fig)
            paths["lgbm_importance_1m"] = str(p)

    html = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Step 008 OOS ML signal models</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; max-width: 1250px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 12px; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: right; }}
th {{ background: #f4f4f4; }}
td:first-child, th:first-child {{ text-align: left; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
</style></head><body>
<h1>Step 008 OOS ML signal models</h1>
<p>Run ID: <code>{run_id}</code></p>
<h2>Model summary</h2>
{summary.to_html(index=False) if not summary.empty else '<p>No summary rows.</p>'}
<h2>Recent monthly IC rows</h2>
{ic_monthly.tail(40).to_html(index=False) if not ic_monthly.empty else '<p>No IC rows.</p>'}
<h2>Recent monthly spread rows</h2>
{spread_monthly.tail(40).to_html(index=False) if not spread_monthly.empty else '<p>No spread rows.</p>'}
</body></html>
"""
    p = html_dir / f"008_oos_ml_dashboard_{run_id}.html"
    p.write_text(html, encoding="utf-8")
    paths["interactive_dashboard"] = str(p)
    return paths


def bundle_outputs(root: Path, run_id: str, log_dir: Path) -> Path:
    bundle = root / "artifacts" / "logs" / f"008_oos_ml_{run_id}_logs_and_results.zip"
    if bundle.exists():
        bundle.unlink()
    patterns = [
        f"logs/008_oos_ml_{run_id}/001_run.log",
        f"artifacts/logs/008_oos_ml_manifest_{run_id}.json",
        f"artifacts/tables/008_*_{run_id}.csv",
        f"artifacts/figures_static/008_*_{run_id}.png",
        f"artifacts/figures_interactive/008_*_{run_id}.html",
        "docs/008_oos_ml_results.md",
        "scripts/008_oos_ml_signal_models.py",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pattern in patterns:
            for p in root.glob(pattern):
                if p.is_file():
                    z.write(p, str(p.relative_to(root)))
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--initial-test-year", type=int, default=2012)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = args.run_id
    log_dir = root / "logs" / f"008_oos_ml_{run_id}"
    local_out = root / "data" / "processed" / "008_oos_ml" / run_id
    table_dir = root / "artifacts" / "tables"
    docs_dir = root / "docs"
    for p in [local_out, table_dir, docs_dir, root / "artifacts" / "logs"]:
        p.mkdir(parents=True, exist_ok=True)

    host = socket.gethostname()
    if not host.lower().startswith("compute-node"):
        raise RuntimeError(f"Step 008 is compute-node-only. Current host={host}")

    step005_manifest, panel = find_step005_panel(root)
    step006_manifest = latest_json(root, "artifacts/logs/006_baseline_signal*_*.json")
    step007_manifest = latest_json(root, "artifacts/logs/007_formal_econometrics_manifest_*.json")

    print(f"[008] Step 005 manifest: {step005_manifest}")
    print(f"[008] Step 005 panel:    {panel}")
    print(f"[008] Step 006 manifest: {step006_manifest}")
    print(f"[008] Step 007 manifest: {step007_manifest}")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.n_jobs}")
    desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{qpath(panel)}')").fetchdf()
    cols = set(desc["column_name"].astype(str))
    raw_features = [c for c in RAW_SIGNAL_FEATURES + CONTROL_FEATURES if c in cols]
    targets = [c for c in TARGETS if c in cols]
    required = ["month", "permno"]
    missing_required = [c for c in required if c not in cols]
    if missing_required:
        raise RuntimeError(f"Missing required columns in panel: {missing_required}")
    if not raw_features:
        raise RuntimeError("No expected feature columns found in panel.")
    if not targets:
        raise RuntimeError("No expected target columns found in panel.")

    selected = required + raw_features + targets
    sql = f"SELECT {', '.join(selected)} FROM read_parquet('{qpath(panel)}')"
    df = con.execute(sql).fetchdf()
    con.close()

    print(f"[008] loaded rows={len(df):,}, columns={selected}")
    df["month"] = month_string(df["month"])
    df = df.dropna(subset=["month", "permno"]).copy()
    for c in raw_features + targets:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("[008] cross-sectional z-scoring features by month")
    z_features: list[str] = []
    for c in raw_features:
        zc = f"z_{c}"
        df[zc] = cross_sectional_z(df, c)
        z_features.append(zc)

    interaction_features: list[str] = []
    for a, b in INTERACTION_PAIRS:
        za, zb = f"z_{a}", f"z_{b}"
        if za in df.columns and zb in df.columns:
            name = f"z_{a}_x_{b}"
            df[name] = (df[za] * df[zb]).astype("float32").clip(-8, 8)
            interaction_features.append(name)

    all_features = z_features + interaction_features
    network_features = [c for c in all_features if ("network" in c or "sell_pressure" in c)]
    no_network_features = [c for c in all_features if "network" not in c]

    feature_sets = {
        "ridge_all": all_features,
        "lgbm_all": all_features,
        "lgbm_no_network": no_network_features,
        "lgbm_network_pressure": network_features,
    }
    feature_sets = {k: v for k, v in feature_sets.items() if v}

    months = sorted(df["month"].dropna().unique().tolist())
    print(f"[008] months={len(months)}, first={months[0]}, last={months[-1]}")
    print(f"[008] feature_sets={ {k: len(v) for k, v in feature_sets.items()} }")

    all_prediction_frames = []
    fit_rows = []
    importance_frames = []

    for target in targets:
        target_nonnull = set(df.loc[df[target].notna(), "month"].unique().tolist())
        splits = [s for s in build_splits(months, target, target_nonnull) if s.test_year >= args.initial_test_year]
        print(f"[008] target={target}: splits={len(splits)} years={[s.test_year for s in splits]}")
        if not splits:
            continue

        for split in splits:
            train_mask = df["month"].isin(split.train_months) & df[target].notna()
            valid_mask = df["month"].isin(split.valid_months) & df[target].notna()
            test_mask = df["month"].isin(split.test_months) & df[target].notna()
            if train_mask.sum() < 10000 or valid_mask.sum() < 1000 or test_mask.sum() < 1000:
                print(
                    f"[008] skip {target} {split.test_year}: "
                    f"train={train_mask.sum()}, valid={valid_mask.sum()}, test={test_mask.sum()}"
                )
                continue

            train_df = df.loc[train_mask]
            valid_df = df.loc[valid_mask]
            test_df = df.loc[test_mask]
            y_train = winsor_train_y(train_df, target)
            y_valid = winsor_train_y(valid_df, target)

            print(
                f"[008] split target={target} year={split.test_year} "
                f"train_months={len(split.train_months)} valid_months={len(split.valid_months)} "
                f"test_months={len(split.test_months)} train_rows={len(train_df):,} "
                f"valid_rows={len(valid_df):,} test_rows={len(test_df):,}"
            )

            for model_name, feats in feature_sets.items():
                X_train = train_df[feats].astype("float32")
                X_valid = valid_df[feats].astype("float32")
                X_test = test_df[feats].astype("float32")

                if model_name.startswith("ridge"):
                    pred, params, imp = fit_predict_ridge(X_train, y_train, X_valid, y_valid, X_test, args.n_jobs)
                else:
                    pred, params, imp = fit_predict_lgbm(X_train, y_train, X_valid, y_valid, X_test, args.n_jobs)

                out = test_df[["month", "permno", target]].copy()
                out["model"] = model_name
                out["target"] = target
                out["test_year"] = split.test_year
                out["prediction"] = pred.astype("float32")
                all_prediction_frames.append(out)

                fit_rows.append(
                    {
                        "model": model_name,
                        "target": target,
                        "test_year": split.test_year,
                        "train_months": len(split.train_months),
                        "valid_months": len(split.valid_months),
                        "test_months": len(split.test_months),
                        "train_rows": int(len(train_df)),
                        "valid_rows": int(len(valid_df)),
                        "test_rows": int(len(test_df)),
                        "n_features": len(feats),
                        "params_json": json.dumps(params, sort_keys=True),
                    }
                )
                imp_df = imp.reset_index()
                imp_df.columns = ["feature", "importance"]
                imp_df.insert(0, "test_year", split.test_year)
                imp_df.insert(0, "target", target)
                imp_df.insert(0, "model", model_name)
                importance_frames.append(imp_df)

    if not all_prediction_frames:
        raise RuntimeError("No OOS predictions were generated.")

    preds = pd.concat(all_prediction_frames, ignore_index=True)
    fit_summary = pd.DataFrame(fit_rows)
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()

    print(f"[008] OOS prediction rows={len(preds):,}")

    summary_frames = []
    ic_frames = []
    spread_frames = []
    for target, g in preds.groupby("target", observed=True, sort=True):
        summary, ic_monthly, spread_monthly = evaluate_predictions(g, target)
        summary_frames.append(summary)
        ic_frames.append(ic_monthly)
        spread_frames.append(spread_monthly)

    model_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    rank_ic_monthly = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()
    spread_monthly = pd.concat(spread_frames, ignore_index=True) if spread_frames else pd.DataFrame()

    pred_path = local_out / f"008_oos_predictions_{run_id}.parquet"
    preds.to_parquet(pred_path, index=False)

    table_paths = {
        "model_summary": table_dir / f"008_oos_model_summary_{run_id}.csv",
        "fit_summary": table_dir / f"008_oos_fit_summary_{run_id}.csv",
        "rank_ic_monthly": table_dir / f"008_oos_rank_ic_monthly_{run_id}.csv",
        "spread_monthly": table_dir / f"008_oos_spread_monthly_{run_id}.csv",
        "feature_importance": table_dir / f"008_oos_feature_importance_{run_id}.csv",
    }
    model_summary.to_csv(table_paths["model_summary"], index=False)
    fit_summary.to_csv(table_paths["fit_summary"], index=False)
    rank_ic_monthly.to_csv(table_paths["rank_ic_monthly"], index=False)
    spread_monthly.to_csv(table_paths["spread_monthly"], index=False)
    importance.to_csv(table_paths["feature_importance"], index=False)

    figures = make_figures(root, run_id, model_summary, rank_ic_monthly, spread_monthly, importance)

    top = model_summary.sort_values(["target", "rank_ic_mean"], ascending=[True, False]).groupby("target", observed=True).head(3)
    print("[008] top OOS models by rank IC:")
    print(top.to_string(index=False))

    manifest = {
        "run_id": run_id,
        "status": "ok",
        "created_utc": now_utc(),
        "host": host,
        "n_jobs": args.n_jobs,
        "step005_manifest": str(step005_manifest),
        "source_step005_panel_local_only": str(panel),
        "step006_manifest": str(step006_manifest) if step006_manifest else None,
        "step007_manifest": str(step007_manifest) if step007_manifest else None,
        "rows_loaded": int(len(df)),
        "months": int(len(months)),
        "stocks": int(df["permno"].nunique()),
        "raw_features": raw_features,
        "z_features": z_features,
        "interaction_features": interaction_features,
        "targets": targets,
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "oos_prediction_rows_local_only": int(len(preds)),
        "local_prediction_path": str(pred_path),
        "tables": {k: str(v) for k, v in table_paths.items()},
        "figures": figures,
        "top_models_by_target": top.to_dict("records"),
    }
    manifest_path = root / "artifacts" / "logs" / f"008_oos_ml_manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

    doc = docs_dir / "008_oos_ml_results.md"
    doc.write_text(
        f"""# Step 008 OOS ML signal models

- Run ID: `{run_id}`
- Status: `ok`
- Host: `{host}`
- Input Step 005 manifest: `{step005_manifest.name}`
- Input Step 007 manifest: `{step007_manifest.name if step007_manifest else 'missing'}`
- Rows loaded: `{len(df):,}`
- OOS prediction rows: `{len(preds):,}`

This step trains expanding-window, embargoed out-of-sample ridge and LightGBM models on the full filing-date-clean ownership-network panel. It writes aggregate model diagnostics to public-safe artifacts. Name-month predictions are local-only under `data/processed/008_oos_ml/` and remain gitignored.

## Top models by target

```text\n{top.to_string(index=False)}\n```
""",
        encoding="utf-8",
    )

    bundle = bundle_outputs(root, run_id, log_dir)
    print(f"[008] wrote manifest: {manifest_path}")
    print(f"[008] wrote local prediction parquet: {pred_path}")
    print(f"[008] wrote bundle: {bundle}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        root = Path(os.environ.get("ROOT", "."))
        run_id = os.environ.get("RUN_ID", "unknown")
        fail = root / "artifacts" / "logs" / f"008_oos_ml_manifest_{run_id}_FAILED.json"
        fail.parent.mkdir(parents=True, exist_ok=True)
        fail.write_text(
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
        print(f"[008] FAILED. Wrote failure manifest: {fail}")
        traceback.print_exc()
        raise
