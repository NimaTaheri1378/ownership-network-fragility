from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

REQUIRED_PILOT_ROLES = [
    "13f_holdings",
    "crsp_monthly_stock",
    "crsp_daily_stock",
    "crsp_stock_names",
]

ROLE_LIMIT_ARG = {
    "13f_holdings": "max_rows_13f",
    "13f_manager_names": "max_rows_reference",
    "13f_security_type": "max_rows_reference",
    "crsp_monthly_stock": "max_rows_crsp_monthly",
    "crsp_daily_stock": "max_rows_crsp_daily",
    "crsp_stock_names": "max_rows_reference",
    "ccm_link_history": "max_rows_reference",
    "compustat_annual": "max_rows_reference",
    "ff_monthly_factors": "max_rows_reference",
    "ff_daily_factors": "max_rows_reference",
    "crsp_mutual_fund_holdings": "max_rows_reference",
}

ROLE_DATE_ALIASES = {
    "13f_holdings": ["fdate", "filedate", "filingdate", "filing_date", "rdate", "report_date", "reportdate"],
    "13f_manager_names": ["fdate", "filedate", "rdate", "report_date"],
    "crsp_monthly_stock": ["date", "mthcaldt", "caldt"],
    "crsp_daily_stock": ["date", "dlycaldt", "caldt"],
    "crsp_stock_names": ["namedt", "st_date", "date"],
    "ccm_link_history": ["linkdt", "link_start", "startdt"],
    "compustat_annual": ["datadate", "date"],
    "ff_monthly_factors": ["date", "caldt", "mcaldt"],
    "ff_daily_factors": ["date", "caldt", "mcaldt"],
    "crsp_mutual_fund_holdings": ["report_dt", "caldt", "date"],
}

ROLE_CORE_ALIASES = {
    "13f_holdings": [
        "mgrno", "managerid", "mgr_id", "rdate", "report_date", "fdate", "filedate", "filingdate", "filing_date",
        "cusip", "cusip8", "ncusip", "shares", "sshprnamt", "value", "market_value", "sole", "shared", "no", "type",
    ],
    "13f_manager_names": ["mgrno", "managerid", "mgr_id", "mgrname", "manager_name", "name", "fdate", "rdate"],
    "13f_security_type": ["type", "typecode", "stkcd", "code", "description", "descrip", "security_type"],
    "crsp_monthly_stock": [
        "permno", "permco", "date", "mthcaldt", "caldt", "ret", "retx", "mthret", "mthretx", "prc", "mthprc",
        "shrout", "mthshrout", "vol", "mthvol", "exchcd", "shrcd",
    ],
    "crsp_daily_stock": [
        "permno", "permco", "date", "dlycaldt", "caldt", "ret", "retx", "dlyret", "dlyretx", "prc", "dlyprc", "vol", "shrout",
    ],
    "crsp_stock_names": [
        "permno", "permco", "namedt", "nameendt", "st_date", "end_date", "cusip", "ncusip", "cusip8", "ticker", "comnam", "shrcd", "exchcd", "siccd",
    ],
    "ccm_link_history": ["gvkey", "lpermno", "lpermco", "permno", "linkdt", "linkenddt", "linktype", "linkprim"],
    "compustat_annual": ["gvkey", "datadate", "fyear", "fyr", "at", "ceq", "seq", "txditc", "pstkrv", "sale", "ni", "ib", "capx"],
    "ff_monthly_factors": ["date", "caldt", "mcaldt", "mktrf", "mkt_rf", "rmrf", "smb", "hml", "rf", "umd", "mom", "rmw", "cma"],
    "ff_daily_factors": ["date", "caldt", "mcaldt", "mktrf", "mkt_rf", "rmrf", "smb", "hml", "rf", "umd", "mom", "rmw", "cma"],
    "crsp_mutual_fund_holdings": ["crsp_portno", "portno", "fundno", "permno", "cusip", "report_dt", "caldt", "nbr_shares", "market_val"],
}

KEY_ALIASES_FOR_STATS = [
    "mgrno", "managerid", "mgr_id", "permno", "permco", "gvkey", "cusip", "ncusip", "cusip8", "rdate", "fdate",
    "date", "mthcaldt", "dlycaldt", "namedt", "nameendt", "linkdt", "linkenddt", "datadate",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not candidates:
        raise SystemExit(f"No files found under {directory} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def safe_identifier(value: str, label: str) -> str:
    if not IDENT_RE.match(value or ""):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def qident(value: str) -> str:
    safe_identifier(value, "identifier")
    return '"' + value.replace('"', '""') + '"'


def fqtn_sql(library: str, table: str) -> str:
    return f"{qident(library)}.{qident(table)}"


def role_map_from_contract(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for role in contract.get("roles", []):
        selected = role.get("selected")
        if selected:
            out[role["role"]] = selected
    return out


def lower_to_actual(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for column in columns:
        out.setdefault(column.lower(), column)
    return out


def find_actual(columns: list[str], aliases: list[str]) -> str | None:
    lut = lower_to_actual(columns)
    for alias in aliases:
        if alias.lower() in lut:
            return lut[alias.lower()]
    return None


def selected_columns_for_role(role: str, selected: dict[str, Any], max_columns: int = 48) -> list[str]:
    columns = list(selected.get("columns") or [])
    lut = lower_to_actual(columns)
    chosen: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in chosen:
            chosen.append(name)

    for alias in ROLE_CORE_ALIASES.get(role, []):
        add(lut.get(alias.lower()))

    for hit in selected.get("required_hits", []):
        for alias in hit.get("hits", []):
            add(lut.get(str(alias).lower()))

    for alias in selected.get("preferred_hits", []):
        add(lut.get(str(alias).lower()))

    if len(chosen) < min(8, len(columns)):
        for column in columns[:12]:
            add(column)

    return chosen[:max_columns]


def dtype_for(selected: dict[str, Any], column: str) -> str:
    column_types = selected.get("column_types") or {}
    for key, value in column_types.items():
        if key.lower() == column.lower():
            return str(value or "").lower()
    return ""


def yyyymmdd(date_string: str) -> int:
    return int(date_string.replace("-", ""))


def date_predicate(selected: dict[str, Any], role: str, start: str, end: str) -> tuple[str, str | None]:
    columns = selected.get("columns") or []
    date_col = find_actual(columns, ROLE_DATE_ALIASES.get(role, []))
    if not date_col:
        return "", None
    dtype = dtype_for(selected, date_col)
    col = qident(date_col)

    if any(token in dtype for token in ["date", "time"]):
        return f"{col} >= DATE '{start}' AND {col} <= DATE '{end}'", date_col
    if any(token in dtype for token in ["int", "numeric", "decimal", "double", "real"]):
        return f"{col} >= {yyyymmdd(start)} AND {col} <= {yyyymmdd(end)}", date_col
    if any(token in dtype for token in ["char", "text", "varchar"]):
        return f"{col} >= '{start}' AND {col} <= '{end}'", date_col
    return "", date_col


def clean_sql(sql: str) -> str:
    return " ".join(sql.split())


def try_rollback(db: Any) -> None:
    for attr in ["connection", "conn"]:
        obj = getattr(db, attr, None)
        if obj is not None:
            try:
                obj.rollback()
                return
            except Exception:
                pass
    try:
        raw = getattr(db, "_connection", None)
        if raw is not None:
            raw.rollback()
    except Exception:
        pass


def read_sql_with_fallback(
    db: Any,
    library: str,
    table: str,
    columns: list[str],
    where_sql: str,
    limit: int,
) -> tuple[pd.DataFrame, str, str]:
    select_cols = ", ".join(qident(c) for c in columns)
    table_sql = fqtn_sql(library, table)
    if where_sql:
        query = f"SELECT {select_cols} FROM {table_sql} WHERE {where_sql} LIMIT {int(limit)}"
        try:
            df = db.raw_sql(query)
            if len(df) > 0:
                return df, clean_sql(query), "date_filtered"
        except Exception as exc:
            first_error = repr(exc)
            try_rollback(db)
        else:
            first_error = "date filter returned zero rows"
    else:
        first_error = "no date predicate available"

    query = f"SELECT {select_cols} FROM {table_sql} LIMIT {int(limit)}"
    df = db.raw_sql(query)
    mode = "limit_only" if first_error == "no date predicate available" else f"fallback_limit_only_after_{first_error[:80]}"
    return df, clean_sql(query), mode


def safe_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def to_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        base = pd.Series(index=series.index, dtype="datetime64[ns]")
        non_null = series.dropna()
        if len(non_null):
            as_str = non_null.astype("Int64").astype(str)
            base.loc[as_str.index] = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
        return base
    return pd.to_datetime(series, errors="coerce")


def summarize_df(role: str, selected: dict[str, Any], df: pd.DataFrame, elapsed: float, query_mode: str, query: str, out_path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "role": role,
        "fqtn": selected.get("fqtn"),
        "library": selected.get("library"),
        "table": selected.get("table"),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(map(str, df.columns)),
        "elapsed_sec": round(float(elapsed), 4),
        "query_mode": query_mode,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "local_path": str(out_path),
        "local_bytes": safe_bytes(out_path),
        "key_column_stats": {},
        "date_column_stats": {},
    }

    for column in df.columns:
        lower = column.lower()
        if lower in KEY_ALIASES_FOR_STATS or lower in ROLE_DATE_ALIASES.get(role, []):
            non_null = int(df[column].notna().sum())
            n_unique = int(df[column].nunique(dropna=True))
            stats["key_column_stats"][column] = {
                "non_null": non_null,
                "non_null_rate": round(non_null / max(len(df), 1), 6),
                "n_unique": n_unique,
            }

    date_aliases = {c.lower() for c in ROLE_DATE_ALIASES.get(role, [])}
    for column in df.columns:
        if column.lower() in date_aliases:
            parsed = to_date_series(df[column])
            non_null = int(parsed.notna().sum())
            if non_null:
                stats["date_column_stats"][column] = {
                    "parsed_non_null": non_null,
                    "min": str(parsed.min().date()),
                    "max": str(parsed.max().date()),
                }
            else:
                stats["date_column_stats"][column] = {"parsed_non_null": 0}

    return stats


def extract_role(db: Any, role: str, selected: dict[str, Any], args: argparse.Namespace, pilot_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    library = safe_identifier(str(selected["library"]), "library")
    table = safe_identifier(str(selected["table"]), "table")
    columns = selected_columns_for_role(role, selected)
    if not columns:
        raise RuntimeError(f"No columns selected for role={role}")
    for col in columns:
        safe_identifier(col, "column")

    limit = int(getattr(args, ROLE_LIMIT_ARG.get(role, "max_rows_reference")))
    where_sql, date_col = date_predicate(selected, role, args.pilot_start, args.pilot_end)

    t0 = time.perf_counter()
    df, query, query_mode = read_sql_with_fallback(db, library, table, columns, where_sql, limit)
    elapsed = time.perf_counter() - t0

    out_path = pilot_dir / f"{role}.parquet"
    df.to_parquet(out_path, index=False)
    stats = summarize_df(role, selected, df, elapsed, query_mode, query, out_path)
    stats["date_filter_column"] = date_col
    stats["row_limit"] = limit
    return df, stats


def normalize_cusip8(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .str.slice(0, 8)
        .where(lambda s: s.str.len() >= 6)
    )


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_quality_audits(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    audits: dict[str, Any] = {}

    if "13f_holdings" in frames:
        df = frames["13f_holdings"]
        cols = {c.lower(): c for c in df.columns}
        audit: dict[str, Any] = {"n_rows": int(len(df))}
        if "fdate" in cols and "rdate" in cols:
            fdate = to_date_series(df[cols["fdate"]])
            rdate = to_date_series(df[cols["rdate"]])
            lag = (fdate - rdate).dt.days
            lag = lag.replace([np.inf, -np.inf], np.nan).dropna()
            if len(lag):
                audit["filing_lag_days"] = {
                    "n": int(len(lag)),
                    "mean": round(float(lag.mean()), 4),
                    "p05": round(float(lag.quantile(0.05)), 4),
                    "p50": round(float(lag.quantile(0.50)), 4),
                    "p95": round(float(lag.quantile(0.95)), 4),
                    "min": round(float(lag.min()), 4),
                    "max": round(float(lag.max()), 4),
                }
        for share_col in ["shares", "sshprnamt"]:
            if share_col in cols:
                shares = numeric_series(df[cols[share_col]])
                audit[f"{share_col}_non_null_rate"] = round(float(shares.notna().mean()), 6) if len(shares) else 0.0
                audit[f"{share_col}_positive_rate"] = round(float((shares > 0).mean()), 6) if len(shares) else 0.0
        audits["13f_timing_and_amounts"] = audit

    if "13f_holdings" in frames and "crsp_stock_names" in frames:
        h = frames["13f_holdings"]
        n = frames["crsp_stock_names"]
        h_cols = {c.lower(): c for c in h.columns}
        n_cols = {c.lower(): c for c in n.columns}
        h_cusip_col = next((h_cols[c] for c in ["cusip", "cusip8", "ncusip"] if c in h_cols), None)
        n_cusip_col = next((n_cols[c] for c in ["cusip", "ncusip", "cusip8"] if c in n_cols), None)
        if h_cusip_col and n_cusip_col:
            h_cusip = normalize_cusip8(h[h_cusip_col]).dropna()
            n_cusip = normalize_cusip8(n[n_cusip_col]).dropna()
            overlap = set(h_cusip.unique()).intersection(set(n_cusip.unique()))
            mapped_rate = float(h_cusip.isin(overlap).mean()) if len(h_cusip) else 0.0
            audits["cusip_mapping_probe"] = {
                "n_13f_rows_with_cusip8": int(len(h_cusip)),
                "n_crsp_rows_with_cusip8": int(len(n_cusip)),
                "n_unique_13f_cusip8": int(h_cusip.nunique()),
                "n_unique_crsp_cusip8": int(n_cusip.nunique()),
                "n_unique_overlap_cusip8": int(len(overlap)),
                "pilot_13f_row_cusip_overlap_rate": round(mapped_rate, 6),
            }

    if "crsp_stock_names" in frames:
        n = frames["crsp_stock_names"]
        cols = {c.lower(): c for c in n.columns}
        audit = {"n_rows": int(len(n))}
        if "shrcd" in cols:
            shrcd = numeric_series(n[cols["shrcd"]])
            audit["common_share_code_rate_10_11"] = round(float(shrcd.isin([10, 11]).mean()), 6) if len(shrcd) else 0.0
            audit["n_common_share_code_10_11"] = int(shrcd.isin([10, 11]).sum())
        if "exchcd" in cols:
            exchcd = numeric_series(n[cols["exchcd"]])
            audit["primary_exchange_rate_1_2_3"] = round(float(exchcd.isin([1, 2, 3]).mean()), 6) if len(exchcd) else 0.0
            audit["n_primary_exchange_1_2_3"] = int(exchcd.isin([1, 2, 3]).sum())
        audits["crsp_common_stock_probe"] = audit

    for role in ["crsp_monthly_stock", "crsp_daily_stock"]:
        if role in frames:
            df = frames[role]
            cols = {c.lower(): c for c in df.columns}
            ret_col = next((cols[c] for c in ["ret", "mthret", "dlyret"] if c in cols), None)
            prc_col = next((cols[c] for c in ["prc", "mthprc", "dlyprc", "altprc"] if c in cols), None)
            audit = {"n_rows": int(len(df))}
            if ret_col:
                ret = numeric_series(df[ret_col])
                audit["return_non_null_rate"] = round(float(ret.notna().mean()), 6) if len(ret) else 0.0
                if ret.notna().any():
                    audit["return_p01"] = round(float(ret.quantile(0.01)), 6)
                    audit["return_p50"] = round(float(ret.quantile(0.50)), 6)
                    audit["return_p99"] = round(float(ret.quantile(0.99)), 6)
            if prc_col:
                prc = numeric_series(df[prc_col]).abs()
                audit["price_non_null_rate"] = round(float(prc.notna().mean()), 6) if len(prc) else 0.0
                audit["price_ge_5_rate"] = round(float((prc >= 5).mean()), 6) if len(prc) else 0.0
            audits[f"{role}_return_probe"] = audit

    return audits


def status_from_rows(role: str, n_rows: int) -> str:
    if n_rows > 0:
        return "pass"
    return "fail" if role in REQUIRED_PILOT_ROLES else "warn"


def write_quality_csv(path: Path, table_stats: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["role", "fqtn", "status", "n_rows", "n_cols", "elapsed_sec", "query_mode", "local_bytes"],
        )
        writer.writeheader()
        for stat in table_stats:
            writer.writerow(
                {
                    "role": stat["role"],
                    "fqtn": stat.get("fqtn"),
                    "status": stat.get("status"),
                    "n_rows": stat.get("n_rows"),
                    "n_cols": stat.get("n_cols"),
                    "elapsed_sec": stat.get("elapsed_sec"),
                    "query_mode": stat.get("query_mode"),
                    "local_bytes": stat.get("local_bytes"),
                }
            )


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# 003 Pilot extraction and quality audit",
        "",
        "This audit performs a small local pilot extraction from the frozen WRDS schema contract. The extracted pilot Parquet files stay under ignored local data directories and are not included in the share bundle.",
        "",
        "## Run metadata",
        "",
        f"- run_id: `{manifest['run_id']}`",
        f"- created_utc: `{manifest['created_utc']}`",
        f"- host: `{manifest['host']}`",
        f"- pilot_start: `{manifest['pilot_window']['start']}`",
        f"- pilot_end: `{manifest['pilot_window']['end']}`",
        f"- contract_json: `{manifest['contract_json']}`",
        "",
        "## Table pilot summary",
        "",
        "| Role | Status | Source table | Rows | Columns | Query mode | Seconds |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for stat in manifest["tables"]:
        icon = "✅" if stat.get("status") == "pass" else ("⚠️" if stat.get("status") == "warn" else "❌")
        lines.append(
            f"| `{stat['role']}` | {icon} `{stat.get('status')}` | `{stat.get('fqtn')}` | {stat.get('n_rows')} | {stat.get('n_cols')} | `{stat.get('query_mode')}` | {stat.get('elapsed_sec')} |"
        )
    lines += ["", "## Quality audits", ""]
    for name, audit in manifest.get("audits", {}).items():
        lines.append(f"### `{name}`")
        lines.append("")
        for key, value in audit.items():
            if isinstance(value, dict):
                lines.append(f"- `{key}`:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - `{sub_key}`: `{sub_value}`")
            else:
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_figures(root: Path, run_id: str, table_stats: list[dict[str, Any]]) -> dict[str, str]:
    static_path = root / "artifacts" / "figures_static" / f"003_pilot_extract_rows_{run_id}.png"
    html_path = root / "artifacts" / "figures_interactive" / f"003_pilot_extract_quality_{run_id}.html"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    fig_paths = {"static_png": str(static_path), "interactive_html": str(html_path)}
    labels = [s["role"] for s in table_stats]
    values = [s.get("n_rows", 0) for s in table_stats]

    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.dpi": 140,
                "savefig.dpi": 300,
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.labelsize": 10,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.bbox": "tight",
            }
        )
        fig, ax = plt.subplots(figsize=(11, 5.8))
        ax.bar(range(len(labels)), values)
        ax.set_yscale("symlog")
        ax.set_title("Step 003 pilot extraction row counts")
        ax.set_ylabel("Rows extracted, symlog scale")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(static_path)
        plt.close(fig)
    except Exception as exc:
        static_path.write_text(f"matplotlib figure failed: {exc}\n", encoding="utf-8")

    try:
        import plotly.graph_objects as go

        statuses = [s.get("status", "unknown") for s in table_stats]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=statuses,
                    hovertext=[s.get("fqtn", "") for s in table_stats],
                    hovertemplate="%{x}<br>rows=%{y}<br>%{hovertext}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Step 003 pilot extraction quality",
            xaxis_title="Frozen source role",
            yaxis_title="Rows extracted",
            yaxis_type="log",
            template="plotly_white",
            margin={"l": 70, "r": 30, "t": 70, "b": 120},
        )
        fig.write_html(html_path, include_plotlyjs="cdn")
    except Exception as exc:
        html_path.write_text(f"plotly figure failed: {exc}\n", encoding="utf-8")

    return fig_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-start", default="2019-01-01")
    parser.add_argument("--pilot-end", default="2020-12-31")
    parser.add_argument("--max-rows-13f", type=int, default=75000)
    parser.add_argument("--max-rows-crsp-monthly", type=int, default=60000)
    parser.add_argument("--max-rows-crsp-daily", type=int, default=120000)
    parser.add_argument("--max-rows-reference", type=int, default=50000)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    contract_path = latest_file(root / "artifacts" / "schema", "002_schema_contract_*.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    roles = role_map_from_contract(contract)

    pilot_dir = root / "data" / "interim" / "003_pilot" / args.run_id
    pilot_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "created_utc": utc_now(),
        "host": socket.gethostname(),
        "python": sys.version,
        "project_root": str(root),
        "contract_json": str(contract_path),
        "pilot_window": {"start": args.pilot_start, "end": args.pilot_end},
        "local_pilot_dir": str(pilot_dir),
        "tables": [],
        "audits": {},
        "errors": [],
    }

    print(f"[003_pilot_extract] contract={contract_path}")
    print(f"[003_pilot_extract] pilot_dir={pilot_dir}")
    print(f"[003_pilot_extract] pilot_window={args.pilot_start}..{args.pilot_end}")

    try:
        import wrds

        username = os.environ.get("WRDS_USERNAME") or os.environ.get("USER")
        db = wrds.Connection(wrds_username=username)
    except Exception as exc:
        manifest["status"] = "failed_connection"
        manifest["errors"].append({"stage": "connect", "error": repr(exc), "traceback": traceback.format_exc(limit=8)})
        out = root / "artifacts" / "logs" / f"003_pilot_extract_manifest_{args.run_id}_FAILED.json"
        out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[003_pilot_extract] FAILED connection; wrote {out}")
        return 11

    frames: dict[str, pd.DataFrame] = {}
    try:
        for role in roles:
            if role == "crsp_mutual_fund_holdings":
                continue
            selected = roles[role]
            print(f"[003_pilot_extract] extracting role={role} table={selected.get('fqtn')}")
            try:
                df, stats = extract_role(db, role, selected, args, pilot_dir)
                stats["status"] = status_from_rows(role, int(stats["n_rows"]))
                frames[role] = df
                manifest["tables"].append(stats)
                print(
                    f"[003_pilot_extract] role={role} rows={stats['n_rows']} cols={stats['n_cols']} "
                    f"mode={stats['query_mode']} sec={stats['elapsed_sec']} status={stats['status']}"
                )
            except Exception as exc:
                try_rollback(db)
                status = "fail" if role in REQUIRED_PILOT_ROLES else "warn"
                err = {
                    "role": role,
                    "fqtn": selected.get("fqtn"),
                    "status": status,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
                manifest["errors"].append(err)
                manifest["tables"].append(
                    {
                        "role": role,
                        "fqtn": selected.get("fqtn"),
                        "status": status,
                        "n_rows": 0,
                        "n_cols": 0,
                        "elapsed_sec": 0,
                        "query_mode": "failed",
                        "local_bytes": 0,
                        "error": repr(exc),
                    }
                )
                print(f"[003_pilot_extract] ERROR role={role}: {exc!r}")

    finally:
        try:
            db.close()
        except Exception:
            pass

    manifest["audits"] = compute_quality_audits(frames)
    required_failures = [s for s in manifest["tables"] if s["role"] in REQUIRED_PILOT_ROLES and s.get("status") != "pass"]
    manifest["status"] = "ok" if not required_failures else "failed_required_roles"
    manifest["required_failures"] = required_failures

    logs_dir = root / "artifacts" / "logs"
    tables_dir = root / "artifacts" / "tables"
    manifest_path = logs_dir / f"003_pilot_extract_manifest_{args.run_id}.json"
    quality_csv = tables_dir / f"003_pilot_extract_quality_{args.run_id}.csv"
    md_path = root / "docs" / "003_pilot_extract_audit.md"

    write_quality_csv(quality_csv, manifest["tables"])
    manifest["figures"] = write_figures(root, args.run_id, manifest["tables"])
    manifest["quality_csv"] = str(quality_csv)
    manifest["markdown_report"] = str(md_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(md_path, manifest)

    print(f"[003_pilot_extract] wrote {manifest_path}")
    print(f"[003_pilot_extract] wrote {quality_csv}")
    print(f"[003_pilot_extract] wrote {md_path}")
    print(f"[003_pilot_extract] status={manifest['status']}")

    if manifest["status"] != "ok":
        return 21
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
