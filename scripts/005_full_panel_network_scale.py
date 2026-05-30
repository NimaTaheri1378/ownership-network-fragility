from __future__ import annotations

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import traceback
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HOLDING_MANAGER = ["mgrno", "managerid", "mgr_id", "manager", "cik"]
HOLDING_CUSIP = ["cusip", "cusip8", "ncusip"]
HOLDING_RDATE = ["rdate", "report_date", "reportdate", "periodofreport"]
HOLDING_FDATE = ["fdate", "filedate", "filingdate", "filing_date", "accepted", "acceptance_datetime"]
HOLDING_SHARES = ["shares", "sshprnamt", "share", "shares_held"]
HOLDING_VALUE = ["value", "market_value", "mktval", "valusd"]
NAME_PERMNO = ["permno", "lpermno"]
NAME_PERMCO = ["permco", "lpermco"]
NAME_CUSIP = ["ncusip", "cusip", "cusip8"]
NAME_START = ["namedt", "st_date", "start_date", "startdt"]
NAME_END = ["nameendt", "end_date", "enddt"]
NAME_SHRCD = ["shrcd", "share_code"]
NAME_EXCHCD = ["exchcd", "exchange_code"]
NAME_TICKER = ["ticker", "tic"]
NAME_COMNAM = ["comnam", "company_name", "name"]
MONTHLY_PERMNO = ["permno", "lpermno"]
MONTHLY_PERMCO = ["permco", "lpermco"]
MONTHLY_DATE = ["date", "mthcaldt", "caldt"]
MONTHLY_RET = ["ret", "mthret"]
MONTHLY_RETX = ["retx", "mthretx"]
MONTHLY_PRC = ["prc", "mthprc", "altprc"]
MONTHLY_SHROUT = ["shrout", "mthshrout"]
MONTHLY_VOL = ["vol", "mthvol"]
MONTHLY_SHRCD = ["shrcd", "share_code"]
MONTHLY_EXCHCD = ["exchcd", "exchange_code"]
ROLE_ALIASES = {
    "13f_holdings": HOLDING_MANAGER + HOLDING_CUSIP + HOLDING_RDATE + HOLDING_FDATE + HOLDING_SHARES + HOLDING_VALUE,
    "crsp_stock_names": NAME_PERMNO + NAME_PERMCO + NAME_CUSIP + NAME_START + NAME_END + NAME_SHRCD + NAME_EXCHCD + NAME_TICKER + NAME_COMNAM,
    "crsp_monthly_stock": MONTHLY_PERMNO + MONTHLY_PERMCO + MONTHLY_DATE + MONTHLY_RET + MONTHLY_RETX + MONTHLY_PRC + MONTHLY_SHROUT + MONTHLY_VOL + MONTHLY_SHRCD + MONTHLY_EXCHCD,
}
DATE_ALIASES = {"13f_holdings": HOLDING_FDATE + HOLDING_RDATE, "crsp_monthly_stock": MONTHLY_DATE, "crsp_stock_names": NAME_START}
MIN_COUNTS = {"clean_13f_rows": 100_000, "mapped_common_rows": 25_000, "position_rows": 25_000, "panel_rows": 10_000, "unique_months": 36, "unique_stocks": 500, "unique_managers": 100}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_file(directory: Path, pattern: str) -> Path:
    files = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not files:
        raise SystemExit(f"No file found under {directory} matching {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def qident(x: str) -> str:
    if not IDENT_RE.match(x or ""):
        raise ValueError(f"Unsafe SQL identifier: {x!r}")
    return '"' + x.replace('"', '""') + '"'


def qpath(path: Path | str) -> str:
    return str(path).replace("'", "''")


def fqtn(library: str, table: str) -> str:
    return f"{qident(library)}.{qident(table)}"


def lower_map(cols: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in cols:
        out.setdefault(str(c).lower(), str(c))
    return out


def find_actual(cols: list[str], aliases: list[str]) -> str | None:
    lut = lower_map(cols)
    for a in aliases:
        if a.lower() in lut:
            return lut[a.lower()]
    return None


def require_col(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    c = find_actual([str(x) for x in df.columns], aliases)
    if c is None:
        raise ValueError(f"Missing {label}; tried {aliases}; available={list(df.columns)[:80]}")
    return c


def role_selected(contract: dict[str, Any], role: str) -> dict[str, Any]:
    for r in contract.get("roles", []):
        if r.get("role") == role and r.get("selected"):
            return r["selected"]
    raise SystemExit(f"No selected schema-contract table for role={role}")


def selected_columns(selected: dict[str, Any], role: str, max_cols: int = 64) -> list[str]:
    cols = [str(c) for c in selected.get("columns") or []]
    lut = lower_map(cols)
    chosen: list[str] = []
    def add(c: str | None) -> None:
        if c and c not in chosen:
            chosen.append(c)
    for a in ROLE_ALIASES.get(role, []):
        add(lut.get(a.lower()))
    for hit in selected.get("required_hits", []):
        for a in hit.get("hits", []):
            add(lut.get(str(a).lower()))
    for a in selected.get("preferred_hits", []):
        add(lut.get(str(a).lower()))
    for c in cols[:20]:
        add(c)
    return chosen[:max_cols]


def yyyymmdd(d: str) -> int:
    return int(d.replace("-", ""))


def month_windows(start: str, end: str, months: int) -> list[tuple[str, str]]:
    cur = pd.Timestamp(start).normalize().replace(day=1)
    end_ts = pd.Timestamp(end).normalize()
    out: list[tuple[str, str]] = []
    while cur <= end_ts:
        nxt = cur + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        if nxt > end_ts:
            nxt = end_ts
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)
    return out


def date_predicates(col: str, start: str, end: str) -> list[str]:
    c = qident(col)
    si, ei = yyyymmdd(start), yyyymmdd(end)
    return [
        f"{c} BETWEEN DATE '{start}' AND DATE '{end}'",
        f"CAST({c} AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'",
        f"{c} BETWEEN {si} AND {ei}",
        f"CAST({c} AS BIGINT) BETWEEN {si} AND {ei}",
        f"CAST({c} AS TEXT) BETWEEN '{start}' AND '{end}'",
        f"CAST({c} AS TEXT) BETWEEN '{si}' AND '{ei}'",
    ]


def parse_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s.astype("string").str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def parse_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        nn = x.dropna().astype("Int64").astype(str)
        ymd = nn.str.fullmatch(r"[12][0-9]{7}")
        ym = nn.str.fullmatch(r"[12][0-9]{5}")
        if ymd.any():
            out.loc[nn[ymd].index] = pd.to_datetime(nn[ymd], format="%Y%m%d", errors="coerce")
        if ym.any():
            out.loc[nn[ym].index] = pd.to_datetime(nn[ym] + "01", format="%Y%m%d", errors="coerce")
        return out
    text = s.astype("string").str.strip()
    out = pd.to_datetime(text, errors="coerce")
    mask = out.isna() & text.str.fullmatch(r"[12][0-9]{7}").fillna(False)
    if mask.any():
        out.loc[mask] = pd.to_datetime(text.loc[mask], format="%Y%m%d", errors="coerce")
    return out


def cusip8(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.slice(0, 8).where(lambda x: x.str.len() >= 6)


def month_date(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d, errors="coerce").dt.to_period("M").dt.to_timestamp()


def extract_role(db: Any, contract: dict[str, Any], role: str, raw_dir: Path, start: str, end: str, chunk_months: int) -> dict[str, Any]:
    selected = role_selected(contract, role)
    library, table = selected["library"], selected["table"]
    cols = selected_columns(selected, role)
    select_sql = ", ".join(qident(c) for c in cols)
    out_dir = raw_dir / role
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"role": role, "library": library, "table": table, "columns": cols, "parts": [], "rows": 0, "status": "started"}
    if role == "crsp_stock_names":
        sql = f"SELECT {select_sql} FROM {fqtn(library, table)}"
        print(f"[005][extract] {role}: selected-column full pull")
        df = db.raw_sql(sql)
        path = out_dir / f"{role}_all.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        stats["parts"].append({"path": str(path), "rows": int(len(df)), "where": "all"})
        stats["rows"] = int(len(df)); stats["status"] = "ok"
        return stats
    date_col = find_actual([str(c) for c in selected.get("columns") or []], DATE_ALIASES[role])
    if not date_col:
        raise RuntimeError(f"No date column for role={role}")
    chosen_variant: int | None = None
    for i, (a, b) in enumerate(month_windows(start, end, chunk_months), start=1):
        preds = date_predicates(date_col, a, b)
        order = ([chosen_variant] if chosen_variant is not None else []) + [j for j in range(len(preds)) if j != chosen_variant]
        df = None; last_error = None; used = None
        for j in order:
            sql = f"SELECT {select_sql} FROM {fqtn(library, table)} WHERE {preds[j]}"
            try:
                df = db.raw_sql(sql)
                chosen_variant = used = j
                break
            except Exception as exc:
                last_error = repr(exc)
                try:
                    db.connection.rollback()
                except Exception:
                    pass
        if df is None:
            raise RuntimeError(f"All predicates failed for {role} {a}:{b}. Last error={last_error}")
        path = out_dir / f"{role}_part_{i:04d}_{a}_{b}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        rows = int(len(df)); stats["rows"] += rows
        stats["parts"].append({"path": str(path), "rows": rows, "start": a, "end": b, "predicate_variant": used})
        print(f"[005][extract] {role} {a}..{b}: rows={rows:,} variant={used}")
        del df
    stats["status"] = "ok"
    return stats


def standardize_holdings(raw: pd.DataFrame, part_tag: str, start: str, end: str) -> pd.DataFrame:
    mgr = require_col(raw, HOLDING_MANAGER, "13F manager")
    cus = require_col(raw, HOLDING_CUSIP, "13F cusip")
    rdt = find_actual([str(c) for c in raw.columns], HOLDING_RDATE)
    fdt = find_actual([str(c) for c in raw.columns], HOLDING_FDATE)
    shr = find_actual([str(c) for c in raw.columns], HOLDING_SHARES)
    val = find_actual([str(c) for c in raw.columns], HOLDING_VALUE)
    out = pd.DataFrame({"manager_id": raw[mgr].astype("string"), "cusip8": cusip8(raw[cus])})
    out["report_date"] = parse_date(raw[rdt]) if rdt else pd.NaT
    out["filing_date"] = parse_date(raw[fdt]) if fdt else pd.NaT
    out["available_date"] = out["filing_date"].fillna(out["report_date"] + pd.Timedelta(days=45))
    out["mapping_date"] = out["report_date"].fillna(out["available_date"])
    out["available_month"] = month_date(out["available_date"])
    out["report_quarter"] = out["report_date"].dt.to_period("Q").astype("string")
    out["filing_lag_days"] = (out["available_date"] - out["report_date"]).dt.days
    out["shares"] = parse_number(raw[shr]) if shr else 1.0
    out["reported_value"] = parse_number(raw[val]) if val else np.nan
    out = out.dropna(subset=["manager_id", "cusip8", "available_date", "available_month"])
    out = out[(out["available_date"] >= pd.Timestamp(start)) & (out["available_date"] <= pd.Timestamp(end))]
    out = out[out["shares"].fillna(0) > 0].copy()
    group_cols = ["manager_id", "cusip8", "report_date", "filing_date", "available_date", "mapping_date", "available_month", "report_quarter", "filing_lag_days"]
    out = out.groupby(group_cols, dropna=False, as_index=False).agg(shares=("shares", "sum"), reported_value=("reported_value", "sum"))
    out["holding_id"] = part_tag + "_" + np.arange(len(out), dtype=np.int64).astype(str)
    return out


def standardize_names(raw: pd.DataFrame) -> pd.DataFrame:
    permno = require_col(raw, NAME_PERMNO, "PERMNO"); cus = require_col(raw, NAME_CUSIP, "CUSIP")
    permco = find_actual([str(c) for c in raw.columns], NAME_PERMCO); start = find_actual([str(c) for c in raw.columns], NAME_START); end = find_actual([str(c) for c in raw.columns], NAME_END)
    shrcd = find_actual([str(c) for c in raw.columns], NAME_SHRCD); exchcd = find_actual([str(c) for c in raw.columns], NAME_EXCHCD); ticker = find_actual([str(c) for c in raw.columns], NAME_TICKER); comnam = find_actual([str(c) for c in raw.columns], NAME_COMNAM)
    out = pd.DataFrame({"permno": pd.to_numeric(raw[permno], errors="coerce").astype("Int64"), "cusip8": cusip8(raw[cus])})
    out["permco"] = pd.to_numeric(raw[permco], errors="coerce").astype("Int64") if permco else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["name_start"] = parse_date(raw[start]) if start else pd.Timestamp("1900-01-01")
    out["name_end"] = parse_date(raw[end]) if end else pd.Timestamp("2099-12-31")
    out["name_start"] = out["name_start"].fillna(pd.Timestamp("1900-01-01")); out["name_end"] = out["name_end"].fillna(pd.Timestamp("2099-12-31"))
    out["shrcd"] = pd.to_numeric(raw[shrcd], errors="coerce").astype("Int64") if shrcd else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["exchcd"] = pd.to_numeric(raw[exchcd], errors="coerce").astype("Int64") if exchcd else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["ticker"] = raw[ticker].astype("string") if ticker else pd.Series(pd.NA, index=raw.index, dtype="string")
    out["comnam"] = raw[comnam].astype("string") if comnam else pd.Series(pd.NA, index=raw.index, dtype="string")
    out = out.dropna(subset=["permno", "cusip8"]).drop_duplicates().copy()
    out["is_common_stock"] = out["shrcd"].isin([10, 11]) if out["shrcd"].notna().any() else True
    return out


def standardize_monthly(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        raw = pd.read_parquet(p)
        permno = require_col(raw, MONTHLY_PERMNO, "monthly PERMNO"); date = require_col(raw, MONTHLY_DATE, "monthly date")
        permco = find_actual([str(c) for c in raw.columns], MONTHLY_PERMCO); ret = find_actual([str(c) for c in raw.columns], MONTHLY_RET); retx = find_actual([str(c) for c in raw.columns], MONTHLY_RETX); prc = find_actual([str(c) for c in raw.columns], MONTHLY_PRC); shrout = find_actual([str(c) for c in raw.columns], MONTHLY_SHROUT); vol = find_actual([str(c) for c in raw.columns], MONTHLY_VOL); shrcd = find_actual([str(c) for c in raw.columns], MONTHLY_SHRCD); exchcd = find_actual([str(c) for c in raw.columns], MONTHLY_EXCHCD)
        out = pd.DataFrame({"permno": pd.to_numeric(raw[permno], errors="coerce").astype("Int64"), "date": parse_date(raw[date])})
        out["permco"] = pd.to_numeric(raw[permco], errors="coerce").astype("Int64") if permco else pd.Series(pd.NA, index=raw.index, dtype="Int64")
        out["month_date"] = month_date(out["date"]); out["month"] = out["month_date"].dt.strftime("%Y-%m")
        out["ret"] = parse_number(raw[ret]) if ret else np.nan; out["retx"] = parse_number(raw[retx]) if retx else np.nan; out["prc"] = parse_number(raw[prc]) if prc else np.nan; out["shrout"] = parse_number(raw[shrout]) if shrout else np.nan; out["vol"] = parse_number(raw[vol]) if vol else np.nan
        out["shrcd"] = pd.to_numeric(raw[shrcd], errors="coerce").astype("Int64") if shrcd else pd.Series(pd.NA, index=raw.index, dtype="Int64")
        out["exchcd"] = pd.to_numeric(raw[exchcd], errors="coerce").astype("Int64") if exchcd else pd.Series(pd.NA, index=raw.index, dtype="Int64")
        out["mktcap_proxy"] = out["prc"].abs() * out["shrout"]
        frames.append(out.dropna(subset=["permno", "date", "month_date", "month"]))
        del raw
    df = pd.concat(frames, ignore_index=True).sort_values(["permno", "date"]).drop_duplicates(["permno", "month"], keep="last")
    g = df.groupby("permno", observed=True)["ret"]
    df["fwd_ret_1m"] = g.shift(-1); r2 = g.shift(-2); r3 = g.shift(-3); df["fwd_ret_3m"] = (1 + df["fwd_ret_1m"]) * (1 + r2) * (1 + r3) - 1
    return df.replace([np.inf, -np.inf], np.nan)


def standardize_all(raw_dir: Path, clean_dir: Path, start: str, end: str) -> dict[str, Any]:
    if clean_dir.exists(): shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    h_dir = clean_dir / "clean_13f_holdings"; h_dir.mkdir(parents=True, exist_ok=True)
    h_rows = 0
    for i, p in enumerate(sorted((raw_dir / "13f_holdings").glob("*.parquet")), start=1):
        raw = pd.read_parquet(p); clean = standardize_holdings(raw, f"h{i:05d}", start, end)
        clean.to_parquet(h_dir / f"clean_13f_holdings_part_{i:05d}.parquet", index=False, compression="zstd")
        h_rows += int(len(clean)); print(f"[005][standardize] holdings part={i:04d} raw={len(raw):,} clean={len(clean):,}")
        del raw, clean
    nfiles = sorted((raw_dir / "crsp_stock_names").glob("*.parquet")); names = standardize_names(pd.concat([pd.read_parquet(p) for p in nfiles], ignore_index=True)); names_path = clean_dir / "crsp_stock_names.parquet"; names.to_parquet(names_path, index=False, compression="zstd"); print(f"[005][standardize] stocknames rows={len(names):,}")
    mfiles = sorted((raw_dir / "crsp_monthly_stock").glob("*.parquet")); monthly = standardize_monthly(mfiles); monthly_path = clean_dir / "crsp_monthly_stock.parquet"; monthly.to_parquet(monthly_path, index=False, compression="zstd"); print(f"[005][standardize] monthly rows={len(monthly):,}")
    return {"clean_13f_rows": h_rows, "stocknames_rows": int(len(names)), "monthly_rows": int(len(monthly)), "clean_holdings_dir": str(h_dir), "stocknames_path": str(names_path), "monthly_path": str(monthly_path), "status": "ok"}


def duckdb_build(clean_dir: Path, out_dir: Path, n_jobs: int, end_date: str) -> dict[str, Any]:
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(out_dir / "005_full_panel.duckdb")); con.execute(f"PRAGMA threads={int(n_jobs)}"); con.execute("PRAGMA memory_limit='180GB'"); con.execute("PRAGMA preserve_insertion_order=false")
    holdings_glob = qpath(clean_dir / "clean_13f_holdings" / "*.parquet"); names_path = qpath(clean_dir / "crsp_stock_names.parquet"); monthly_path = qpath(clean_dir / "crsp_monthly_stock.parquet")
    print("[005][duckdb] creating local tables")
    con.execute(f"CREATE OR REPLACE TABLE holdings AS SELECT * FROM read_parquet('{holdings_glob}')")
    con.execute(f"CREATE OR REPLACE TABLE names AS SELECT * FROM read_parquet('{names_path}')")
    con.execute(f"CREATE OR REPLACE TABLE monthly AS SELECT * FROM read_parquet('{monthly_path}')")
    print("[005][duckdb] mapping 13F CUSIP8 to date-valid CRSP common stocks")
    con.execute("""
        CREATE OR REPLACE TABLE mapped_common AS
        WITH ranked AS (
            SELECT h.*, n.permno, n.permco, n.shrcd, n.exchcd, n.ticker, n.comnam, n.is_common_stock,
                   ROW_NUMBER() OVER (PARTITION BY h.holding_id ORDER BY CAST(n.is_common_stock AS INTEGER) DESC, n.name_start DESC, n.name_end ASC, n.permno ASC) AS rn
            FROM holdings h JOIN names n ON h.cusip8=n.cusip8 AND h.mapping_date>=n.name_start AND h.mapping_date<=n.name_end
        ) SELECT * EXCLUDE(rn) FROM ranked WHERE rn=1 AND COALESCE(is_common_stock, TRUE)
    """)
    con.execute("""
        CREATE OR REPLACE TABLE mapped_snapshot AS
        SELECT manager_id, permno, MAX(permco) AS permco, report_date, available_date, available_month, mapping_date, MAX(report_quarter) AS report_quarter,
               SUM(shares) AS shares, SUM(reported_value) AS reported_value, AVG(filing_lag_days) AS filing_lag_days
        FROM mapped_common GROUP BY manager_id, permno, report_date, available_date, available_month, mapping_date
    """)
    print("[005][duckdb] expanding filings to active monthly manager-stock positions")
    con.execute(f"""
        CREATE OR REPLACE TABLE filing_windows AS
        WITH filings AS (SELECT DISTINCT manager_id, report_date, available_date, available_month AS active_start_month FROM mapped_snapshot WHERE available_month <= DATE '{end_date}'),
        ordered AS (SELECT *, LEAD(active_start_month) OVER (PARTITION BY manager_id ORDER BY active_start_month, available_date, report_date) AS next_active_start_month FROM filings)
        SELECT manager_id, report_date, available_date, active_start_month,
               CASE WHEN next_active_start_month IS NOT NULL AND next_active_start_month > active_start_month THEN CAST(next_active_start_month - INTERVAL 1 MONTH AS DATE) ELSE CAST(active_start_month + INTERVAL 5 MONTH AS DATE) END AS active_end_month
        FROM ordered WHERE active_start_month IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TABLE manager_stock_positions AS
        WITH expanded AS (
            SELECT gs.month_date::DATE AS month_date, STRFTIME(gs.month_date::DATE, '%Y-%m') AS month, h.manager_id, h.permno, h.permco, h.report_date, h.available_date, h.report_quarter, h.shares, h.reported_value, h.filing_lag_days,
                   ROW_NUMBER() OVER (PARTITION BY gs.month_date::DATE, h.manager_id, h.permno ORDER BY h.available_date DESC, h.report_date DESC) AS rn
            FROM mapped_snapshot h JOIN filing_windows w ON h.manager_id=w.manager_id AND h.report_date IS NOT DISTINCT FROM w.report_date AND h.available_date IS NOT DISTINCT FROM w.available_date
            JOIN generate_series(w.active_start_month, w.active_end_month, INTERVAL 1 MONTH) AS gs(month_date) ON TRUE
        ), current_snapshot AS (SELECT * EXCLUDE(rn) FROM expanded WHERE rn=1),
        valued AS (
            SELECT c.*, m.ret, m.retx, m.prc, m.shrout, m.vol, m.mktcap_proxy, m.fwd_ret_1m, m.fwd_ret_3m,
                   CASE WHEN c.reported_value IS NOT NULL AND c.reported_value > 0 THEN c.reported_value WHEN m.prc IS NOT NULL AND ABS(m.prc)>0 THEN c.shares*ABS(m.prc) ELSE c.shares END AS position_value_proxy
            FROM current_snapshot c LEFT JOIN monthly m ON c.permno=m.permno AND c.month_date=m.month_date
        ), positive AS (SELECT * FROM valued WHERE position_value_proxy IS NOT NULL AND position_value_proxy > 0),
        weighted AS (SELECT *, SUM(position_value_proxy) OVER (PARTITION BY month_date, manager_id) AS manager_total_value_proxy FROM positive)
        SELECT *, position_value_proxy / NULLIF(manager_total_value_proxy, 0) AS portfolio_weight, YEAR(month_date)::INTEGER AS year FROM weighted WHERE manager_total_value_proxy > 0
    """)
    print("[005][duckdb] stock ownership features")
    con.execute("""CREATE OR REPLACE TABLE manager_features AS SELECT month_date, month, manager_id, COUNT(DISTINCT permno) AS manager_breadth, SUM(position_value_proxy) AS manager_total_value_proxy, SUM(portfolio_weight*portfolio_weight) AS manager_concentration FROM manager_stock_positions GROUP BY month_date, month, manager_id""")
    con.execute("""
        CREATE OR REPLACE TABLE pos_enriched AS
        WITH joined AS (SELECT p.*, mf.manager_breadth, mf.manager_concentration, SUM(p.position_value_proxy) OVER (PARTITION BY p.month_date, p.permno) AS stock_total_value_proxy FROM manager_stock_positions p LEFT JOIN manager_features mf ON p.month_date=mf.month_date AND p.manager_id=mf.manager_id),
        owned AS (SELECT *, position_value_proxy / NULLIF(stock_total_value_proxy, 0) AS stock_owner_share, LAG(position_value_proxy) OVER (PARTITION BY manager_id, permno ORDER BY month_date) AS lag_position_value_proxy FROM joined)
        SELECT *, GREATEST(-(position_value_proxy - COALESCE(lag_position_value_proxy, position_value_proxy)), 0) AS sell_amount_proxy FROM owned
    """)
    con.execute("""
        CREATE OR REPLACE TABLE stock_ownership_features AS
        SELECT month_date, month, permno, MAX(permco) AS permco, COUNT(DISTINCT manager_id) AS owner_count, SUM(position_value_proxy) AS total_position_value_proxy,
               SUM(stock_owner_share*stock_owner_share) AS owner_hhi, MAX(stock_owner_share) AS top_owner_share, AVG(manager_concentration) AS fragility_proxy,
               AVG(manager_breadth) AS avg_manager_breadth, SUM(sell_amount_proxy)/NULLIF(SUM(position_value_proxy), 0) AS stock_sell_pressure,
               MAX(ret) AS ret, MAX(retx) AS retx, MAX(prc) AS prc, MAX(shrout) AS shrout, MAX(vol) AS vol, MAX(mktcap_proxy) AS mktcap_proxy, MAX(fwd_ret_1m) AS fwd_ret_1m, MAX(fwd_ret_3m) AS fwd_ret_3m, YEAR(month_date)::INTEGER AS year
        FROM pos_enriched GROUP BY month_date, month, permno
    """)
    pos_dir = out_dir / "manager_stock_positions"; pos_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM manager_stock_positions ORDER BY month_date, manager_id, permno) TO '{qpath(pos_dir)}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (year))")
    stock_path = out_dir / "stock_ownership_features.parquet"; con.execute(f"COPY (SELECT * FROM stock_ownership_features ORDER BY month_date, permno) TO '{qpath(stock_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    mapped_path = out_dir / "mapped_common_holdings.parquet"; con.execute(f"COPY (SELECT * FROM mapped_common ORDER BY available_date, manager_id, permno) TO '{qpath(mapped_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    monthly_summary_path = out_dir / "monthly_coverage.csv"; con.execute("SELECT month, COUNT(*) AS position_rows, COUNT(DISTINCT permno) AS stocks, COUNT(DISTINCT manager_id) AS managers, AVG(portfolio_weight) AS avg_portfolio_weight FROM manager_stock_positions GROUP BY month ORDER BY month").fetchdf().to_csv(monthly_summary_path, index=False)
    counts = {"clean_13f_rows": int(con.execute("SELECT COUNT(*) FROM holdings").fetchone()[0]), "mapped_common_rows": int(con.execute("SELECT COUNT(*) FROM mapped_common").fetchone()[0]), "position_rows": int(con.execute("SELECT COUNT(*) FROM manager_stock_positions").fetchone()[0]), "panel_rows_pre_network": int(con.execute("SELECT COUNT(*) FROM stock_ownership_features").fetchone()[0]), "unique_months": int(con.execute("SELECT COUNT(DISTINCT month) FROM stock_ownership_features").fetchone()[0]), "unique_stocks": int(con.execute("SELECT COUNT(DISTINCT permno) FROM stock_ownership_features").fetchone()[0]), "unique_managers": int(con.execute("SELECT COUNT(DISTINCT manager_id) FROM manager_stock_positions").fetchone()[0])}
    counts["common_mapping_rate"] = round(counts["mapped_common_rows"] / max(counts["clean_13f_rows"], 1), 6)
    con.close()
    return {"positions_dir": str(pos_dir), "stock_features_path": str(stock_path), "mapped_common_path": str(mapped_path), "monthly_summary_path": str(monthly_summary_path), "counts": counts}


def network_worker(args: tuple[str, str, str, str]) -> dict[str, Any]:
    month, pos_glob, stock_path, out_dir_text = args
    out_dir = Path(out_dir_text); con = duckdb.connect()
    pos = con.execute(f"SELECT manager_id, permno, portfolio_weight FROM read_parquet('{pos_glob}', hive_partitioning=true) WHERE month=? AND portfolio_weight IS NOT NULL AND portfolio_weight>0", [month]).fetchdf()
    stock = con.execute(f"SELECT permno, stock_sell_pressure FROM read_parquet('{stock_path}') WHERE month=?", [month]).fetchdf(); con.close()
    if pos.empty:
        out = pd.DataFrame(columns=["month", "month_date", "permno", "network_degree", "network_weighted_degree", "network_peer_sell_pressure"])
        path = out_dir / f"network_features_{month}.parquet"; out.to_parquet(path, index=False, compression="zstd"); return {"month": month, "rows": 0, "nonzeros": 0, "status": "empty"}
    pos["manager_id"] = pos["manager_id"].astype("string"); pos["permno"] = pd.to_numeric(pos["permno"], errors="coerce").astype("Int64"); pos["portfolio_weight"] = pd.to_numeric(pos["portfolio_weight"], errors="coerce"); pos = pos.dropna(subset=["manager_id", "permno", "portfolio_weight"]); pos = pos[pos["portfolio_weight"] > 0]
    managers = pd.Index(pd.unique(pos["manager_id"])); stocks = pd.Index(pd.unique(pos["permno"])); mc = pd.Categorical(pos["manager_id"], categories=managers).codes; sc = pd.Categorical(pos["permno"], categories=stocks).codes
    W = sparse.csr_matrix((pos["portfolio_weight"].to_numpy(float), (mc, sc)), shape=(len(managers), len(stocks)))
    G = (W.T @ W).tocsr(); G.setdiag(0.0); G.eliminate_zeros()
    deg = np.asarray(G.getnnz(axis=1)).ravel(); wdeg = np.asarray(G.sum(axis=1)).ravel(); sell_map = stock.set_index("permno")["stock_sell_pressure"].to_dict() if not stock.empty else {}; sell = np.array([float(sell_map.get(int(p), 0.0) or 0.0) for p in stocks], dtype=float); peer = np.divide(G.dot(sell), wdeg, out=np.zeros_like(wdeg), where=wdeg != 0)
    out = pd.DataFrame({"month": month, "month_date": pd.Timestamp(month + "-01"), "permno": stocks.astype("Int64"), "network_degree": deg.astype(np.int64), "network_weighted_degree": wdeg, "network_peer_sell_pressure": peer})
    path = out_dir / f"network_features_{month}.parquet"; out.to_parquet(path, index=False, compression="zstd")
    return {"month": month, "rows": int(len(out)), "nonzeros": int(G.nnz), "managers": int(len(managers)), "stocks": int(len(stocks)), "status": "ok"}


def build_network(out_dir: Path, panel_info: dict[str, Any], jobs: int) -> dict[str, Any]:
    n_dir = out_dir / "network_features_by_month"; shutil.rmtree(n_dir, ignore_errors=True); n_dir.mkdir(parents=True, exist_ok=True)
    pos_glob = qpath(Path(panel_info["positions_dir"]) / "**" / "*.parquet"); stock_path = qpath(Path(panel_info["stock_features_path"]))
    con = duckdb.connect(); months = [r[0] for r in con.execute(f"SELECT DISTINCT month FROM read_parquet('{stock_path}') ORDER BY month").fetchall()]; con.close()
    print(f"[005][network] months={len(months):,} jobs={jobs}")
    tasks = [(m, pos_glob, stock_path, str(n_dir)) for m in months]; results = []
    workers = max(1, min(int(jobs), len(tasks) or 1))
    if workers == 1:
        for t in tasks:
            res = network_worker(t); results.append(res); print(f"[005][network] {res['month']} rows={res['rows']:,} nnz={res['nonzeros']:,}")
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(network_worker, tasks, chunksize=1):
                results.append(res); print(f"[005][network] {res['month']} rows={res['rows']:,} nnz={res['nonzeros']:,}")
    n_glob = qpath(n_dir / "*.parquet"); final_panel = out_dir / "stock_month_network_panel.parquet"; con = duckdb.connect(); con.execute(f"PRAGMA threads={int(os.environ.get('ONF_N_JOBS', '32'))}")
    con.execute(f"""COPY (SELECT sf.*, COALESCE(nf.network_degree,0) AS network_degree, COALESCE(nf.network_weighted_degree,0.0) AS network_weighted_degree, COALESCE(nf.network_peer_sell_pressure,0.0) AS network_peer_sell_pressure FROM read_parquet('{stock_path}') sf LEFT JOIN read_parquet('{n_glob}') nf ON sf.month=nf.month AND sf.permno=nf.permno ORDER BY sf.month_date, sf.permno) TO '{qpath(final_panel)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    counts = {"panel_rows": int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{qpath(final_panel)}')").fetchone()[0]), "panel_fwd_ret_1m_coverage": float(con.execute(f"SELECT AVG(CASE WHEN fwd_ret_1m IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM read_parquet('{qpath(final_panel)}')").fetchone()[0] or 0), "panel_fwd_ret_3m_coverage": float(con.execute(f"SELECT AVG(CASE WHEN fwd_ret_3m IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM read_parquet('{qpath(final_panel)}')").fetchone()[0] or 0), "median_owner_count": float(con.execute(f"SELECT MEDIAN(owner_count) FROM read_parquet('{qpath(final_panel)}')").fetchone()[0] or 0), "median_network_degree": float(con.execute(f"SELECT MEDIAN(network_degree) FROM read_parquet('{qpath(final_panel)}')").fetchone()[0] or 0), "median_network_weighted_degree": float(con.execute(f"SELECT MEDIAN(network_weighted_degree) FROM read_parquet('{qpath(final_panel)}')").fetchone()[0] or 0)}; con.close()
    return {"network_dir": str(n_dir), "network_months": len(results), "total_graph_nonzeros": int(sum(int(r.get("nonzeros", 0)) for r in results)), "final_panel_path": str(final_panel), "counts": counts, "network_results_sample": sorted(results, key=lambda x: x.get("month", ""))[:10]}


def write_outputs(root: Path, run_id: str, manifest: dict[str, Any]) -> None:
    logs = root / "artifacts" / "logs"; tables = root / "artifacts" / "tables"; figs = root / "artifacts" / "figures_static"; htmls = root / "artifacts" / "figures_interactive"
    for d in [logs, tables, figs, htmls]: d.mkdir(parents=True, exist_ok=True)
    metrics = manifest["metrics"]
    pd.DataFrame([{"metric": k, "value": json.dumps(v, default=str) if isinstance(v, (dict, list)) else v} for k, v in metrics.items()]).to_csv(tables / f"005_full_panel_metrics_{run_id}.csv", index=False)
    shutil.copyfile(manifest["panel_info"]["monthly_summary_path"], tables / f"005_full_panel_monthly_coverage_{run_id}.csv")
    try:
        import matplotlib.pyplot as plt
        keys = ["clean_13f_rows", "mapped_common_rows", "position_rows", "panel_rows"]; vals = [float(metrics.get(k, 0) or 0) for k in keys]
        fig, ax = plt.subplots(figsize=(9.5, 4.8)); ax.barh([k.replace("_", " ") for k in keys], vals); ax.set_title("Step 005 full-scale mapping and panel funnel"); ax.set_xlabel("Rows"); ax.grid(axis="x", alpha=0.25); fig.tight_layout(); fig.savefig(figs / f"005_mapping_funnel_{run_id}.png", dpi=220); plt.close(fig)
        monthly = pd.read_csv(manifest["panel_info"]["monthly_summary_path"]); fig, ax = plt.subplots(figsize=(12, 5)); x = pd.to_datetime(monthly["month"] + "-01"); ax.plot(x, monthly["stocks"], label="stocks"); ax.plot(x, monthly["managers"], label="managers"); ax.legend(); ax.set_title("Step 005 monthly ownership-network coverage"); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(figs / f"005_monthly_coverage_{run_id}.png", dpi=220); plt.close(fig)
    except Exception as exc:
        print(f"[005][figure-warning] matplotlib failed: {exc}")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        monthly = pd.read_csv(manifest["panel_info"]["monthly_summary_path"]); fig = make_subplots(rows=2, cols=2, specs=[[{"type":"bar"},{"type":"scatter"}], [{"type":"table"},{"type":"bar"}]], subplot_titles=("Mapping funnel", "Monthly coverage", "Metrics", "Validation ratios"))
        keys = ["clean_13f_rows", "mapped_common_rows", "position_rows", "panel_rows"]; fig.add_trace(go.Bar(x=[metrics.get(k,0) for k in keys], y=keys, orientation="h"), row=1, col=1); fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["stocks"], mode="lines", name="stocks"), row=1, col=2); fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["managers"], mode="lines", name="managers"), row=1, col=2)
        show = [k for k,v in metrics.items() if not isinstance(v, (list, dict))][:28]; fig.add_trace(go.Table(header={"values":["Metric", "Value"]}, cells={"values":[show, [metrics[k] for k in show]]}), row=2, col=1); thresh = list(MIN_COUNTS); fig.add_trace(go.Bar(x=thresh, y=[metrics.get(k,0)/max(MIN_COUNTS[k],1) for k in thresh]), row=2, col=2); fig.update_layout(title="Step 005 full-scale ownership-network audit", template="plotly_white", height=900); fig.write_html(htmls / f"005_full_panel_network_{run_id}.html", include_plotlyjs="cdn")
    except Exception as exc:
        (htmls / f"005_full_panel_network_{run_id}.html").write_text(f"plotly failed: {exc}\n", encoding="utf-8")
    lines = ["# 005 Full-scale filing-date-clean panel and ownership-network audit", "", "This is the one-time scale-up after the pilot mapping/network validation. Raw and derived vendor-level Parquet files remain local under ignored `data/` directories and are not bundled.", "", "## Key metrics", "", "| Metric | Value |", "|---|---:|"]
    for k in ["clean_13f_rows", "mapped_common_rows", "common_mapping_rate", "position_rows", "panel_rows", "unique_months", "unique_stocks", "unique_managers", "network_months", "total_graph_nonzeros", "panel_fwd_ret_1m_coverage", "panel_fwd_ret_3m_coverage", "median_owner_count", "median_network_degree", "median_network_weighted_degree"]:
        lines.append(f"| {k} | {metrics.get(k)} |")
    if manifest.get("problems"):
        lines += ["", "## Validation problems", ""] + [f"- {p}" for p in manifest["problems"]]
    (root / "docs" / "005_full_panel_network_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.update({"metrics_csv": str(tables / f"005_full_panel_metrics_{run_id}.csv"), "monthly_coverage_csv": str(tables / f"005_full_panel_monthly_coverage_{run_id}.csv"), "markdown_report": str(root / "docs" / "005_full_panel_network_audit.md")})
    (logs / f"005_full_panel_manifest_{run_id}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run(root: Path, run_id: str, start: str, end: str, chunk_months: int, n_jobs: int, network_jobs: int) -> int:
    contract_path = latest_file(root / "artifacts" / "schema", "002_schema_contract_*.json"); pilot_path = latest_file(root / "artifacts" / "logs", "004_pilot_panel_manifest_*.json"); contract = read_json(contract_path)
    raw_dir = root / "data" / "interim" / "005_full_core" / run_id / "raw"; clean_dir = root / "data" / "interim" / "005_full_core" / run_id / "standardized"; out_dir = root / "data" / "processed" / "005_full_panel" / run_id
    for d in [raw_dir, clean_dir, out_dir]: d.mkdir(parents=True, exist_ok=True)
    print(f"[005] contract={contract_path}"); print(f"[005] pilot_manifest={pilot_path}"); print(f"[005] date_range={start}..{end}")
    import wrds
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME") or os.environ.get("USER")); extract_stats = []
    try:
        for role in ["13f_holdings", "crsp_stock_names", "crsp_monthly_stock"]:
            extract_stats.append(extract_role(db, contract, role, raw_dir, start, end, chunk_months))
    finally:
        try: db.close()
        except Exception: pass
    std = standardize_all(raw_dir, clean_dir, start, end); panel = duckdb_build(clean_dir, out_dir, n_jobs, end); net = build_network(out_dir, panel, network_jobs)
    metrics: dict[str, Any] = {}; metrics.update(std); metrics.update(panel["counts"]); metrics.update(net["counts"]); metrics["network_months"] = net["network_months"]; metrics["total_graph_nonzeros"] = net["total_graph_nonzeros"]
    problems = []
    for k, minimum in MIN_COUNTS.items():
        if float(metrics.get(k, 0) or 0) < minimum: problems.append(f"{k} below threshold: {metrics.get(k)} < {minimum}")
    if float(metrics.get("common_mapping_rate", 0) or 0) < 0.02: problems.append(f"common_mapping_rate suspiciously low: {metrics.get('common_mapping_rate')}")
    if float(metrics.get("panel_fwd_ret_1m_coverage", 0) or 0) < 0.50: problems.append(f"1m forward-return coverage below 50%: {metrics.get('panel_fwd_ret_1m_coverage')}")
    manifest = {"run_id": run_id, "created_utc": utc_now(), "host": socket.gethostname(), "python": sys.version, "project_root": str(root), "start_date": start, "end_date": end, "chunk_months": chunk_months, "n_jobs": n_jobs, "network_jobs": network_jobs, "contract_json": str(contract_path), "pilot_manifest_json": str(pilot_path), "local_raw_dir": str(raw_dir), "local_standardized_dir": str(clean_dir), "local_processed_dir": str(out_dir), "extract_stats": extract_stats, "standardize_stats": std, "panel_info": panel, "network_info": {k:v for k,v in net.items() if k != "network_results_sample"}, "network_results_sample": net.get("network_results_sample", []), "metrics": metrics, "problems": problems, "status": "ok" if not problems else "needs_attention"}
    write_outputs(root, run_id, manifest)
    for k in ["clean_13f_rows", "mapped_common_rows", "position_rows", "panel_rows", "unique_months", "unique_stocks", "unique_managers", "network_months"]: print(f"[005] {k}={metrics.get(k)}")
    print(f"[005] status={manifest['status']}")
    for p in problems: print(f"[005][problem] {p}")
    return 0 if not problems else 20


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", required=True); p.add_argument("--run-id", required=True); p.add_argument("--start-date", default=os.environ.get("ONF_FULL_START_DATE", "2000-01-01")); p.add_argument("--end-date", default=os.environ.get("ONF_FULL_END_DATE", "2025-12-31")); p.add_argument("--chunk-months", type=int, default=int(os.environ.get("ONF_WRDS_CHUNK_MONTHS", "3"))); p.add_argument("--n-jobs", type=int, default=int(os.environ.get("ONF_N_JOBS", "32"))); p.add_argument("--network-jobs", type=int, default=int(os.environ.get("ONF_NETWORK_JOBS", "8")))
    a = p.parse_args()
    try: return run(Path(a.project_root).resolve(), a.run_id, a.start_date, a.end_date, a.chunk_months, a.n_jobs, a.network_jobs)
    except Exception as exc:
        root = Path(a.project_root).resolve(); logs = root / "artifacts" / "logs"; logs.mkdir(parents=True, exist_ok=True); fail = logs / f"005_full_panel_manifest_{a.run_id}_FAILED.json"; fail.write_text(json.dumps({"run_id": a.run_id, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "created_utc": utc_now()}, indent=2), encoding="utf-8"); print(f"[005] FAILED wrote {fail}"); traceback.print_exc(); return 99


if __name__ == "__main__":
    raise SystemExit(main())
