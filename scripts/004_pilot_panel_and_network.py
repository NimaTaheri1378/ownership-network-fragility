from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

HOLDING_MANAGER = ["mgrno", "managerid", "mgr_id", "manager", "cik"]
HOLDING_CUSIP = ["cusip", "cusip8", "ncusip"]
HOLDING_RDATE = ["rdate", "report_date", "reportdate", "periodofreport"]
HOLDING_FDATE = ["fdate", "filedate", "filingdate", "filing_date", "accepted", "acceptance_datetime"]
HOLDING_SHARES = ["shares", "sshprnamt", "share", "shares_held"]

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
MONTHLY_DATE = ["date", "mthcaldt", "caldt"]
MONTHLY_RET = ["ret", "mthret"]
MONTHLY_RETX = ["retx", "mthretx"]
MONTHLY_PRC = ["prc", "mthprc", "altprc"]
MONTHLY_SHROUT = ["shrout", "mthshrout"]
MONTHLY_VOL = ["vol", "mthvol"]

MIN_PANEL_ROWS = 50
MIN_MAPPED_COMMON = 50
MIN_STOCKS = 5
MIN_MANAGERS = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not candidates:
        raise SystemExit(f"No files found under {directory} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_dir(directory: Path) -> Path:
    candidates = [p for p in directory.glob("*") if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No directories found under {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_path(path_text: str | None, root: Path, fallback: Path) -> Path:
    if path_text:
        p = Path(path_text)
        if p.exists():
            return p.resolve()
        marker = "/github/Filing-Date-Clean Ownership Network Fragility"
        s = str(p)
        if marker in s:
            candidate = root / s.split(marker, 1)[1].lstrip("/")
            if candidate.exists():
                return candidate.resolve()
    return fallback.resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lower_map(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in columns:
        out.setdefault(str(c).lower(), str(c))
    return out


def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lut = lower_map([str(c) for c in df.columns])
    for a in aliases:
        if a.lower() in lut:
            return lut[a.lower()]
    return None


def require_col(df: pd.DataFrame, aliases: list[str], label: str) -> str:
    c = find_col(df, aliases)
    if c is None:
        raise ValueError(f"Missing required {label}. Tried {aliases}. Available {list(df.columns)}")
    return c


def parse_number(s: pd.Series) -> pd.Series:
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
    return pd.to_datetime(s, errors="coerce")


def cusip8(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.slice(0, 8).where(lambda x: x.str.len() >= 6)


def month_string(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce").dt.to_period("M").astype("string")


def safe_ident(x: str, label: str) -> str:
    if not IDENT_RE.match(x or ""):
        raise ValueError(f"Unsafe {label}: {x!r}")
    return x


def qident(x: str) -> str:
    safe_ident(x, "identifier")
    return '"' + x.replace('"', '""') + '"'


def table_sql(lib: str, table: str) -> str:
    return f"{qident(lib)}.{qident(table)}"


def rollback(db: Any) -> None:
    for attr in ["connection", "conn", "_connection"]:
        obj = getattr(db, attr, None)
        if obj is not None:
            try:
                obj.rollback()
                return
            except Exception:
                pass


def role_selected(contract: dict[str, Any], role: str) -> dict[str, Any] | None:
    for r in contract.get("roles", []):
        if r.get("role") == role and r.get("selected"):
            return r["selected"]
    return None


def select_contract_columns(selected: dict[str, Any], alias_groups: list[list[str]], extra: int = 12) -> list[str]:
    cols = [str(c) for c in selected.get("columns") or []]
    lut = lower_map(cols)
    chosen: list[str] = []
    def add(c: str | None) -> None:
        if c and c not in chosen:
            chosen.append(c)
    for group in alias_groups:
        for a in group:
            c = lut.get(a.lower())
            if c:
                add(c)
                break
    for c in cols[:extra]:
        add(c)
    return chosen[:64]


def yyyymmdd(date_text: str) -> int:
    return int(date_text.replace("-", ""))


def wrds_supplement(root: Path, contract: dict[str, Any], role: str, run_id: str, start: str, end: str, max_rows: int) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    alias_map = {
        "crsp_stock_names": [[*NAME_PERMNO], [*NAME_PERMCO], [*NAME_CUSIP], [*NAME_START], [*NAME_END], [*NAME_SHRCD], [*NAME_EXCHCD], [*NAME_TICKER], [*NAME_COMNAM]],
        "crsp_monthly_stock": [[*MONTHLY_PERMNO], [*MONTHLY_DATE], [*MONTHLY_RET], [*MONTHLY_RETX], [*MONTHLY_PRC], [*MONTHLY_SHROUT], [*MONTHLY_VOL]],
    }
    selected = role_selected(contract, role)
    info: dict[str, Any] = {"role": role, "attempted": True, "status": "not_started"}
    if not selected:
        info["status"] = "no_contract_role"
        return None, info
    cols = select_contract_columns(selected, alias_map[role])
    lib, tbl = str(selected["library"]), str(selected["table"])
    lib, tbl = safe_ident(lib, "library"), safe_ident(tbl, "table")
    info.update({"source": f"{lib}.{tbl}", "columns": cols, "max_rows": max_rows})

    lut = lower_map([str(c) for c in selected.get("columns") or []])
    if role == "crsp_stock_names":
        start_col = next((lut[a] for a in [x.lower() for x in NAME_START] if a in lut), None)
        end_col = next((lut[a] for a in [x.lower() for x in NAME_END] if a in lut), None)
        if start_col and end_col:
            predicates = [
                f"({qident(start_col)} <= DATE '{end}' OR {qident(start_col)} IS NULL) AND ({qident(end_col)} >= DATE '{start}' OR {qident(end_col)} IS NULL)",
                f"({qident(start_col)} <= {yyyymmdd(end)} OR {qident(start_col)} IS NULL) AND ({qident(end_col)} >= {yyyymmdd(start)} OR {qident(end_col)} IS NULL)",
            ]
        else:
            predicates = ["1=1"]
    else:
        date_col = next((lut[a] for a in [x.lower() for x in MONTHLY_DATE] if a in lut), None)
        if date_col:
            predicates = [
                f"{qident(date_col)} >= DATE '{start}' AND {qident(date_col)} <= DATE '{end}'",
                f"{qident(date_col)} >= {yyyymmdd(start)} AND {qident(date_col)} <= {yyyymmdd(end)}",
            ]
        else:
            predicates = ["1=1"]

    select_cols = ", ".join(qident(c) for c in cols)
    out_dir = root / "data" / "interim" / "004_crosswalk" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wrds
        db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME") or os.environ.get("USER"))
    except Exception as exc:
        info["status"] = "connection_failed"
        info["error"] = repr(exc)
        return None, info

    try:
        last_error = ""
        for i, pred in enumerate(predicates):
            query = f"SELECT {select_cols} FROM {table_sql(lib, tbl)} WHERE {pred} LIMIT {int(max_rows)}"
            try:
                df = db.raw_sql(query)
                out_path = out_dir / f"{role}_supplement.parquet"
                df.to_parquet(out_path, index=False)
                info.update({"status": "ok", "rows": int(len(df)), "query_variant": i, "local_path": str(out_path)})
                return df, info
            except Exception as exc:
                last_error = repr(exc)
                rollback(db)
        info.update({"status": "failed_queries", "last_error": last_error})
        return None, info
    finally:
        try:
            db.close()
        except Exception:
            pass


def read_role_frame(manifest: dict[str, Any], root: Path, pilot_dir: Path, role: str) -> pd.DataFrame:
    for stat in manifest.get("tables", []):
        if stat.get("role") == role and stat.get("local_path"):
            p = resolve_path(str(stat["local_path"]), root, pilot_dir / f"{role}.parquet")
            if p.exists():
                return pd.read_parquet(p)
    p = pilot_dir / f"{role}.parquet"
    if not p.exists():
        raise SystemExit(f"Missing pilot parquet for role={role}: {p}")
    return pd.read_parquet(p)


def standardize_holdings(raw: pd.DataFrame) -> pd.DataFrame:
    mgr = require_col(raw, HOLDING_MANAGER, "13F manager id")
    cus = require_col(raw, HOLDING_CUSIP, "13F CUSIP")
    rdt = find_col(raw, HOLDING_RDATE)
    fdt = find_col(raw, HOLDING_FDATE)
    shr = find_col(raw, HOLDING_SHARES)
    out = pd.DataFrame({"manager_id": raw[mgr].astype("string"), "cusip8": cusip8(raw[cus])})
    out["report_date"] = parse_date(raw[rdt]) if rdt else pd.NaT
    out["filing_date"] = parse_date(raw[fdt]) if fdt else pd.NaT
    out["available_date"] = out["filing_date"].fillna(out["report_date"] + pd.Timedelta(days=45))
    out["mapping_date"] = out["report_date"].fillna(out["available_date"])
    out["filing_lag_days"] = (out["available_date"] - out["report_date"]).dt.days
    out["month"] = month_string(out["available_date"])
    out["report_quarter"] = out["report_date"].dt.to_period("Q").astype("string")
    out["shares"] = parse_number(raw[shr]) if shr else 1.0
    out = out.dropna(subset=["manager_id", "cusip8", "available_date", "month"])
    out = out[out["shares"].fillna(0) > 0].copy()
    group_cols = ["manager_id", "cusip8", "report_date", "available_date", "mapping_date", "filing_lag_days", "month", "report_quarter"]
    out = out.groupby(group_cols, dropna=False, as_index=False).agg(shares=("shares", "sum"))
    out["holding_id"] = np.arange(len(out), dtype=np.int64)
    return out


def standardize_names(raw: pd.DataFrame) -> pd.DataFrame:
    permno = require_col(raw, NAME_PERMNO, "CRSP PERMNO")
    cus = require_col(raw, NAME_CUSIP, "CRSP CUSIP/NCUSIP")
    permco, start, end = find_col(raw, NAME_PERMCO), find_col(raw, NAME_START), find_col(raw, NAME_END)
    shrcd, exchcd, ticker, comnam = find_col(raw, NAME_SHRCD), find_col(raw, NAME_EXCHCD), find_col(raw, NAME_TICKER), find_col(raw, NAME_COMNAM)
    out = pd.DataFrame({"permno": pd.to_numeric(raw[permno], errors="coerce").astype("Int64"), "cusip8": cusip8(raw[cus])})
    out["permco"] = pd.to_numeric(raw[permco], errors="coerce").astype("Int64") if permco else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["name_start"] = parse_date(raw[start]) if start else pd.Timestamp("1900-01-01")
    out["name_end"] = parse_date(raw[end]) if end else pd.Timestamp("2099-12-31")
    out["name_start"] = out["name_start"].fillna(pd.Timestamp("1900-01-01"))
    out["name_end"] = out["name_end"].fillna(pd.Timestamp("2099-12-31"))
    out["shrcd"] = pd.to_numeric(raw[shrcd], errors="coerce").astype("Int64") if shrcd else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["exchcd"] = pd.to_numeric(raw[exchcd], errors="coerce").astype("Int64") if exchcd else pd.Series(pd.NA, index=raw.index, dtype="Int64")
    out["ticker"] = raw[ticker].astype("string") if ticker else pd.Series(pd.NA, index=raw.index, dtype="string")
    out["comnam"] = raw[comnam].astype("string") if comnam else pd.Series(pd.NA, index=raw.index, dtype="string")
    out = out.dropna(subset=["permno", "cusip8"]).drop_duplicates().copy()
    out["is_common_stock"] = out["shrcd"].isin([10, 11]) if out["shrcd"].notna().any() else True
    return out


def standardize_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    permno = require_col(raw, MONTHLY_PERMNO, "CRSP monthly PERMNO")
    date = require_col(raw, MONTHLY_DATE, "CRSP monthly date")
    ret, retx, prc, shrout, vol = find_col(raw, MONTHLY_RET), find_col(raw, MONTHLY_RETX), find_col(raw, MONTHLY_PRC), find_col(raw, MONTHLY_SHROUT), find_col(raw, MONTHLY_VOL)
    out = pd.DataFrame({"permno": pd.to_numeric(raw[permno], errors="coerce").astype("Int64"), "date": parse_date(raw[date])})
    out["month"] = month_string(out["date"])
    out["ret"] = parse_number(raw[ret]) if ret else np.nan
    out["retx"] = parse_number(raw[retx]) if retx else np.nan
    out["prc"] = parse_number(raw[prc]) if prc else np.nan
    out["shrout"] = parse_number(raw[shrout]) if shrout else np.nan
    out["vol"] = parse_number(raw[vol]) if vol else np.nan
    out["mktcap_proxy"] = out["prc"].abs() * out["shrout"]
    out = out.dropna(subset=["permno", "date", "month"]).sort_values(["permno", "date"])
    out = out.drop_duplicates(["permno", "month"], keep="last").copy()
    g = out.groupby("permno", observed=True)["ret"]
    out["fwd_ret_1m"] = g.shift(-1)
    out["fwd_ret_2m_leg"] = g.shift(-2)
    out["fwd_ret_3m_leg"] = g.shift(-3)
    out["fwd_ret_3m"] = (1 + out["fwd_ret_1m"]) * (1 + out["fwd_ret_2m_leg"]) * (1 + out["fwd_ret_3m_leg"]) - 1
    return out.drop(columns=["fwd_ret_2m_leg", "fwd_ret_3m_leg"])


def map_holdings(holdings: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    m = holdings.merge(names, on="cusip8", how="left", suffixes=("", "_name"))
    m["date_window_match"] = (m["mapping_date"] >= m["name_start"]) & (m["mapping_date"] <= m["name_end"])
    m = m[m["permno"].notna() & m["date_window_match"]].copy()
    m["common_rank"] = m["is_common_stock"].fillna(False).astype(int)
    m = m.sort_values(["holding_id", "common_rank", "name_start", "name_end"], ascending=[True, False, False, True])
    return m.drop_duplicates("holding_id", keep="first")


def build_positions(mapped_common: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    if mapped_common.empty:
        return pd.DataFrame()
    val = monthly[["permno", "month", "prc", "shrout", "mktcap_proxy", "ret", "fwd_ret_1m", "fwd_ret_3m"]].copy()
    base = mapped_common.copy()
    base["permno"] = pd.to_numeric(base["permno"], errors="coerce").astype("Int64")
    x = base.merge(val, on=["permno", "month"], how="left")
    x["position_value_proxy"] = x["shares"] * x["prc"].abs()
    miss = x["position_value_proxy"].isna() | (x["position_value_proxy"] <= 0)
    x.loc[miss, "position_value_proxy"] = x.loc[miss, "shares"]
    g = x.groupby(["month", "manager_id", "permno"], dropna=False, as_index=False).agg(
        shares=("shares", "sum"), position_value_proxy=("position_value_proxy", "sum"),
        report_date=("report_date", "max"), available_date=("available_date", "max"),
        prc=("prc", "last"), mktcap_proxy=("mktcap_proxy", "last"), ret=("ret", "last"),
        fwd_ret_1m=("fwd_ret_1m", "last"), fwd_ret_3m=("fwd_ret_3m", "last"),
    )
    g = g[g["position_value_proxy"].fillna(0) > 0].copy()
    total = g.groupby(["month", "manager_id"], observed=True)["position_value_proxy"].transform("sum")
    g["manager_total_value_proxy"] = total
    g["portfolio_weight"] = g["position_value_proxy"] / total.replace({0: np.nan})
    return g.replace([np.inf, -np.inf], np.nan)


def add_features(pos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mgr = pos.groupby(["month", "manager_id"], observed=True).agg(
        manager_breadth=("permno", "nunique"),
        manager_total_value_proxy=("position_value_proxy", "sum"),
        manager_concentration=("portfolio_weight", lambda s: float(np.nansum(np.square(s)))),
    ).reset_index()
    e = pos.merge(mgr, on=["month", "manager_id"], how="left", suffixes=("", "_mgr"))
    stock_total = e.groupby(["month", "permno"], observed=True)["position_value_proxy"].transform("sum")
    e["stock_owner_share"] = e["position_value_proxy"] / stock_total.replace({0: np.nan})
    e = e.sort_values(["manager_id", "permno", "month"])
    e["lag_position_value_proxy"] = e.groupby(["manager_id", "permno"], observed=True)["position_value_proxy"].shift(1)
    e["sell_amount_proxy"] = (-(e["position_value_proxy"] - e["lag_position_value_proxy"])).clip(lower=0).fillna(0)
    sell = e.groupby(["month", "permno"], observed=True).agg(stock_sell_amount_proxy=("sell_amount_proxy", "sum"), stock_total_value_proxy=("position_value_proxy", "sum")).reset_index()
    sell["stock_sell_pressure"] = sell["stock_sell_amount_proxy"] / sell["stock_total_value_proxy"].replace({0: np.nan})
    stock = e.groupby(["month", "permno"], observed=True).agg(
        owner_count=("manager_id", "nunique"), total_position_value_proxy=("position_value_proxy", "sum"),
        owner_hhi=("stock_owner_share", lambda s: float(np.nansum(np.square(s)))), top_owner_share=("stock_owner_share", "max"),
        fragility_proxy=("manager_concentration", "mean"), avg_manager_breadth=("manager_breadth", "mean"),
        fwd_ret_1m=("fwd_ret_1m", "last"), fwd_ret_3m=("fwd_ret_3m", "last"), ret=("ret", "last"), prc=("prc", "last"), mktcap_proxy=("mktcap_proxy", "last"),
    ).reset_index()
    stock = stock.merge(sell[["month", "permno", "stock_sell_pressure"]], on=["month", "permno"], how="left")
    return e, mgr, stock


def build_network(pos: pd.DataFrame, stock: pd.DataFrame, max_edges: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows, edges = [], []
    stats: dict[str, Any] = {"months": [], "total_graph_nonzeros": 0, "total_edges_sampled": 0}
    sell = stock.set_index(["month", "permno"])["stock_sell_pressure"].to_dict() if len(stock) else {}
    for month, sub in pos.dropna(subset=["portfolio_weight", "permno", "manager_id"]).groupby("month", sort=True, observed=True):
        sub = sub[sub["portfolio_weight"].fillna(0) > 0].copy()
        if sub.empty:
            continue
        managers = pd.Index(pd.unique(sub["manager_id"].astype("string")))
        stocks = pd.Index(pd.unique(sub["permno"].astype("Int64")))
        mc = pd.Categorical(sub["manager_id"].astype("string"), categories=managers).codes
        sc = pd.Categorical(sub["permno"].astype("Int64"), categories=stocks).codes
        w = pd.to_numeric(sub["portfolio_weight"], errors="coerce").fillna(0).to_numpy(float)
        keep = (mc >= 0) & (sc >= 0) & np.isfinite(w) & (w > 0)
        if keep.sum() == 0:
            continue
        W = sparse.csr_matrix((w[keep], (mc[keep], sc[keep])), shape=(len(managers), len(stocks)))
        G = (W.T @ W).tocsr()
        G.setdiag(0)
        G.eliminate_zeros()
        degree_w = np.asarray(G.sum(axis=1)).ravel()
        degree = np.diff(G.indptr)
        stock_list = [int(x) for x in stocks]
        sell_vec = np.array([float(sell.get((month, p), 0.0) or 0.0) for p in stock_list], dtype=float)
        if G.nnz:
            peer_num = G @ sell_vec
            peer_sell = np.divide(peer_num, degree_w, out=np.full_like(peer_num, np.nan), where=degree_w > 0)
        else:
            peer_sell = np.full(len(stocks), np.nan)
        rows.append(pd.DataFrame({"month": str(month), "permno": stock_list, "network_degree": degree.astype(int), "network_weighted_degree": degree_w, "network_peer_sell_pressure": peer_sell}))
        stats["months"].append({"month": str(month), "n_managers": int(len(managers)), "n_stocks": int(len(stocks)), "graph_nonzeros": int(G.nnz)})
        stats["total_graph_nonzeros"] += int(G.nnz)
        if G.nnz:
            coo = G.tocoo(); vals = coo.data
            idx = np.argpartition(vals, -min(max_edges, len(vals)))[-min(max_edges, len(vals)):]
            ed = pd.DataFrame({"month": str(month), "permno_i": [stock_list[i] for i in coo.row[idx]], "permno_j": [stock_list[j] for j in coo.col[idx]], "overlap_weight": vals[idx]})
            ed = ed[ed["permno_i"] < ed["permno_j"]]
            edges.append(ed)
            stats["total_edges_sampled"] += int(len(ed))
    net = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["month", "permno", "network_degree", "network_weighted_degree", "network_peer_sell_pressure"])
    edf = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame(columns=["month", "permno_i", "permno_j", "overlap_weight"])
    return net, edf, stats


def safe_float(x: Any) -> float | None:
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        for k, v in metrics.items():
            w.writerow({"metric": k, "value": v})


def write_md(path: Path, manifest: dict[str, Any]) -> None:
    m = manifest["metrics"]
    rows = [
        ("Raw 13F rows", m.get("raw_13f_rows")), ("Clean 13F rows", m.get("clean_13f_rows")),
        ("Mapped common rows", m.get("mapped_common_rows")), ("Panel rows", m.get("panel_rows")),
        ("Unique stocks", m.get("unique_stocks")), ("Unique managers", m.get("unique_managers")),
        ("Unique months", m.get("unique_months")), ("1m forward-return coverage", m.get("panel_fwd_ret_1m_coverage")),
        ("Total graph nonzeros", m.get("total_graph_nonzeros")),
    ]
    lines = ["# 004 Pilot panel and ownership-network audit", "", "This step converts the Step 003 local pilot into a filing-date-aware 13F-to-CRSP mapping, a pilot stock-month ownership panel, and sparse common-ownership network features. Local derived Parquet files stay in ignored `data/` folders and are not included in the share bundle.", "", "## Run metadata", "", f"- run_id: `{manifest['run_id']}`", f"- created_utc: `{manifest['created_utc']}`", f"- host: `{manifest['host']}`", f"- status: `{manifest['status']}`", f"- pilot_dir: `{manifest['pilot_dir']}`", f"- local_processed_dir: `{manifest['local_processed_dir']}`", "", "## Key metrics", "", "| Metric | Value |", "|---|---:|"]
    for k, v in rows:
        lines.append(f"| {k} | `{v}` |")
    lines += ["", "## Data policy", "", "The uploaded bundle contains only logs, manifests, aggregate tables, documentation, and derived figures. It excludes WRDS/vendor row-level Parquet files."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_figures(root: Path, run_id: str, metrics: dict[str, Any], monthly_summary: pd.DataFrame, panel: pd.DataFrame) -> dict[str, str]:
    fig_dir = root / "artifacts" / "figures_static"; html_dir = root / "artifacts" / "figures_interactive"
    fig_dir.mkdir(parents=True, exist_ok=True); html_dir.mkdir(parents=True, exist_ok=True)
    funnel_path = fig_dir / f"004_mapping_funnel_{run_id}.png"
    coverage_path = fig_dir / f"004_monthly_coverage_{run_id}.png"
    degree_path = fig_dir / f"004_network_weighted_degree_{run_id}.png"
    html_path = html_dir / f"004_pilot_panel_network_{run_id}.html"
    labels = ["Raw 13F", "Clean", "Mapped common", "Panel"]
    values = [metrics.get("raw_13f_rows", 0), metrics.get("clean_13f_rows", 0), metrics.get("mapped_common_rows", 0), metrics.get("panel_rows", 0)]
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
        fig, ax = plt.subplots(figsize=(9, 4.8)); y = np.arange(len(labels)); ax.barh(y, values); ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel("Rows"); ax.set_title("Step 004 pilot mapping funnel")
        for i, v in enumerate(values): ax.text(v, i, f" {int(v):,}", va="center")
        fig.tight_layout(); fig.savefig(funnel_path); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5.2))
        if len(monthly_summary):
            x = pd.to_datetime(monthly_summary["month"].astype(str) + "-01", errors="coerce")
            ax.plot(x, monthly_summary["positions"], marker="o", label="positions"); ax.plot(x, monthly_summary["stocks"], marker="o", label="stocks"); ax.plot(x, monthly_summary["managers"], marker="o", label="managers"); ax.legend(frameon=False)
        ax.set_title("Step 004 pilot monthly coverage"); ax.set_xlabel("Availability month"); ax.set_ylabel("Count"); fig.tight_layout(); fig.savefig(coverage_path); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4.8)); deg = pd.to_numeric(panel.get("network_weighted_degree", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(deg): ax.hist(deg, bins=min(40, max(5, int(np.sqrt(len(deg))))))
        ax.set_title("Step 004 network weighted-degree distribution"); ax.set_xlabel("Weighted degree"); ax.set_ylabel("Stock-month count"); fig.tight_layout(); fig.savefig(degree_path); plt.close(fig)
    except Exception as exc:
        for p in [funnel_path, coverage_path, degree_path]:
            if not p.exists(): p.write_text(f"figure failed: {exc}\n", encoding="utf-8")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "histogram"}, {"type": "table"}]], subplot_titles=("Mapping funnel", "Monthly coverage", "Weighted degree", "Key metrics"))
        fig.add_trace(go.Bar(x=values, y=labels, orientation="h"), row=1, col=1)
        if len(monthly_summary):
            fig.add_trace(go.Scatter(x=monthly_summary["month"], y=monthly_summary["positions"], mode="lines+markers", name="positions"), row=1, col=2)
            fig.add_trace(go.Scatter(x=monthly_summary["month"], y=monthly_summary["stocks"], mode="lines+markers", name="stocks"), row=1, col=2)
            fig.add_trace(go.Scatter(x=monthly_summary["month"], y=monthly_summary["managers"], mode="lines+markers", name="managers"), row=1, col=2)
        deg = pd.to_numeric(panel.get("network_weighted_degree", pd.Series(dtype=float)), errors="coerce").dropna(); fig.add_trace(go.Histogram(x=deg, nbinsx=35), row=2, col=1)
        keys = list(metrics.keys())[:18]
        fig.add_trace(go.Table(header={"values": ["Metric", "Value"]}, cells={"values": [keys, [metrics[k] for k in keys]]}), row=2, col=2)
        fig.update_layout(title="Step 004 pilot panel and ownership-network audit", template="plotly_white", height=860, margin={"l": 70, "r": 30, "t": 90, "b": 70})
        fig.write_html(html_path, include_plotlyjs="cdn")
    except Exception as exc:
        html_path.write_text(f"plotly dashboard failed: {exc}\n", encoding="utf-8")
    return {"mapping_funnel_png": str(funnel_path), "monthly_coverage_png": str(coverage_path), "network_weighted_degree_png": str(degree_path), "interactive_html": str(html_path)}


def build_once(root: Path, run_id: str, pilot_start: str, pilot_end: str, max_stocknames: int, max_monthly: int) -> int:
    m003_path = latest_file(root / "artifacts" / "logs", "003_pilot_extract_manifest_*.json")
    m003 = read_json(m003_path)
    pilot_dir = resolve_path(m003.get("local_pilot_dir"), root, latest_dir(root / "data" / "interim" / "003_pilot"))
    contract_path = latest_file(root / "artifacts" / "schema", "002_schema_contract_*.json")
    contract = read_json(contract_path)
    print(f"[004] step003_manifest={m003_path}"); print(f"[004] pilot_dir={pilot_dir}"); print(f"[004] contract={contract_path}")

    raw_h = read_role_frame(m003, root, pilot_dir, "13f_holdings")
    raw_n = read_role_frame(m003, root, pilot_dir, "crsp_stock_names")
    raw_m = read_role_frame(m003, root, pilot_dir, "crsp_monthly_stock")
    holdings = standardize_holdings(raw_h)
    names = standardize_names(raw_n)
    monthly = standardize_monthly(raw_m)

    mapped = map_holdings(holdings, names)
    initial_common = int(mapped["is_common_stock"].fillna(False).sum()) if len(mapped) else 0
    supplement_info = {"stocknames": {"attempted": False, "status": "not_needed"}, "monthly": {"attempted": False, "status": "not_needed"}}
    print(f"[004] initial mapped rows={len(mapped)} common={initial_common} rate={len(mapped)/max(len(holdings),1):.4f}")
    if initial_common < MIN_MAPPED_COMMON or len(mapped) / max(len(holdings), 1) < 0.02:
        print("[004] mapping thin; pulling local ignored CRSP stocknames active-window supplement")
        sup, info = wrds_supplement(root, contract, "crsp_stock_names", run_id, pilot_start, pilot_end, max_stocknames)
        supplement_info["stocknames"] = info
        if sup is not None and len(sup):
            names = standardize_names(sup)
            mapped = map_holdings(holdings, names)
            print(f"[004] supplemental mapped rows={len(mapped)} common={int(mapped['is_common_stock'].fillna(False).sum()) if len(mapped) else 0}")

    mapped_common = mapped[mapped["is_common_stock"].fillna(False)].copy() if len(mapped) else pd.DataFrame()
    positions = build_positions(mapped_common, monthly)
    if len(positions) < MIN_PANEL_ROWS:
        print("[004] position panel thin; pulling local ignored CRSP monthly supplement")
        supm, infom = wrds_supplement(root, contract, "crsp_monthly_stock", run_id, pilot_start, "2021-03-31", max_monthly)
        supplement_info["monthly"] = infom
        if supm is not None and len(supm):
            monthly = standardize_monthly(supm)
            positions = build_positions(mapped_common, monthly)
            print(f"[004] supplemental monthly positions={len(positions)}")

    if len(positions):
        pos_rows, manager_features, stock_features = add_features(positions)
        network_features, edge_sample, network_stats = build_network(pos_rows, stock_features)
        panel = stock_features.merge(network_features, on=["month", "permno"], how="left")
        monthly_summary = positions.groupby("month", observed=True).agg(positions=("permno", "size"), stocks=("permno", "nunique"), managers=("manager_id", "nunique")).reset_index().sort_values("month")
    else:
        pos_rows = manager_features = stock_features = network_features = edge_sample = panel = pd.DataFrame()
        monthly_summary = pd.DataFrame(columns=["month", "positions", "stocks", "managers"])
        network_stats = {"months": [], "total_graph_nonzeros": 0, "total_edges_sampled": 0}

    out_dir = root / "data" / "processed" / "004_pilot_panel" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    holdings.to_parquet(out_dir / "clean_13f_holdings.parquet", index=False)
    mapped_common.to_parquet(out_dir / "mapped_common_holdings.parquet", index=False)
    positions.to_parquet(out_dir / "manager_stock_positions.parquet", index=False)
    panel.to_parquet(out_dir / "stock_month_network_panel.parquet", index=False)
    edge_sample.to_parquet(out_dir / "network_edge_sample.parquet", index=False)
    monthly_summary.to_parquet(out_dir / "monthly_coverage.parquet", index=False)

    metrics: dict[str, Any] = {
        "raw_13f_rows": int(len(raw_h)), "clean_13f_rows": int(len(holdings)), "raw_stocknames_rows": int(len(raw_n)), "standardized_stocknames_rows": int(len(names)), "crsp_monthly_rows": int(len(monthly)),
        "mapped_rows": int(len(mapped)), "mapped_common_rows": int(len(mapped_common)), "mapping_rate": round(float(len(mapped) / max(len(holdings), 1)), 6), "common_mapping_rate": round(float(len(mapped_common) / max(len(holdings), 1)), 6),
        "position_rows": int(len(positions)), "panel_rows": int(len(panel)), "unique_managers": int(positions["manager_id"].nunique()) if len(positions) else 0, "unique_stocks": int(positions["permno"].nunique()) if len(positions) else 0, "unique_months": int(positions["month"].nunique()) if len(positions) else 0,
        "panel_fwd_ret_1m_coverage": round(float(panel["fwd_ret_1m"].notna().mean()), 6) if len(panel) and "fwd_ret_1m" in panel else 0.0, "panel_fwd_ret_3m_coverage": round(float(panel["fwd_ret_3m"].notna().mean()), 6) if len(panel) and "fwd_ret_3m" in panel else 0.0,
        "median_owner_count": safe_float(panel["owner_count"].median()) if len(panel) and "owner_count" in panel else None, "median_network_degree": safe_float(panel["network_degree"].median()) if len(panel) and "network_degree" in panel else None, "median_network_weighted_degree": safe_float(panel["network_weighted_degree"].median()) if len(panel) and "network_weighted_degree" in panel else None,
        "total_graph_nonzeros": int(network_stats.get("total_graph_nonzeros", 0)), "total_edges_sampled": int(network_stats.get("total_edges_sampled", 0)),
    }
    lag = holdings["filing_lag_days"].dropna()
    if len(lag):
        metrics.update({"filing_lag_median_days": round(float(lag.median()), 4), "filing_lag_p05_days": round(float(lag.quantile(.05)), 4), "filing_lag_p95_days": round(float(lag.quantile(.95)), 4), "filing_lag_negative_share": round(float((lag < 0).mean()), 6)})

    problems = []
    if metrics["mapped_common_rows"] < MIN_MAPPED_COMMON: problems.append(f"mapped_common_rows below threshold: {metrics['mapped_common_rows']} < {MIN_MAPPED_COMMON}")
    if metrics["panel_rows"] < MIN_PANEL_ROWS: problems.append(f"panel_rows below threshold: {metrics['panel_rows']} < {MIN_PANEL_ROWS}")
    if metrics["unique_stocks"] < MIN_STOCKS: problems.append(f"unique_stocks below threshold: {metrics['unique_stocks']} < {MIN_STOCKS}")
    if metrics["unique_managers"] < MIN_MANAGERS: problems.append(f"unique_managers below threshold: {metrics['unique_managers']} < {MIN_MANAGERS}")
    if metrics.get("filing_lag_negative_share", 0) > 0.01: problems.append(f"negative filing lag share too high: {metrics.get('filing_lag_negative_share')}")

    figures = write_figures(root, run_id, metrics, monthly_summary, panel)
    manifest = {"run_id": run_id, "created_utc": utc_now(), "host": socket.gethostname(), "python": sys.version, "project_root": str(root), "step003_manifest": str(m003_path), "contract_json": str(contract_path), "pilot_dir": str(pilot_dir), "local_processed_dir": str(out_dir), "supplements": supplement_info, "metrics": metrics, "network_stats": network_stats, "figures": figures, "problems": problems, "status": "ok" if not problems else "needs_attention"}

    logs_dir = root / "artifacts" / "logs"; tables_dir = root / "artifacts" / "tables"
    logs_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = logs_dir / f"004_pilot_panel_manifest_{run_id}.json"
    quality_csv = tables_dir / f"004_pilot_panel_quality_{run_id}.csv"
    monthly_csv = tables_dir / f"004_pilot_panel_monthly_coverage_{run_id}.csv"
    md_path = root / "docs" / "004_pilot_panel_network_audit.md"
    write_metrics(quality_csv, metrics); monthly_summary.to_csv(monthly_csv, index=False)
    manifest.update({"quality_csv": str(quality_csv), "monthly_coverage_csv": str(monthly_csv), "markdown_report": str(md_path)})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_md(md_path, manifest)

    print(f"[004] mapped_common_rows={metrics['mapped_common_rows']}"); print(f"[004] panel_rows={metrics['panel_rows']}"); print(f"[004] unique_stocks={metrics['unique_stocks']}"); print(f"[004] unique_managers={metrics['unique_managers']}")
    print(f"[004] wrote {manifest_path}"); print(f"[004] wrote {quality_csv}"); print(f"[004] wrote {md_path}"); print(f"[004] status={manifest['status']}")
    if problems:
        print("[004] problems:")
        for p in problems: print(f"  - {p}")
        return 12
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-start", default="2019-01-01")
    parser.add_argument("--pilot-end", default="2020-12-31")
    parser.add_argument("--max-stocknames-supplement-rows", type=int, default=250000)
    parser.add_argument("--max-monthly-supplement-rows", type=int, default=400000)
    args = parser.parse_args()
    try:
        return build_once(Path(args.project_root).resolve(), args.run_id, args.pilot_start, args.pilot_end, args.max_stocknames_supplement_rows, args.max_monthly_supplement_rows)
    except Exception as exc:
        root = Path(args.project_root).resolve()
        fail = {"run_id": args.run_id, "created_utc": utc_now(), "status": "failed_exception", "error": repr(exc), "traceback": traceback.format_exc(limit=30)}
        path = root / "artifacts" / "logs" / f"004_pilot_panel_manifest_{args.run_id}_FAILED.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fail, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[004] FAILED wrote {path}")
        print(traceback.format_exc())
        return 21


if __name__ == "__main__":
    raise SystemExit(main())
