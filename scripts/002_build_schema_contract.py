from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_file(directory: Path, pattern: str) -> Path:
    candidates = [p for p in directory.glob(pattern) if "FAILED" not in p.name]
    if not candidates:
        raise SystemExit(f"No files found under {directory} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def norm_name(value: Any) -> str:
    return str(value or "").strip().lower()


def column_name(row: dict[str, Any]) -> str:
    for key in ["column_name", "name", "Column", "column", "field", "attname"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def data_type(row: dict[str, Any]) -> str:
    for key in ["data_type", "type", "Data Type", "format", "udt_name"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def ordinal(row: dict[str, Any], fallback: int) -> int:
    for key in ["ordinal_position", "position", "Column #", "attnum"]:
        value = row.get(key)
        try:
            return int(value)
        except Exception:
            pass
    return fallback


def normalize_columns(table_record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = table_record.get("columns") or table_record.get("description_records") or []
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            row = {"raw": str(row)}
        name = column_name(row)
        if not name:
            continue
        out.append(
            {
                "name": name,
                "name_lower": name.lower(),
                "data_type": data_type(row),
                "ordinal_position": ordinal(row, i),
                "raw": row,
            }
        )
    out.sort(key=lambda r: (r.get("ordinal_position", 10**9), r["name_lower"]))
    return out


def load_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise SystemExit(f"Schema discovery file is not ok: {path}")
    return payload


@dataclass(frozen=True)
class RoleSpec:
    role: str
    description: str
    priority_tables: tuple[tuple[str, str], ...]
    library_regex: str
    table_regex: str
    required_any_groups: tuple[tuple[str, ...], ...]
    preferred_columns: tuple[str, ...] = ()
    optional: bool = False
    later_stage: bool = False


ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec(
        role="13f_holdings",
        description="Thomson/Refinitiv institutional Form 13F holdings rows for manager-security-quarter disclosure records.",
        priority_tables=(("tr_13f", "s34"), ("tfn", "s34")),
        library_regex=r"^(tr_13f|tfn)$",
        table_regex=r"^s34$",
        required_any_groups=(
            ("mgrno", "managerid", "mgr_id"),
            ("rdate", "report_date", "reportdate"),
            ("fdate", "filedate", "filingdate", "filing_date"),
            ("cusip", "cusip8", "ncusip"),
            ("shares", "sshprnamt", "value", "market_value"),
        ),
        preferred_columns=("mgrno", "rdate", "fdate", "cusip", "shares", "sole", "shared", "no", "type"),
    ),
    RoleSpec(
        role="13f_manager_names",
        description="13F manager names and manager identifiers used to label and audit owner histories.",
        priority_tables=(("tr_13f", "s34names"), ("tfn", "s34names")),
        library_regex=r"^(tr_13f|tfn)$",
        table_regex=r"^s34names$",
        required_any_groups=(
            ("mgrno", "managerid", "mgr_id"),
            ("mgrname", "manager_name", "name"),
        ),
        preferred_columns=("mgrno", "mgrname", "fdate", "rdate"),
    ),
    RoleSpec(
        role="13f_security_type",
        description="13F security classification/type table for common-stock filters and derivative exclusions.",
        priority_tables=(("tr_13f", "s34type1"), ("tfn", "s34type1"), ("tr_13f", "s34type3"), ("tfn", "s34type3")),
        library_regex=r"^(tr_13f|tfn)$",
        table_regex=r"^s34type[0-9]+$",
        required_any_groups=(("type", "typecode", "stkcd", "code"),),
        preferred_columns=("type", "typecode", "description", "descrip", "security_type"),
        optional=True,
    ),
    RoleSpec(
        role="crsp_monthly_stock",
        description="CRSP monthly stock returns, prices, shares, and implementation variables.",
        priority_tables=(("crsp_a_stock", "msf"), ("crsp_a_stock", "msf_v2"), ("crsp", "msf"), ("crsp", "msf_v2")),
        library_regex=r"^(crsp_a_stock|crsp)$",
        table_regex=r"^(msf|msf_v2|wrds_msfv2_query)$",
        required_any_groups=(
            ("permno",),
            ("date", "mthcaldt", "caldt"),
            ("ret", "mthret"),
            ("prc", "mthprc", "altprc"),
            ("shrout", "mthshrout"),
        ),
        preferred_columns=("permno", "permco", "date", "mthcaldt", "ret", "retx", "mthret", "mthretx", "prc", "mthprc", "shrout", "vol"),
    ),
    RoleSpec(
        role="crsp_daily_stock",
        description="CRSP daily stock returns for label construction, event windows, volatility, and liquidity controls.",
        priority_tables=(("crsp_a_stock", "dsf"), ("crsp_a_stock", "dsf_v2"), ("crsp", "dsf"), ("crsp", "dsf_v2")),
        library_regex=r"^(crsp_a_stock|crsp)$",
        table_regex=r"^(dsf|dsf_v2|wrds_dsfv2_query)$",
        required_any_groups=(
            ("permno",),
            ("date", "dlycaldt", "caldt"),
            ("ret", "dlyret"),
            ("prc", "dlyprc", "askhi", "bidlo"),
        ),
        preferred_columns=("permno", "permco", "date", "dlycaldt", "ret", "retx", "dlyret", "dlyretx", "prc", "dlyprc", "vol", "shrout"),
    ),
    RoleSpec(
        role="crsp_stock_names",
        description="CRSP security names, CUSIP histories, exchange codes, and share codes for common-stock universe filters.",
        priority_tables=(("crsp_a_stock", "stocknames_v2"), ("crsp_a_stock", "stocknames"), ("crsp_a_stock", "msenames"), ("crsp", "stocknames_v2"), ("crsp", "stocknames"), ("crsp", "msenames")),
        library_regex=r"^(crsp_a_stock|crsp)$",
        table_regex=r"^(stocknames|stocknames_v2|msenames)$",
        required_any_groups=(
            ("permno",),
            ("cusip", "ncusip", "cusip8"),
            ("namedt", "nameendt", "st_date", "end_date", "date"),
            ("shrcd", "sharecode", "share_code"),
            ("exchcd", "exchange_code", "primaryexch"),
        ),
        preferred_columns=("permno", "permco", "namedt", "nameendt", "cusip", "ncusip", "ticker", "comnam", "shrcd", "exchcd", "siccd"),
    ),
    RoleSpec(
        role="ccm_link_history",
        description="CRSP/Compustat link history for fundamental controls and accounting identifiers.",
        priority_tables=(("crsp_a_ccm", "ccmxpf_lnkhist"), ("crsp", "ccmxpf_lnkhist"), ("crsp_a_ccm", "ccmxpf_linktable"), ("crsp", "ccmxpf_linktable")),
        library_regex=r"^(crsp_a_ccm|crsp)$",
        table_regex=r"^(ccmxpf_lnkhist|ccmxpf_linktable|ccmxpf_lnkused)$",
        required_any_groups=(
            ("gvkey",),
            ("lpermno", "permno"),
            ("linkdt", "link_start", "startdt"),
            ("linkenddt", "link_end", "enddt"),
            ("linktype",),
        ),
        preferred_columns=("gvkey", "lpermno", "lpermco", "linkdt", "linkenddt", "linktype", "linkprim"),
    ),
    RoleSpec(
        role="compustat_annual",
        description="Compustat annual fundamentals for book-to-market, profitability, investment, leverage, and controls.",
        priority_tables=(("comp_na_daily_all", "funda"), ("comp", "funda"), ("comp", "company"), ("compsamp_all", "funda")),
        library_regex=r"^(comp|comp_na_daily_all|compsamp|compsamp_all)$",
        table_regex=r"^(funda|company)$",
        required_any_groups=(
            ("gvkey",),
            ("datadate", "date"),
            ("at", "seq", "ceq", "sale", "ni"),
        ),
        preferred_columns=("gvkey", "datadate", "fyear", "fyr", "at", "ceq", "seq", "txditc", "pstkrv", "sale", "ni", "ib", "capx"),
        optional=True,
    ),
    RoleSpec(
        role="ff_monthly_factors",
        description="Fama-French monthly factors for alpha attribution and benchmark controls.",
        priority_tables=(("ff_all", "factors_monthly"), ("ff", "factors_monthly"), ("ff_all", "fivefactors_monthly"), ("ff", "fivefactors_monthly")),
        library_regex=r"^(ff_all|ff)$",
        table_regex=r"^(factors_monthly|fivefactors_monthly)$",
        required_any_groups=(
            ("date", "caldt", "mcaldt"),
            ("mktrf", "mkt_rf", "rmrf"),
            ("smb",),
            ("hml",),
        ),
        preferred_columns=("date", "mktrf", "mkt_rf", "smb", "hml", "rf", "umd", "mom", "rmw", "cma"),
    ),
    RoleSpec(
        role="ff_daily_factors",
        description="Fama-French daily factors for event-window and daily attribution checks.",
        priority_tables=(("ff_all", "factors_daily"), ("ff", "factors_daily"), ("ff_all", "fivefactors_daily"), ("ff", "fivefactors_daily")),
        library_regex=r"^(ff_all|ff)$",
        table_regex=r"^(factors_daily|fivefactors_daily)$",
        required_any_groups=(
            ("date", "caldt", "mcaldt"),
            ("mktrf", "mkt_rf", "rmrf"),
            ("smb",),
            ("hml",),
        ),
        preferred_columns=("date", "mktrf", "mkt_rf", "smb", "hml", "rf", "umd", "mom", "rmw", "cma"),
        optional=True,
    ),
    RoleSpec(
        role="crsp_mutual_fund_holdings",
        description="Optional CRSP mutual-fund holdings bridge for later owner-type and flow-pressure extensions.",
        priority_tables=(("crsp_q_mutualfunds", "holdings"), ("crsp", "holdings"), ("crspsamp_mf", "holdings")),
        library_regex=r"^(crsp_q_mutualfunds|crsp|crspsamp_mf)$",
        table_regex=r"^(holdings|portnomap|crsp_portno_map)$",
        required_any_groups=(("crsp_portno", "portno", "fundno"),),
        preferred_columns=("crsp_portno", "fundno", "permno", "cusip", "report_dt", "caldt", "nbr_shares", "market_val"),
        optional=True,
        later_stage=True,
    ),
)


def flatten_schema(schema: dict[str, Any]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for library, library_record in schema.get("libraries", {}).items():
        described = library_record.get("described_tables", {}) or {}
        for table, table_record in described.items():
            columns = normalize_columns(table_record if isinstance(table_record, dict) else {})
            tables.append(
                {
                    "library": library,
                    "table": table,
                    "fqtn": f"{library}.{table}",
                    "n_columns": len(columns),
                    "columns": columns,
                    "column_names": [c["name"] for c in columns],
                    "column_set": {c["name_lower"] for c in columns},
                    "table_error": table_record.get("error") if isinstance(table_record, dict) else None,
                }
            )
    return tables


def group_hits(column_set: set[str], group: tuple[str, ...]) -> list[str]:
    aliases = {g.lower() for g in group}
    return sorted(column_set.intersection(aliases))


def score_table(table: dict[str, Any], spec: RoleSpec) -> dict[str, Any]:
    lib = norm_name(table["library"])
    tbl = norm_name(table["table"])
    cols = table["column_set"]

    required_hits: list[dict[str, Any]] = []
    required_score = 0
    missing_groups: list[list[str]] = []
    for group in spec.required_any_groups:
        hits = group_hits(cols, group)
        required_hits.append({"aliases": list(group), "hits": hits, "ok": bool(hits)})
        if hits:
            required_score += 1
        else:
            missing_groups.append(list(group))

    preferred_hits = sorted(cols.intersection({c.lower() for c in spec.preferred_columns}))

    score = 0.0
    if re.search(spec.library_regex, lib) and re.search(spec.table_regex, tbl):
        score += 40.0
    if re.search(spec.library_regex, lib):
        score += 10.0
    if re.search(spec.table_regex, tbl):
        score += 10.0

    for rank, (plib, ptbl) in enumerate(spec.priority_tables):
        if lib == plib and tbl == ptbl:
            score += 100.0 - rank * 5.0
            break

    score += required_score * 15.0
    score += min(len(preferred_hits), 16) * 2.0
    score += min(table["n_columns"], 80) / 20.0

    status = "pass" if not missing_groups else ("warn" if spec.optional or required_score >= max(1, len(spec.required_any_groups) - 1) else "fail")

    return {
        "score": round(score, 4),
        "status": status,
        "required_hits": required_hits,
        "missing_groups": missing_groups,
        "preferred_hits": preferred_hits,
    }


def choose_role(tables: list[dict[str, Any]], spec: RoleSpec) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for table in tables:
        detail = score_table(table, spec)
        if detail["score"] <= 0:
            continue
        scored.append({"table": table, "detail": detail})
    scored.sort(key=lambda x: x["detail"]["score"], reverse=True)

    alternatives = []
    for item in scored[:8]:
        table = item["table"]
        detail = item["detail"]
        alternatives.append(
            {
                "library": table["library"],
                "table": table["table"],
                "fqtn": table["fqtn"],
                "score": detail["score"],
                "status": detail["status"],
                "n_columns": table["n_columns"],
                "preferred_hits": detail["preferred_hits"],
                "missing_groups": detail["missing_groups"],
            }
        )

    if not scored:
        return {
            "role": spec.role,
            "description": spec.description,
            "status": "missing_optional" if spec.optional else "missing_required",
            "optional": spec.optional,
            "later_stage": spec.later_stage,
            "selected": None,
            "alternatives": [],
        }

    best = scored[0]
    table = best["table"]
    detail = best["detail"]
    selected = {
        "library": table["library"],
        "table": table["table"],
        "fqtn": table["fqtn"],
        "score": detail["score"],
        "status": detail["status"],
        "n_columns": table["n_columns"],
        "columns": table["column_names"],
        "column_types": {c["name"]: c.get("data_type", "") for c in table["columns"]},
        "required_hits": detail["required_hits"],
        "missing_groups": detail["missing_groups"],
        "preferred_hits": detail["preferred_hits"],
    }
    return {
        "role": spec.role,
        "description": spec.description,
        "status": detail["status"],
        "optional": spec.optional,
        "later_stage": spec.later_stage,
        "selected": selected,
        "alternatives": alternatives,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except Exception:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, width=110), encoding="utf-8")


def status_icon(status: str) -> str:
    return {
        "pass": "✅",
        "warn": "⚠️",
        "missing_optional": "➖",
        "missing_required": "❌",
        "fail": "❌",
    }.get(status, "❓")


def write_markdown(path: Path, contract: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "# 002 Schema contract",
            "",
            "This document freezes the WRDS source-table contract inferred from Phase 0 metadata. It contains metadata only: table names, column names, data types, and role assignments. It does not contain raw vendor rows.",
            "",
            "## Run metadata",
            "",
            f"- run_id: `{contract['run_id']}`",
            f"- created_utc: `{contract['created_utc']}`",
            f"- source_schema_json: `{contract['source_schema_json']}`",
            f"- available_libraries: `{contract.get('available_library_count', 'unknown')}`",
            "",
            "## Selected source roles",
            "",
            "| Role | Status | Selected table | Score | Required column coverage |",
            "|---|---:|---|---:|---|",
        ]
    )

    for role in contract["roles"]:
        selected = role.get("selected") or {}
        coverage = []
        for hit in selected.get("required_hits", []):
            label = "/".join(hit["aliases"][:3])
            coverage.append(("✓ " if hit["ok"] else "× ") + label)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{role['role']}`",
                    f"{status_icon(role['status'])} `{role['status']}`",
                    f"`{selected.get('fqtn', 'not selected')}`",
                    str(selected.get("score", "")),
                    "; ".join(coverage),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Pipeline implication",
            "",
            "```mermaid",
            "flowchart LR",
            "    A[13F holdings and manager metadata] --> B[Filing-date availability and amendment logic]",
            "    B --> C[CUSIP to CRSP mapping through stock-name histories]",
            "    C --> D[CRSP common-stock monthly and daily panel]",
            "    D --> E[Manager-stock sparse matrices]",
            "    E --> F[Stock-stock common-ownership graph]",
            "    F --> G[Fragility and spillover features]",
            "    G --> H[Returns, downside labels, and attribution factors]",
            "```",
            "",
            "## Column previews",
            "",
        ]
    )

    for role in contract["roles"]:
        selected = role.get("selected")
        if not selected:
            continue
        lines.extend([f"### `{role['role']}` — `{selected['fqtn']}`", ""])
        cols = selected.get("columns", [])
        types = selected.get("column_types", {})
        preview = cols[:80]
        lines.extend(["| Column | Type |", "|---|---|"])
        for col in preview:
            lines.append(f"| `{col}` | `{types.get(col, '')}` |")
        if len(cols) > len(preview):
            lines.append(f"| ... | {len(cols) - len(preview)} more columns |")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, contract: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role",
                "status",
                "optional",
                "later_stage",
                "selected_table",
                "score",
                "n_columns",
                "preferred_hits",
                "missing_groups",
            ],
        )
        writer.writeheader()
        for role in contract["roles"]:
            selected = role.get("selected") or {}
            writer.writerow(
                {
                    "role": role["role"],
                    "status": role["status"],
                    "optional": role["optional"],
                    "later_stage": role["later_stage"],
                    "selected_table": selected.get("fqtn", ""),
                    "score": selected.get("score", ""),
                    "n_columns": selected.get("n_columns", ""),
                    "preferred_hits": ",".join(selected.get("preferred_hits", [])),
                    "missing_groups": json.dumps(selected.get("missing_groups", [])),
                }
            )


def write_coverage_figure(path: Path, contract: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[002_build_schema_contract] skipped matplotlib figure: {exc}")
        return

    roles = [r["role"] for r in contract["roles"]]
    max_groups = max(len((r.get("selected") or {}).get("required_hits", [])) for r in contract["roles"])
    matrix = np.full((len(roles), max_groups), np.nan)
    labels = []
    for i, role in enumerate(contract["roles"]):
        selected = role.get("selected") or {}
        hits = selected.get("required_hits", [])
        labels.append(role["role"].replace("_", "\n"))
        for j, hit in enumerate(hits):
            matrix[i, j] = 1.0 if hit.get("ok") else 0.0

    fig_w = max(9, max_groups * 1.15)
    fig_h = max(6, len(roles) * 0.58)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    ax.set_title("Step 002 schema contract: required column coverage")
    ax.set_xlabel("Required alias group")
    ax.set_ylabel("Source role")
    ax.set_yticks(range(len(roles)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(max_groups))
    ax.set_xticklabels([f"G{i+1}" for i in range(max_groups)])
    for i, role in enumerate(contract["roles"]):
        selected = role.get("selected") or {}
        hits = selected.get("required_hits", [])
        for j in range(max_groups):
            if j >= len(hits):
                txt = ""
            else:
                txt = "✓" if hits[j].get("ok") else "×"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Coverage")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_interactive_html(path: Path, contract: dict[str, Any]) -> None:
    rows = []
    for role in contract["roles"]:
        selected = role.get("selected") or {}
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(role['role'])}</code></td>"
            f"<td>{status_icon(role['status'])} <code>{html.escape(role['status'])}</code></td>"
            f"<td><code>{html.escape(selected.get('fqtn', 'not selected'))}</code></td>"
            f"<td>{html.escape(str(selected.get('score', '')))}</td>"
            f"<td>{html.escape(', '.join(selected.get('preferred_hits', [])[:20]))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>002 Schema Contract</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.45; }}
    h1 {{ margin-bottom: 0.2rem; }}
    .meta {{ color: #555; margin-bottom: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.55rem; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: white; }}
    code {{ background: #f6f8fa; padding: 0.1rem 0.25rem; border-radius: 0.25rem; }}
  </style>
</head>
<body>
  <h1>002 Schema Contract</h1>
  <div class="meta">Run <code>{html.escape(contract['run_id'])}</code> · created {html.escape(contract['created_utc'])} · metadata only</div>
  <table>
    <thead><tr><th>Role</th><th>Status</th><th>Selected table</th><th>Score</th><th>Preferred column hits</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser()
    schema_dir = root / "artifacts" / "schema"
    schema_path = latest_file(schema_dir, "schema_discovery_full_*.json")
    schema = load_schema(schema_path)
    tables = flatten_schema(schema)

    print(f"[002_build_schema_contract] source schema: {schema_path}")
    print(f"[002_build_schema_contract] discovered described tables: {len(tables)}")

    roles = [choose_role(tables, spec) for spec in ROLE_SPECS]

    status_counts: dict[str, int] = {}
    for role in roles:
        status_counts[role["status"]] = status_counts.get(role["status"], 0) + 1

    contract = {
        "contract_version": "0.2.0",
        "run_id": args.run_id,
        "created_utc": utc_now(),
        "project_root": str(root),
        "source_schema_json": str(schema_path),
        "source_schema_run_id": schema.get("run_id"),
        "available_library_count": schema.get("available_library_count"),
        "candidate_libraries": schema.get("candidate_libraries", []),
        "metadata_only": True,
        "policy": {
            "raw_vendor_rows_extracted": False,
            "phase": "schema_contract_freeze",
            "next_allowed_action": "small date-bounded pilot extraction after contract validation",
        },
        "status_counts": status_counts,
        "roles": roles,
    }

    out_base = schema_dir / f"002_schema_contract_{args.run_id}"
    json_path = out_base.with_suffix(".json")
    yaml_path = out_base.with_suffix(".yaml")
    csv_path = schema_dir / f"002_schema_contract_{args.run_id}_summary.csv"
    md_path = schema_dir / f"002_schema_contract_{args.run_id}.md"
    docs_path = root / "docs" / "schema_contract.md"
    fig_path = root / "artifacts" / "figures_static" / f"002_schema_contract_coverage_{args.run_id}.png"
    html_path = root / "artifacts" / "figures_interactive" / f"002_schema_contract_{args.run_id}.html"
    config_path = root / "configs" / "schema_contract.yaml"

    write_json(json_path, contract)
    write_yaml(yaml_path, contract)
    write_yaml(config_path, contract)
    write_csv(csv_path, contract)
    write_markdown(md_path, contract)
    write_markdown(docs_path, contract)
    write_coverage_figure(fig_path, contract)
    write_interactive_html(html_path, contract)

    print(f"[002_build_schema_contract] wrote {json_path}")
    print(f"[002_build_schema_contract] wrote {yaml_path}")
    print(f"[002_build_schema_contract] wrote {csv_path}")
    print(f"[002_build_schema_contract] wrote {md_path}")
    print(f"[002_build_schema_contract] wrote {docs_path}")
    print(f"[002_build_schema_contract] wrote {fig_path}")
    print(f"[002_build_schema_contract] wrote {html_path}")
    print(f"[002_build_schema_contract] status_counts={status_counts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
