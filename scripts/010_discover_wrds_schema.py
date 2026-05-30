from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import sys
import traceback
from typing import Any


DEFAULT_LIBRARIES = [
    "tr_13f",
    "tfn",
    "crsp_a_stock",
    "crsp",
    "crsp_a_ccm",
    "comp",
    "ff_all",
    "crsp_q_mutualfunds",
    "tr_mutualfunds",
]

DEFAULT_TABLE_REGEX = (
    r"(13f|s34|hold|holding|mgr|manager|fil|form|cusip|type|"
    r"msf|dsf|msenames|stocknames|ccm|link|factor|port|fund|mflink)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mask_username(value: str | None) -> str:
    if not value:
        return "unset"
    if len(value) <= 3:
        return value[0] + "***"
    return value[:2] + "***" + value[-1]


def package_version(package: str) -> str:
    try:
        name = "scikit-learn" if package == "sklearn" else package
        return metadata.version(name)
    except Exception:
        return "unknown"


def load_yaml_config(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "data.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def select_candidate_libraries(available_libraries: list[str], configured: list[str]) -> list[str]:
    available_set = set(available_libraries)
    selected = [library for library in configured if library in available_set]

    fallback_patterns = ["13f", "tfn", "crsp", "comp", "ff", "mutual"]
    for library in available_libraries:
        lower = library.lower()
        if any(pattern in lower for pattern in fallback_patterns) and library not in selected:
            selected.append(library)

    return selected[:30]


def select_candidate_tables(tables: list[str], regex: str, limit: int) -> list[str]:
    pattern = re.compile(regex, re.IGNORECASE)
    matched = [table for table in tables if pattern.search(table)]
    selected = matched if matched else tables
    return sorted(selected)[:limit]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# WRDS schema discovery report",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- mode: `{payload['mode']}`",
        f"- started_utc: `{payload['started_utc']}`",
        f"- finished_utc: `{payload['finished_utc']}`",
        f"- wrds_username: `{payload['wrds_username_masked']}`",
        f"- available_library_count: `{payload.get('available_library_count', 0)}`",
        f"- candidate_library_count: `{len(payload.get('candidate_libraries', []))}`",
        "",
        "## Candidate libraries",
        "",
    ]
    for library in payload.get("candidate_libraries", []):
        info = payload.get("libraries", {}).get(library, {})
        lines.append(
            f"- `{library}`: {info.get('n_tables', 0)} tables, "
            f"{len(info.get('candidate_tables', []))} candidate tables, "
            f"{len(info.get('described_tables', {}))} described"
        )

    lines.extend(["", "## Notes", ""])
    lines.append(
        "This phase records metadata from `information_schema.columns` only and does not extract security-level data rows."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def query_one_column(db: Any, sql: str, params: dict[str, Any] | None = None) -> list[str]:
    result = db.raw_sql(sql, params=params)
    if result.empty:
        return []
    return [str(x) for x in result.iloc[:, 0].dropna().tolist()]


def list_libraries_safe(db: Any) -> list[str]:
    try:
        return sorted(str(x) for x in db.list_libraries())
    except Exception:
        return query_one_column(
            db,
            """
            select schema_name
            from information_schema.schemata
            order by schema_name
            """,
        )


def list_tables_metadata_only(db: Any, schema: str) -> list[str]:
    return query_one_column(
        db,
        """
        select table_name
        from information_schema.tables
        where table_schema = %(schema)s
          and table_type in ('BASE TABLE', 'VIEW', 'FOREIGN TABLE')
        order by table_name
        """,
        {"schema": schema},
    )


def describe_table_metadata_only(db: Any, schema: str, table: str) -> list[dict[str, Any]]:
    frame = db.raw_sql(
        """
        select
            column_name,
            ordinal_position,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            datetime_precision,
            is_nullable
        from information_schema.columns
        where table_schema = %(schema)s
          and table_name = %(table)s
        order by ordinal_position
        """,
        params={"schema": schema, "table": table},
    )
    return [
        {str(k): (None if v is None else str(v)) for k, v in row.items()}
        for row in frame.to_dict("records")
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml_config(root)
    wrds_config = config.get("wrds", {})
    schema_config = config.get("schema_discovery", {})

    configured_libraries = wrds_config.get("candidate_libraries", DEFAULT_LIBRARIES)
    table_regex = schema_config.get("table_name_regex", DEFAULT_TABLE_REGEX)

    if args.mode == "pilot":
        table_limit = int(schema_config.get("pilot_tables_per_library", 5))
    else:
        table_limit = int(schema_config.get("full_tables_per_library", 80))

    username_env_var = wrds_config.get("username_env_var", "WRDS_USERNAME")
    wrds_username = os.environ.get(username_env_var) or os.environ.get("USER")

    payload: dict[str, Any] = {
        "run_id": args.run_id,
        "mode": args.mode,
        "started_utc": utc_now(),
        "project_root": str(root),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "wrds_username_masked": mask_username(wrds_username),
        "configured_libraries": configured_libraries,
        "table_regex": table_regex,
        "table_limit_per_library": table_limit,
        "metadata_only": True,
        "package_versions": {
            package: package_version(package)
            for package in [
                "wrds",
                "pandas",
                "numpy",
                "sqlalchemy",
                "psycopg2",
                "pyarrow",
                "duckdb",
                "polars",
            ]
        },
        "libraries": {},
        "errors": [],
    }

    print(f"[010_schema_discovery] mode={args.mode} run_id={args.run_id}")
    print(f"[010_schema_discovery] root={root}")
    print(f"[010_schema_discovery] out_dir={out_dir}")
    print("[010_schema_discovery] connecting to WRDS")

    try:
        import wrds

        db = wrds.Connection(wrds_username=wrds_username)
    except Exception as exc:
        payload["finished_utc"] = utc_now()
        payload["status"] = "failed_connection"
        payload["errors"].append(
            {
                "stage": "connect",
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=10),
            }
        )
        failed_path = out_dir / f"schema_discovery_{args.mode}_{args.run_id}_FAILED.json"
        write_json(failed_path, payload)
        print(f"[010_schema_discovery] FAILED connection. Wrote {failed_path}")
        return 11

    try:
        available_libraries = list_libraries_safe(db)
        payload["available_library_count"] = len(available_libraries)
        payload["available_libraries"] = available_libraries

        candidate_libraries = select_candidate_libraries(available_libraries, configured_libraries)
        payload["candidate_libraries"] = candidate_libraries
        payload["missing_configured_libraries"] = [
            library for library in configured_libraries if library not in available_libraries
        ]

        print(f"[010_schema_discovery] available libraries: {len(available_libraries)}")
        print(f"[010_schema_discovery] candidate libraries: {candidate_libraries}")

        for library in candidate_libraries:
            print(f"[010_schema_discovery] listing metadata tables for {library}")
            library_record: dict[str, Any] = {
                "n_tables": 0,
                "candidate_tables": [],
                "described_tables": {},
                "errors": [],
            }

            try:
                tables = list_tables_metadata_only(db, library)
                library_record["n_tables"] = len(tables)
                candidate_tables = select_candidate_tables(tables, table_regex, table_limit)
                library_record["candidate_tables"] = candidate_tables
            except Exception as exc:
                library_record["errors"].append({"stage": "list_tables", "error": repr(exc)})
                payload["libraries"][library] = library_record
                continue

            for table in library_record["candidate_tables"]:
                print(f"[010_schema_discovery] metadata columns for {library}.{table}")
                try:
                    records = describe_table_metadata_only(db, library, table)
                    library_record["described_tables"][table] = {
                        "n_columns": len(records),
                        "columns": records,
                    }
                except Exception as exc:
                    library_record["described_tables"][table] = {
                        "error": repr(exc),
                        "n_columns": 0,
                        "columns": [],
                    }

            payload["libraries"][library] = library_record

        payload["status"] = "ok"
        payload["finished_utc"] = utc_now()

    finally:
        try:
            db.close()
        except Exception:
            pass

    json_path = out_dir / f"schema_discovery_{args.mode}_{args.run_id}.json"
    csv_path = out_dir / f"schema_discovery_{args.mode}_{args.run_id}_summary.csv"
    md_path = out_dir / f"schema_discovery_{args.mode}_{args.run_id}.md"

    write_json(json_path, payload)
    write_markdown_report(md_path, payload)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "library",
                "table",
                "n_columns",
                "column_preview",
                "error",
            ],
        )
        writer.writeheader()
        for library, library_record in payload["libraries"].items():
            for table, table_record in library_record.get("described_tables", {}).items():
                records = table_record.get("columns", [])
                preview_names = [row.get("column_name", "") for row in records[:14]]
                writer.writerow(
                    {
                        "mode": args.mode,
                        "library": library,
                        "table": table,
                        "n_columns": table_record.get("n_columns", len(records)),
                        "column_preview": " | ".join(str(x) for x in preview_names if x),
                        "error": table_record.get("error", ""),
                    }
                )

    print(f"[010_schema_discovery] wrote {json_path}")
    print(f"[010_schema_discovery] wrote {csv_path}")
    print(f"[010_schema_discovery] wrote {md_path}")
    print("[010_schema_discovery] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
