"""Serialize the current SQL-backed sandbox binding for code generation only."""

import json
import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal

MAX_DATA_JSON_BYTES = 24_000


def json_value(value):
    """Display representations must not alter the typed rows sent to Monty."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def python_data_context(results, result_ids, *, max_bytes=MAX_DATA_JSON_BYTES):
    """Expose exact result IDs/structure, a bounded JSON snapshot and runtime types."""
    ids = list(dict.fromkeys(result_ids))
    data, descriptors, issues = {}, {}, []
    for result_id in ids:
        if result_id not in results:
            issues.append(
                f"SQL result {result_id!r} is unavailable; use the current producer IDs"
            )
            continue
        result = results[result_id]
        rows = result["rows"]
        columns = result.get("columns", list(rows[0]) if rows else [])
        arrow_fields = {
            field["name"]: field for field in result.get("column_types", [])
        }
        descriptors[result_id] = {
            "stored_row_count": len(rows),
            "sql_truncated": bool(result.get("truncated", False)),
            "usable_in_python": not result.get("truncated", False),
            "columns": {
                column: {
                    "arrow_type": arrow_fields.get(column, {}).get("type"),
                    "nullable": arrow_fields.get(column, {}).get("nullable"),
                    "python_types": sorted(
                        {
                            type(row[column]).__module__
                            + "."
                            + type(row[column]).__qualname__
                            for row in rows
                            if column in row
                        }
                    ),
                }
                for column in columns
            },
        }
        if result.get("truncated", False):
            issues.append(
                f"{result_id} is SQL-truncated and cannot be used in Monty; repair the SQL input first"
            )
        else:
            data[result_id] = json_value(rows)

    def encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)

    serialized = encode(data)
    complete = not issues
    if len(serialized.encode("utf-8")) > max_bytes:
        complete = False
        preview = {key: [] for key in data}
        serialized = encode(preview)
        if len(serialized.encode("utf-8")) > max_bytes:
            raise ValueError(
                "The Python data-context byte budget cannot hold the result IDs"
            )
        # Prefix rows per result, distributed fairly; never truncate inside a value.
        pending = set(data)
        while pending:
            for key in data:
                if key not in pending:
                    continue
                index = len(preview[key])
                if index == len(data[key]):
                    pending.remove(key)
                    continue
                candidate = {**preview, key: [*preview[key], data[key][index]]}
                encoded = encode(candidate)
                if len(encoded.encode("utf-8")) <= max_bytes:
                    preview = candidate
                    serialized = encoded
                else:
                    pending.remove(key)
        data = preview
    for key, descriptor in descriptors.items():
        descriptor["snapshot_row_count"] = len(data.get(key, []))

    return {
        "data_json": serialized,
        "snapshot_complete": complete,
        "snapshot_bytes": len(serialized.encode("utf-8")),
        "max_snapshot_bytes": max_bytes,
        "binding_type": "dict[str, list[dict[str, object]]]",
        "results": descriptors,
        "input_issues": issues,
        "serialization_note": (
            "This JSON is a display of the host-supplied data object, not code or replacement inputs. "
            "Decimal values display as exact strings but remain decimal.Decimal in Monty. "
            "Dates/times display as ISO strings; bytes as hex; non-finite floats and timedeltas as strings. "
            "Use the actual python_types below, including builtins.NoneType for nulls. "
            "Read data[result_id] directly; do not json.loads(data) or copy snapshot values into code. "
            "An incomplete snapshot is a model-context preview only; Monty receives all complete selected SQL rows."
        ),
    }
