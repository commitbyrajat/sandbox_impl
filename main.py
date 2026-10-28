"""Query local Parquet and a PostgreSQL snapshot with datafusion-python."""

import argparse
import os
from pathlib import Path

import psycopg
from psycopg import sql
import pyarrow as pa
from datafusion import SessionContext

from datasource_mapping import SOURCES, SourceDefinition, SourceId


ROOT = Path(__file__).resolve().parent
PG_TABLES = tuple(SOURCES["smartwealth"].tables.values())

# PostgreSQL OIDs for the types used by the sample database. Explicit types
# preserve dates, decimals, and empty tables instead of guessing from values.
PG_TYPES = {
    16: pa.bool_(), 20: pa.int64(), 21: pa.int16(), 23: pa.int32(),
    25: pa.string(), 1042: pa.string(), 1043: pa.string(), 1082: pa.date32(),
}


def create_context(
    parquet_dir: Path | None = None,
    *, source_ids: list[SourceId] | tuple[SourceId, ...] | None = None,
    source_definitions: dict[SourceId, SourceDefinition] | None = None,
) -> SessionContext:
    """Load only selected sources. Existing callers default to both datasets."""
    definitions = SOURCES if source_definitions is None else source_definitions
    selected = tuple(definitions) if source_ids is None else tuple(source_ids)
    if set(selected) - definitions.keys():
        raise ValueError("Unknown data source")
    ctx = SessionContext()
    if "credit_cards" in selected:
        source = definitions["credit_cards"]
        directory = parquet_dir if parquet_dir is not None else ROOT / source.location
        for registered_name, filename in source.tables.items():
            ctx.register_parquet(registered_name, str(directory / filename))

    if "smartwealth" not in selected:
        return ctx

    connection = {
        "host": os.getenv("SMARTWEALTH_DB_HOST", "127.0.0.1"),
        "port": os.getenv("SMARTWEALTH_DB_PORT", "5433"),
        "dbname": os.getenv("SMARTWEALTH_DB_NAME", "smartwealth"),
        "user": os.getenv("SMARTWEALTH_DB_USER", "smartwealth"),
        "password": os.getenv("SMARTWEALTH_DB_PASSWORD", "smartwealth_local"),
        "connect_timeout": 5,
    }
    with psycopg.connect(**connection) as conn:
        # All six relations come from one consistent, read-only snapshot.
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        for registered_name, name in definitions["smartwealth"].tables.items():
            with conn.cursor() as cursor:
                cursor.execute(sql.SQL("SELECT * FROM {}.{}").format(
                    sql.Identifier(definitions["smartwealth"].location), sql.Identifier(name),
                ))
                fields = []
                for column in cursor.description:
                    if column.type_code == 1700:  # PostgreSQL NUMERIC
                        dtype = pa.decimal128(
                            column.precision if column.precision is not None else 38,
                            column.scale if column.scale is not None else 6,
                        )
                    else:
                        dtype = PG_TYPES[column.type_code]
                    fields.append(pa.field(column.name, dtype))
                schema = pa.schema(fields)
                batches = []
                while rows := cursor.fetchmany(1024):
                    batches.append(pa.RecordBatch.from_arrays(
                        [pa.array(values, type=field.type)
                         for values, field in zip(zip(*rows), schema)],
                        schema=schema,
                    ))
                if not batches:
                    batches.append(pa.RecordBatch.from_arrays(
                        [pa.array([], type=field.type) for field in schema], schema=schema,
                    ))
                ctx.register_record_batches(registered_name, [batches])
    return ctx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", help="Run custom DataFusion SQL against cc_* and sw_* tables")
    parser.add_argument("--sql-file", type=Path, help="Read a DataFusion SQL query from a file")
    parser.add_argument("--limit", type=int, default=20, help="Maximum displayed rows")
    args = parser.parse_args()
    if args.sql and args.sql_file:
        parser.error("Use either --sql or --sql-file")
    if args.limit < 1:
        parser.error("--limit must be positive")
    query = args.sql or (args.sql_file or ROOT / "queries/customer_overview.sql").read_text()
    ctx = create_context()
    ctx.sql(query).show(num=args.limit)


if __name__ == "__main__":
    main()
