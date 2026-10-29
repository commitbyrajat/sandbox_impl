"""Workspace-isolated DataFusion access and snapshot loading."""

import json
from dataclasses import dataclass, field
from typing import Any

import sqlglot
from datafusion import SessionContext, SQLOptions
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from datasource_mapping import (RequestScope, SourceId, TenantId, WorkspaceId,
                                resolve_scope, tables_for)
from main import create_context
from planning import ExecutionPlan

MAX_ROWS = 200


@dataclass
class DataSources:
    context: SessionContext
    tenant_id: TenantId
    workspace_id: WorkspaceId
    queries: list[str] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        scope = resolve_scope(self.tenant_id, self.workspace_id)
        self.tenant_id, self.workspace_id = scope.tenant_id, scope.workspace_id

    @property
    def selected_sources(self) -> tuple[SourceId, ...]:
        return tuple(resolve_scope(self.tenant_id, self.workspace_id).workspace.sources)

    @property
    def tables(self) -> tuple[str, ...]:
        scope = resolve_scope(self.tenant_id, self.workspace_id)
        return tables_for(self.selected_sources, definitions=scope.workspace.sources)

    def describe(self) -> dict[str, Any]:
        return {
            name: [
                {"name": f.name, "type": str(f.type)}
                for f in self.context.table(name).schema()
            ]
            for name in self.tables
        }

    def query(self, statement: str) -> dict[str, Any]:
        """Read only routed tables, resolving CTEs by scope to prevent name bypasses."""
        if len(statement) > 20_000:
            raise ValueError("SQL must be at most 20,000 characters")
        statements = sqlglot.parse(statement, read="postgres")
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            raise ValueError("Only one SELECT query (optionally with CTEs) is allowed")
        tree = statements[0]
        # Inspect resolved sources instead of globally trusting all CTE aliases.
        for scope in traverse_scope(tree):
            for source in scope.sources.values():
                if isinstance(source, exp.Table) and (
                    not isinstance(source.this, exp.Identifier)
                    or source.db
                    or source.catalog
                    or source.name.lower() not in self.tables
                ):
                    raise ValueError(
                        "Query only tables belonging to the specified workspace"
                    )
        options = (
            SQLOptions()
            .with_allow_ddl(False)
            .with_allow_dml(False)
            .with_allow_statements(False)
        )
        frame = self.context.sql(statement, options=options)
        rows = frame.limit(MAX_ROWS + 1).to_pylist()
        result_id = f"query_{len(self.results) + 1}"
        result = {
            "result_id": result_id,
            "columns": frame.schema().names,
            "rows": rows[:MAX_ROWS],
            "truncated": len(rows) > MAX_ROWS,
        }
        if len(json.dumps(result, default=str)) > 24_000:
            raise ValueError(
                "Result too large. Aggregate or select fewer columns/rows in SQL"
            )
        self.queries.append(statement)
        self.results[result_id] = result
        return result

    def python_inputs(self, result_ids: list[str]) -> dict[str, Any]:
        if not result_ids:
            raise ValueError("Run SQL first and provide at least one result_id")
        data = {}
        for result_id in result_ids:
            if result_id not in self.results:
                raise ValueError(f"Unknown SQL result_id: {result_id}")
            result = self.results[result_id]
            if result["truncated"]:
                raise ValueError(
                    "Cannot calculate over truncated rows; aggregate/filter in SQL first"
                )
            data[result_id] = result["rows"]
        return data


def load_sources(scope: RequestScope, plan: ExecutionPlan) -> DataSources:
    context = create_context(
        source_ids=plan.sources, source_definitions=scope.workspace.sources
    )
    return DataSources(
        context, tenant_id=scope.tenant_id, workspace_id=scope.workspace_id
    )
