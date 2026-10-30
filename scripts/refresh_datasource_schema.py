"""Refresh registry column names/types from the configured DataFusion sources."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasource_mapping import resolve_scope
from main import create_context


def main():
    load_dotenv(ROOT / ".env")
    path = ROOT / "datasource.json"
    document = json.loads(path.read_text())
    tables = 0
    for tenant_id, tenant in document["tenants"].items():
        for workspace_id, workspace in tenant["workspaces"].items():
            scope = resolve_scope(tenant_id, workspace_id)
            context = create_context(
                source_ids=list(workspace["sources"]),
                source_definitions=scope.workspace.sources,
            )
            for source in workspace["sources"].values():
                source["columns"] = {
                    name: {
                        field.name: {
                            "type": str(field.type),
                            "nullable": field.nullable,
                        }
                        for field in context.table(name).schema()
                    }
                    for name in source["tables"]
                }
                tables += len(source["tables"])
    # Write only after every configured source has been inspected successfully.
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Updated datasource.json column schemas for {tables} tables")


if __name__ == "__main__":
    main()
