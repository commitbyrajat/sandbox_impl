"""Bind configured analytical observation units before validating a generated plan."""

import copy

from runtime import trace
import json


def bind_observations(value, catalog, policies, analysis_kind=None):
    if (
        not policies
        or not isinstance(value, dict)
        or not isinstance(value.get("steps"), list)
    ):
        return value
    value = copy.deepcopy(value)
    consumers = {
        name
        for step in value["steps"]
        if isinstance(step, dict) and step.get("action") == "python"
        for name in step.get("input_steps", [])
    }
    equivalences = {}
    for relation in catalog.relationships.values():
        for left, right in relation.column_pairs:
            a, b = (relation.left_table, left), (relation.right_table, right)
            equivalences.setdefault(a, set()).add(b)
            equivalences.setdefault(b, set()).add(a)

    def equivalent(a, b):
        pending = [a]
        seen = set()
        while pending:
            key = pending.pop()
            if key == b:
                return True
            if key not in seen:
                seen.add(key)
                pending.extend(equivalences.get(key, set()) - seen)
        return False

    def metric_tables(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen or name not in catalog.metrics:
            return set()
        seen.add(name)
        metric = catalog.metrics[name]
        result = {metric.table} if metric.table else set()
        if metric.formula:

            def walk(node):
                if node.op == "ref":
                    result.update(metric_tables(node.name, seen))
                for arg in node.args:
                    walk(arg)

            walk(metric.formula)
        if metric.cumulative:
            result.update(metric_tables(metric.cumulative.input_metric, seen))
        return result

    for step in value["steps"]:
        if (
            not isinstance(step, dict)
            or step.get("action") != "sql"
            or step.get("step_id") not in consumers
        ):
            continue
        query, contract = step.get("logical_query"), step.get("output_contract")
        if not isinstance(query, dict) or not isinstance(contract, dict):
            continue
        aliases = {
            item.get("name")
            for section in ("dimensions", "metrics", "calculations")
            for item in query.get(section, [])
            if isinstance(item, dict)
        }
        for alias in contract.get("required_columns", {}):
            if alias not in aliases and alias in catalog.metrics:
                query.setdefault("metrics", []).append({"name": alias, "metric": alias})
                trace(
                    "CATALOG_OUTPUT_BOUND: "
                    + json.dumps({"step_id": step["step_id"], "metric": alias})
                )
        selected = {
            m.get("metric") for m in query.get("metrics", []) if isinstance(m, dict)
        }
        for name, policy in policies.items():
            if not selected.intersection(policy.metrics):
                continue
            catalog.check_field(policy.table, None)
            grain = catalog.tables[policy.table].grain
            dimensions = query.setdefault("dimensions", [])
            required = contract.setdefault("required_columns", {})
            metric_names = {item.get("name") for item in query.get("metrics", [])}
            calculation_names = {
                item.get("name") for item in query.get("calculations", [])
            }
            composite_keys = []
            # Identity aliases are explicit semantic metadata. Never infer an
            # identity role from a name, free-text meaning, or ordinary alias.
            if len(grain) > 1:
                for alias in policy.identity_aliases:
                    if (
                        alias in catalog.columns[policy.table]
                        or alias in metric_names
                        or alias in calculation_names
                        or alias not in required
                        or required[alias].get("type") not in ("string", "any")
                    ):
                        continue
                    projection = next(
                        (d for d in dimensions if d.get("name") == alias), None
                    )
                    if projection is not None:
                        field = projection.get("field", {})
                        if (
                            field.get("table") != policy.table
                            or field.get("column") not in grain
                        ):
                            continue
                        dimensions.remove(projection)
                    composite_keys.append(alias)
            aliases = {
                item.get("name")
                for section in ("dimensions", "metrics", "calculations")
                for item in query.get(section, [])
                if isinstance(item, dict)
            }
            retained = []
            for column in grain:
                # A required canonical identity alias cannot disappear when the
                # model gives the same field another label (e.g. a synthetic key).
                # Keep both projections when both names are explicitly requested.
                canonical_required = column in required
                candidate = next(
                    (
                        d
                        for d in dimensions
                        if not canonical_required or d.get("name") == column
                        if d.get("transform", "identity") == "identity"
                        and isinstance(d.get("field"), dict)
                        and equivalent(
                            (d["field"].get("table"), d["field"].get("column")),
                            (policy.table, column),
                        )
                    ),
                    None,
                )
                if candidate is None:
                    alias = column
                    while alias in aliases:
                        alias = "observation_" + alias
                    candidate = {
                        "name": alias,
                        "field": {"table": policy.table, "column": column},
                        "transform": "identity",
                    }
                    dimensions.append(candidate)
                    aliases.add(alias)
                candidate["field"] = {"table": policy.table, "column": column}
                alias = candidate["name"]
                retained.append(alias)
                required.setdefault(
                    alias,
                    {
                        "type": "any",
                        "nullable": False,
                        "meaning": "Configured observation identity: "
                        + policy.table
                        + "."
                        + column,
                    },
                )
            query["base_table"] = policy.table
            for alias in composite_keys:
                args = [{"op": "ref", "name": key} for key in retained]
                while len(args) > 8:
                    args = [{"op": "key", "args": args[:8]}, *args[8:]]
                query.setdefault("calculations", []).append(
                    {
                        "name": alias,
                        "expression": {"op": "key", "args": args},
                    }
                )
                trace(
                    "OBSERVATION_IDENTITY_BOUND: "
                    + json.dumps(
                        {
                            "step_id": step["step_id"],
                            "alias": alias,
                            "components": retained,
                        }
                    )
                )
            ordering = query.setdefault("order_by", [])
            ordered = {item.get("name") for item in ordering if isinstance(item, dict)}
            ordering.extend(
                {"name": alias, "descending": False}
                for alias in retained
                if alias not in ordered
            )
            contract["grain"] = list(
                dict.fromkeys([*retained, *contract.get("grain", [])])
            )
            contract["population"] = None
            contract["min_rows"] = max(contract.get("min_rows", 0), policy.min_rows)
            time = query.setdefault("time", {"selection": "all"})
            if time.get("selection") in ("latest", "latest_common"):
                tables = {policy.table}
                for metric in selected:
                    tables.update(metric_tables(metric))
                time["selection"] = "latest_common"
                time["tables"] = sorted(
                    t for t in tables if catalog.tables[t].snapshot_column
                )
            trace(
                "OBSERVATION_GRAIN_BOUND: "
                + json.dumps(
                    {
                        "policy": name,
                        "step_id": step["step_id"],
                        "table": policy.table,
                        "grain": retained,
                    }
                )
            )
            if policy.calculation_method:
                for consumer in value["steps"]:
                    if (
                        isinstance(consumer, dict)
                        and consumer.get("action") == "python"
                        and step["step_id"] in consumer.get("input_steps", [])
                    ):
                        calculation = consumer.get("calculation") or {}
                        if not isinstance(calculation, dict):
                            calculation = {}
                        consumer["calculation"] = {
                            **calculation,
                            "kind": analysis_kind or calculation.get("kind", "custom"),
                            "method": policy.calculation_method,
                        }
    return value
