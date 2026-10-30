"""Bind configured relationship keys to business meaning and review SQL join scopes."""

import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope


class JoinBinding(BaseModel):
    relationship_id: str
    left_table: str
    right_table: str
    column_pairs: list[tuple[str, str]] = Field(min_length=1)
    business: dict[str, Any] | None = None


def configured_joins(sources: dict, meanings: dict) -> list[JoinBinding]:
    """Use registry keys, retaining legacy equalities when no structured key exists."""
    bindings = {}
    for source_id, source in sources.items():
        for name, key in source.relationship_keys.items():
            if (
                key.left_table not in source.tables
                or key.right_table not in source.tables
            ):
                continue
            if name in bindings:
                raise ValueError(
                    f"Duplicate relationship ID in selected sources: {name}"
                )
            bindings[name] = JoinBinding(
                relationship_id=name,
                **key.model_dump(),
                business=meanings[name].model_dump() if name in meanings else None,
            )
        for index, text in enumerate(source.relationships):
            match = re.fullmatch(
                r"\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)(?:\s*\(.*\))?\s*", text
            )
            if not match:
                continue
            left, a, right, b = match.groups()
            if left not in source.tables or right not in source.tables:
                continue
            if any(
                {binding.left_table, binding.right_table} == {left, right}
                for binding in bindings.values()
            ):
                continue
            name = f"{source_id}_legacy_{index}"
            bindings[name] = JoinBinding(
                relationship_id=name,
                left_table=left,
                right_table=right,
                column_pairs=[(a, b)],
            )
    return list(bindings.values())


def validate_join_metadata(bindings, schemas):
    for binding in bindings:
        for table, names in (
            (binding.left_table, [a for a, _ in binding.column_pairs]),
            (binding.right_table, [b for _, b in binding.column_pairs]),
        ):
            if table not in schemas or set(names) - set(schemas[table].names):
                raise ValueError(
                    f"Invalid relationship metadata for {binding.relationship_id}: keys must exist in the selected source schemas"
                )
        if binding.business:
            for side, table in (
                ("left_grain", binding.left_table),
                ("right_grain", binding.right_table),
            ):
                if set(binding.business[side]) - set(schemas[table].names):
                    raise ValueError(
                        f"Invalid business grain for {binding.relationship_id}: {side} must reference existing fields"
                    )


def review_joins(tree, bindings: Sequence[JoinBinding]):
    """Validate resolvable keys; explicitly report computed/opaque lineage as unverified."""
    graph = {}
    table_graph = {}
    for binding in bindings:
        table_graph.setdefault(binding.left_table, []).append(
            (binding.right_table, binding)
        )
        table_graph.setdefault(binding.right_table, []).append(
            (binding.left_table, binding)
        )
        if len(binding.column_pairs) == 1:
            a, b = binding.column_pairs[0]
            left, right = (binding.left_table, a), (binding.right_table, b)
            graph.setdefault(left, {})[right] = binding
            graph.setdefault(right, {})[left] = binding

    def bridge_guidance(left_table, right_table):
        pending = [(left_table, [], {left_table})]
        while pending:
            table, path, visited = pending.pop(0)
            if table == right_table and path:
                return (
                    " Configured table relationship path (join through the intermediate tables, preserving observation grain): "
                    + " then ".join(
                        " AND ".join(
                            f"{key.left_table}.{a} = {key.right_table}.{b}"
                            for a, b in key.column_pairs
                        )
                        for key in path
                    )
                    + ". Carry the observation key through the input CTE and aggregate at the required grain before joining its results."
                )
            if len(path) < 4:
                pending.extend(
                    (neighbor, [*path, key], visited | {neighbor})
                    for neighbor, key in table_graph.get(table, [])
                    if neighbor not in visited
                )
        return ""

    def connected(a, b):
        pending, visited = [(a, [])], set()
        while pending:
            item, path = pending.pop(0)
            if item == b:
                return path
            if item not in visited:
                visited.add(item)
                pending.extend(
                    (neighbor, [*path, binding])
                    for neighbor, binding in graph.get(item, {}).items()
                )
        return None

    def origin(scope, column, visited=None):
        visited = set() if visited is None else visited
        identity = (id(scope), column.sql())
        if identity in visited:
            return None
        visited = visited | {identity}
        candidates = (
            [scope.selected_sources[column.table][1]]
            if column.table in scope.selected_sources
            else []
        )
        found = []
        for source in candidates:
            if isinstance(source, exp.Table):
                found.append((source.name, column.name))
            elif isinstance(source, Scope):
                for index, projection in enumerate(source.expression.selects):
                    name = (
                        source.outer_columns[index]
                        if index < len(source.outer_columns)
                        else projection.alias_or_name
                    )
                    value = (
                        projection.this
                        if isinstance(projection, exp.Alias)
                        else projection
                    )
                    if name != column.name:
                        continue
                    if isinstance(value, exp.Column):
                        result = origin(source, value, visited)
                        if result:
                            found.append(result)
        return found[0] if len(found) == 1 else None

    def unique_group(scope, alias, columns):
        source = scope.selected_sources.get(alias, (None, None))[1]
        if not isinstance(source, Scope) or not isinstance(
            source.expression, exp.Select
        ):
            return False
        select = source.expression
        group = select.args.get("group")
        if (
            group is None
            or not group.expressions
            or any(
                group.args.get(key)
                for key in ("rollup", "cube", "grouping_sets", "all")
            )
        ):
            return False
        projections = {
            (
                source.outer_columns[index]
                if index < len(source.outer_columns)
                else projection.alias_or_name
            ): (projection.this if isinstance(projection, exp.Alias) else projection)
            for index, projection in enumerate(select.selects)
        }
        key_expressions = {
            projections[name].sql(normalize=True)
            for name in columns
            if name in projections
        }
        for value in group.expressions:
            if isinstance(value, exp.Literal) and value.is_int:
                position = int(value.this) - 1
                if not 0 <= position < len(select.selects):
                    return False
                value = select.selects[position]
                if isinstance(value, exp.Alias):
                    value = value.this
            if isinstance(value, exp.Column) and not value.table:
                value = projections.get(value.name, value)
            if value.sql(normalize=True) not in key_expressions:
                return False
        return True

    def equalities(node):
        if isinstance(node, exp.Paren):
            yield from equalities(node.this)
        elif isinstance(node, exp.And):
            yield from equalities(node.this)
            yield from equalities(node.expression)
        elif isinstance(node, exp.EQ):
            yield node

    issues, reviewed, warnings = [], [], []
    for scope_index, scope in enumerate(traverse_scope(tree)):
        joins = scope.expression.args.get("joins", [])
        if not joins:
            continue
        predicates = []
        for join in joins:
            on = join.args.get("on")
            required_equalities = list(equalities(on))
            predicates.extend(required_equalities)
            has_column_key = any(
                isinstance(node.this, exp.Column)
                and isinstance(node.expression, exp.Column)
                and node.this.table != node.expression.table
                for node in required_equalities
            )
            if on is not None and on.find(exp.Or) is not None and not has_column_key:
                issues.append(
                    {
                        "kind": "join_condition_not_guaranteed",
                        "scope": f"query_{scope_index + 1}",
                        "reference": on.sql(),
                        "repair": "Required relationship equalities must hold for every joined row. Put the complete keys in conjunctive ON conditions; an OR branch must not bypass them.",
                    }
                )
            if join.args.get("on") is None:
                warnings.append(
                    {
                        "kind": "unverified_join",
                        "scope": scope_index,
                        "reason": "Cross/natural/implicit join needs explicit business grain review.",
                    }
                )
            elif not has_column_key:
                warnings.append(
                    {
                        "kind": "unverified_join",
                        "scope": scope_index,
                        "reason": "No unconditional column equality establishes this join's relationship keys. Review its predicate and business grain.",
                    }
                )
        where = scope.expression.args.get("where")
        if where is not None:
            predicates.extend(equalities(where.this))
        pairs = {}
        for equality in predicates:
            left, right = equality.this, equality.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                left_aliases = {
                    column.table
                    for column in left.find_all(exp.Column)
                    if column.find_ancestor(exp.Select) is scope.expression
                }
                right_aliases = {
                    column.table
                    for column in right.find_all(exp.Column)
                    if column.find_ancestor(exp.Select) is scope.expression
                }
                if (
                    left_aliases
                    and right_aliases
                    and len(left_aliases | right_aliases) > 1
                ):
                    warnings.append(
                        {"kind": "unverified_lineage", "predicate": equality.sql()}
                    )
                continue
            if left.table == right.table:
                continue
            a, b = origin(scope, left), origin(scope, right)
            if a is None or b is None:
                warnings.append(
                    {"kind": "unverified_lineage", "predicate": equality.sql()}
                )
                continue
            aliases = tuple(sorted((left.table, right.table)))
            pairs.setdefault(aliases, []).append((left, right, a, b))
        for aliases, provided in pairs.items():
            a, b = provided[0][2:]
            direct = [
                key
                for key in bindings
                if {key.left_table, key.right_table} == {a[0], b[0]}
            ]
            expressions = {(x, y) for _, _, x, y in provided} | {
                (y, x) for _, _, x, y in provided
            }
            matched = []
            for key in direct:
                expected = {
                    ((key.left_table, x), (key.right_table, y))
                    for x, y in key.column_pairs
                }
                if expected.issubset(expressions):
                    matched.append(key)
            basis = "configured_keys"
            if direct and not matched:
                column_names = {
                    alias: {
                        column.name
                        for left, right, _, _ in provided
                        for column in (left, right)
                        if column.table == alias
                    }
                    for alias in aliases
                }
                if all(
                    unique_group(scope, alias, column_names[alias]) for alias in aliases
                ) and any(
                    ((key.left_table, x), (key.right_table, y)) in expressions
                    for key in direct
                    for x, y in key.column_pairs
                ):
                    matched = direct
                    basis = "unique_aggregate_grain"
                    warnings.append(
                        {
                            "kind": "coarser_aggregate_join",
                            "scope": scope_index,
                            "reason": "Both grouped inputs expose unique keys at the coarser grain; review aggregated measure meaning.",
                        }
                    )
                else:
                    required = [
                        " AND ".join(
                            f"{key.left_table}.{x} = {key.right_table}.{y}"
                            for x, y in key.column_pairs
                        )
                        for key in direct
                    ]
                    meaning = [key.business for key in direct if key.business]
                    issues.append(
                        {
                            "kind": "invalid_join_keys",
                            "scope": f"query_{scope_index+1}",
                            "reference": " AND ".join(
                                f"{left.sql()} = {right.sql()}"
                                for left, right, _, _ in provided
                            ),
                            "repair": "Use a complete configured relationship: "
                            + " OR ".join(required)
                            + ". Business guidance: "
                            + str(meaning),
                        }
                    )
                    continue
            paths = [connected(x, y) for _, _, x, y in provided] if not direct else []
            if not direct and any(path is None for path in paths):
                issues.append(
                    {
                        "kind": "undeclared_join",
                        "scope": f"query_{scope_index+1}",
                        "reference": " AND ".join(
                            f"{left.sql()} = {right.sql()}"
                            for left, right, _, _ in provided
                        ),
                        "repair": "Use configured relationship keys. Similar identifier names or types do not establish a business relationship."
                        + bridge_guidance(a[0], b[0]),
                    }
                )
                continue
            for key in matched:
                reviewed.append(
                    {
                        "scope": scope_index,
                        "aliases": aliases,
                        "relationship_id": key.relationship_id,
                        "verification_basis": basis,
                        "business": key.business,
                    }
                )
                if key.business and key.business["cardinality"] != "one_to_one":
                    warnings.append(
                        {
                            "kind": "aggregation_grain_review",
                            "relationship_id": key.relationship_id,
                            "cardinality": key.business["cardinality"],
                            "rules": key.business["aggregation_rules"],
                            "reason": "Configured base-grain cardinality is guidance, not proof of the query result cardinality.",
                        }
                    )
            if not direct:
                traversed = {
                    binding.relationship_id: binding
                    for path in paths
                    for binding in path
                }
                reviewed.append(
                    {
                        "scope": scope_index,
                        "aliases": aliases,
                        "verification_basis": (
                            "configured_key_path" if traversed else "same_source_key"
                        ),
                        "path_relationships": [
                            binding.model_dump() for binding in traversed.values()
                        ],
                    }
                )
                warnings.append(
                    {
                        "kind": "aggregation_grain_review",
                        "reason": "A shared identifier path does not prove that a fact-to-fact join preserves measures. Apply the grain and aggregation rules from each configured path relationship.",
                        "rules": [
                            rule
                            for binding in traversed.values()
                            if binding.business
                            for rule in binding.business["aggregation_rules"]
                        ],
                    }
                )
    return issues, reviewed, warnings
