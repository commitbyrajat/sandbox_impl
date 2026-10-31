"""Static checks for generated Monty code; no SQL execution or domain vocabulary."""

import ast
import builtins
import symtable

from scientific_models import scientific_catalog

SANDBOX_IMPORTS = {
    "math",
    "decimal",
    "random",
    "datetime",
    "json",
    "collections",
    "itertools",
    "functools",
    "bisect",
    "re",
    "typing",
    "asyncio",
    "base64",
    "hashlib",
    "csv",
    "copy",
    "unicodedata",
    "tomllib",
    "time",
}


class PythonPreflightError(ValueError):
    """A rejected proposal must be corrected before starting a Monty session."""


def preflight_python(
    code: str, inputs: dict, *, scientific: bool, analysis_kind: str | None = None
) -> str:
    code = prepare_python(code)
    tree = ast.parse(code)
    catalog = scientific_catalog()
    operations = catalog["operations"] if scientific else {}
    modules, functions = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [entry.name for entry in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".")[0]
                if root not in SANDBOX_IMPORTS and not (
                    root == "scientific" and scientific
                ):
                    raise PythonPreflightError(
                        f"Unsupported sandbox import {name!r}. Available external modules: {['scientific'] if scientific else []}. "
                        "NumPy/SciPy execute in the numerical worker. Pass nested Python lists directly to scientific.kmeans; the worker handles float64 conversion."
                    )
            if isinstance(node, ast.Import):
                for entry in node.names:
                    if entry.name == "scientific":
                        modules[entry.asname or entry.name] = True
            elif node.module == "scientific":
                for entry in node.names:
                    if entry.name not in operations:
                        raise PythonPreflightError(
                            f"Unknown scientific function {entry.name!r}; use {sorted(operations)}"
                        )
                    functions[entry.asname or entry.name] = entry.name

    def operation(call):
        if not isinstance(call, ast.Call):
            return None
        fn = call.func
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and fn.value.id in modules
        ):
            if fn.attr not in operations:
                raise PythonPreflightError(
                    f"Unknown scientific function {fn.attr!r}; use {sorted(operations)}"
                )
            return fn.attr
        return functions.get(fn.id) if isinstance(fn, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = operation(node)
            if (
                name
                and analysis_kind
                and analysis_kind
                not in operations[name].get(
                    "compatible_kinds", [operations[name]["analysis_kind"], "custom"]
                )
            ):
                raise PythonPreflightError(
                    f"scientific.{name} does not implement the published {analysis_kind!r} outcome. Use the published algorithm; similarity clustering does not balance totals."
                )
    # Track only direct top-level bindings; uncertain control-flow is left to Monty.
    outputs, rowsets = {}, {}
    data_is_mapping = True
    for statement in tree.body:
        validated_lookups = set()
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    outputs.pop(target.id, None)
                    value = statement.value
                    if isinstance(value, ast.Await) and (
                        name := operation(value.value)
                    ):
                        outputs[target.id] = operations[name].get("returns", {})
                    if (
                        isinstance(value, ast.Subscript)
                        and isinstance(value.value, ast.Name)
                        and value.value.id == "data"
                        and isinstance(value.slice, ast.Constant)
                        and data_is_mapping
                    ):
                        key = value.slice.value
                        if key not in inputs:
                            raise PythonPreflightError(
                                f"Unknown SQL result {key!r}; use current result IDs {sorted(inputs)}"
                            )
                        validated_lookups.add(id(value))
                        rowsets[target.id] = (
                            set(inputs[key][0]) if inputs[key] else set()
                        )
                        if target.id == "data":
                            data_is_mapping = False
        rowvars = {}
        for node in ast.walk(statement):
            if (
                isinstance(node, (ast.For, ast.comprehension))
                and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Name)
                and node.iter.id in rowsets
            ):
                rowvars[node.target.id] = rowsets[node.iter.id]
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
            ):
                name, key = node.value.id, node.slice.value
                if (
                    name == "data"
                    and not data_is_mapping
                    and isinstance(key, str)
                    and id(node) not in validated_lookups
                ):
                    raise PythonPreflightError(
                        "data now contains a SQL row list; do not index it with a result ID/customer ID. Iterate the rows or bind rows = data[result_id] before reusing the mapping."
                    )
                if name in outputs and outputs[name] and key not in outputs[name]:
                    raise PythonPreflightError(
                        f"Scientific result key {key!r} is unavailable; actual return keys are {sorted(outputs[name])}"
                    )
                if name in rowvars and rowvars[name] and key not in rowvars[name]:
                    raise PythonPreflightError(
                        f"SQL row key {key!r} is unavailable; actual columns are {sorted(rowvars[name])}"
                    )
    return code


def undefined_python_names(code: str) -> set[str]:
    """Find unresolved globals, including references inside lambdas/functions."""
    try:
        table = symtable.symtable(code, "<monty>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python scope: {exc.msg}") from exc
    defined = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    available = defined | set(vars(builtins)) | {"data"}

    def unresolved(scope):
        names = {
            symbol.get_name()
            for symbol in scope.get_symbols()
            if symbol.is_referenced()
            and symbol.is_global()
            and symbol.get_name() not in available
        }
        for child in scope.get_children():
            names.update(unresolved(child))
        return names

    return unresolved(table)


def prepare_python(code: str) -> str:
    """Require SQL-backed input and retain the value of a final print expression."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python: {exc.msg}") from exc
    if not any(
        isinstance(node, ast.Name)
        and node.id == "data"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    ):
        raise ValueError(
            "Read SQL rows from data[result_id]; do not copy rows or amounts into code"
        )
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigns_data = any(
                isinstance(name, ast.Name) and name.id == "data"
                for target in targets
                for name in ast.walk(target)
            )
            reads_input = node.value is not None and any(
                isinstance(name, ast.Name)
                and name.id == "data"
                and isinstance(name.ctx, ast.Load)
                for name in ast.walk(node.value)
            )
            if assigns_data and not reads_input:
                raise ValueError(
                    "data is supplied by the host. Do not replace it with copied rows; read data[result_id] directly."
                )
    # Decimal is a sandbox-native library, never an external host callback.
    # Every checkout is fresh; add its import when the model omitted it.
    if "Decimal" in undefined_python_names(code):
        tree.body.insert(
            0,
            ast.ImportFrom(
                module="decimal", names=[ast.alias(name="Decimal")], level=0
            ),
        )
        code = ast.unparse(ast.fix_missing_locations(tree))
    missing = undefined_python_names(code)
    if missing:
        raise ValueError(
            "Undefined sandbox names: "
            + ", ".join(sorted(missing))
            + ". Import supported modules and define variables in this snippet. "
            + "Read rows from data[result_id]; no state persists between Python steps."
        )
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        call = tree.body[-1].value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "print"
            and len(call.args) == 1
            and not call.keywords
        ):
            tree.body[-1] = ast.Expr(value=call.args[0])
            return ast.unparse(ast.fix_missing_locations(tree))
    return code
