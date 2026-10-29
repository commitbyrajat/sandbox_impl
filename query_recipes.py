"""Scoped, configured SQL recipes for well-defined deterministic requests."""

import hashlib
import re
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent


class RecipeParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value_group: str
    direction_group: str


class SqlRecipeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_pattern: str
    parameters: dict[str, RecipeParameter]
    directions: dict[str, int]
    sources: list[str]
    template_path: str
    title: str
    reason: str
    expected_output: str
    validation_checks: list[str]
    invalid_input_field: str | None = None
    invalid_input_message: str = "Scenario inputs are invalid"

    @field_validator("request_pattern")
    @classmethod
    def valid_pattern(cls, value):
        re.compile(value, re.IGNORECASE)
        return value

    @field_validator("template_path")
    @classmethod
    def local_path(cls, value):
        if (
            Path(value).is_absolute()
            or ".." in Path(value).parts
            or not value.endswith(".sql")
        ):
            raise ValueError("Recipe templates must be repository-relative SQL paths")
        return value


class SqlRecipeCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe_id: str
    parameters: dict[str, Decimal]
    template_sha256: str

    @field_validator("parameters")
    @classmethod
    def finite_parameters(cls, value):
        if any(
            not x.is_finite()
            or abs(x) >= Decimal("10000")
            or x.as_tuple().exponent < -4
            for x in value.values()
        ):
            raise ValueError(
                "Recipe parameters must be finite decimal numbers with at most four fractional digits and magnitude below 10000"
            )
        return value


def template_bytes(spec: SqlRecipeDefinition) -> bytes:
    path = (ROOT / spec.template_path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Recipe template escapes repository root")
    return path.read_bytes()


def match_sql_recipe(
    question: str, recipes: dict[str, SqlRecipeDefinition]
) -> SqlRecipeCall | None:
    matches = []
    for key, spec in recipes.items():
        match = re.fullmatch(spec.request_pattern, question.strip(), re.IGNORECASE)
        if match is None:
            continue
        parameters = {
            name: Decimal(match[p.value_group])
            * spec.directions[match[p.direction_group].lower()]
            for name, p in spec.parameters.items()
        }
        matches.append(
            SqlRecipeCall(
                recipe_id=key,
                parameters=parameters,
                template_sha256=hashlib.sha256(template_bytes(spec)).hexdigest(),
            )
        )
    if len(matches) > 1:
        raise ValueError("Question matches multiple configured SQL recipes")
    return matches[0] if matches else None


def render_sql_recipe(
    call: SqlRecipeCall, recipes: dict[str, SqlRecipeDefinition]
) -> str:
    call = SqlRecipeCall.model_validate(call.model_dump())
    if call.recipe_id not in recipes:
        raise ValueError("SQL recipe is not configured for this workspace")
    spec = recipes[call.recipe_id]
    raw = template_bytes(spec)
    if hashlib.sha256(raw).hexdigest() != call.template_sha256:
        raise ValueError("SQL recipe changed after planning; publish a new plan")
    text = raw.decode()
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", text))
    if set(call.parameters) != set(spec.parameters) or placeholders != set(
        spec.parameters
    ):
        raise ValueError("SQL recipe parameters do not match the configured template")
    for name, value in call.parameters.items():
        text = text.replace("{{" + name + "}}", format(value, "f"))
    return text
