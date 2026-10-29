"""Deterministic numeric checks for summaries; domain rules remain configuration."""

import ast
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from domain_instructions import DomainProfile
from execution import ExecutionResult

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w|\.\d)")


def numeric(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        return None
    text = str(value).replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        return None
    try:
        result = Decimal(text)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


@dataclass
class SummaryEvidence:
    numbers: set[Decimal] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    aggregates: list[dict[str, Any]] = field(default_factory=list)
    requested_percentages: set[Decimal] = field(default_factory=set)
    method_numbers: set[Decimal] = field(default_factory=set)

    def collect(self, value: Any) -> None:
        if isinstance(value, (date, datetime)):
            day = value.isoformat()[:10]
        elif isinstance(value, str) and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}(?:T.*)?", value
        ):
            day = value[:10]
        else:
            day = None
        if day:
            self.dates.add(day)
            self.numbers.update(Decimal(part) for part in day.split("-"))
            return
        number = numeric(value)
        if number is not None:
            self.numbers.add(number)
            try:
                with localcontext() as context:
                    context.prec = 76
                    self.numbers.add(number.quantize(Decimal("0.01")))
                    self.numbers.add(number.quantize(Decimal("1")))
            except InvalidOperation:
                pass
        elif isinstance(value, dict):
            for item in value.values():
                self.collect(item)
        elif isinstance(value, (list, tuple)):
            self.numbers.add(Decimal(len(value)))
            for item in value:
                self.collect(item)

    def unsupported_values(self, answer: str) -> list[str]:
        problems = [day for day in ISO_DATE.findall(answer) if day not in self.dates]
        text = ISO_DATE.sub(" ", answer)
        # Preserve financial signs when a currency symbol separates minus and amount.
        text = re.sub(r"-\s*(?:INR|₹|Rs\.?)\s*(?=\d)", "-", text, flags=re.IGNORECASE)

        def loss_magnitude(match):
            value = numeric(match["value"])
            return " " if value is not None and -value in self.numbers else match[0]

        text = re.sub(
            r"\bloss(?:es)?(?:\s+(?:of|is|was|totaling))?\s*:?\s*(?:INR|₹|Rs\.?)\s*(?P<value>\d[\d,]*(?:\.\d+)?)",
            loss_magnitude,
            text,
            flags=re.IGNORECASE,
        )

        # Requested confidence levels describe the method, not a monetary result.
        text = re.sub(
            r"(\d+(?:\.\d+)?)\s*%",
            lambda match: (
                " " if numeric(match[1]) in self.requested_percentages else match[0]
            ),
            text,
        )

        def method_parameter(match):
            value = numeric(match["value"])
            if value in self.method_numbers:
                return " "
            return match[0]

        # Method literals are permitted only with method units, never as INR amounts.
        text = re.sub(
            r"(?P<value>\d[\d,]*(?:\.\d+)?)\s+(?:bootstrap\s+)?(?:resamples|resamplings|samples|iterations)\b",
            method_parameter,
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bseed(?:ed)?(?:\s+(?:of|at|with))?\s*[:=]?\s*(?P<value>\d+)\b",
            method_parameter,
            text,
            flags=re.IGNORECASE,
        )

        def ordinal(match):
            value = numeric(match[1])
            if value not in self.numbers and value not in self.method_numbers:
                problems.append(match[1])
            return " "

        text = re.sub(r"(\d+(?:\.\d+)?)(?:st|nd|rd|th)\b", ordinal, text)
        text = re.sub(
            r"(?m)^\s*\d+[.)]\s+", "", text
        )  # Formatting, not a numeric claim.
        problems.extend(
            token
            for token in NUMBER.findall(text)
            if numeric(token) not in self.numbers
        )
        return list(dict.fromkeys(problems))


def prepare_evidence(
    execution: ExecutionResult, profile: DomainProfile, *, question: str = ""
) -> SummaryEvidence:
    evidence = SummaryEvidence()
    evidence.requested_percentages = {
        Decimal(value)
        for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", question)
        if Decimal(0) <= Decimal(value) <= Decimal(100)
    }
    for result_id, result in execution.sql_results.items():
        rows = result.get("rows", [])
        evidence.collect(rows)
        if result.get("truncated"):
            continue
        groups: dict[tuple, list[Decimal]] = {}
        for row in rows:
            partition = tuple(
                (name, str(row[name]))
                for name in profile.summary_partition_fields
                if name in row
            )
            for column in profile.summary_additive_fields:
                number = numeric(row.get(column))
                if number is not None:
                    groups.setdefault((column, partition), []).append(number)
        for (column, partition), values in groups.items():
            with localcontext() as context:
                context.prec = 76
                total = sum(values, Decimal(0))
            evidence.collect(total)
            evidence.aggregates.append(
                {
                    "result_id": result_id,
                    "field": column,
                    "partition": dict(partition),
                    "sum": total,
                }
            )
    for run in execution.python_runs:
        evidence.collect(run.get("output"))
        try:
            tree = ast.parse(run.get("code", ""))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
                value = numeric(node.value)
                if value is not None:
                    evidence.method_numbers.add(value)
    return evidence
