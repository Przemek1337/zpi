from __future__ import annotations

from typing import Any, Mapping

from pydantic import ValidationError

_MAX_INPUT_PREVIEW = 80


def describe_validation_error(error: ValidationError) -> str:
    return "\n".join(_describe_issue(issue) for issue in error.errors())


def _describe_issue(issue: Mapping[str, Any]) -> str:
    location = ".".join(str(part) for part in issue["loc"]) or "<root>"
    return f"  - {location}: {issue['msg']} (got {_preview(issue.get('input'))})"


def _preview(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _MAX_INPUT_PREVIEW else f"{text[:_MAX_INPUT_PREVIEW]}..."
