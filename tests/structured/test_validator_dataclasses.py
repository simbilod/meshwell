"""Unit tests for Issue / ValidationResult dataclasses."""
import dataclasses

import pytest

from meshwell.structured.validator import Issue, ValidationResult


def test_issue_is_frozen_dataclass():
    issue = Issue(
        severity="error",
        check="watertight",
        message="hole in volume",
        entities=(("face", 42),),
    )
    assert issue.severity == "error"
    assert issue.check == "watertight"
    assert issue.message == "hole in volume"
    assert issue.entities == (("face", 42),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.check = "other"


def test_validation_result_truthy_when_no_errors():
    result = ValidationResult(errors=(), warnings=())
    assert bool(result) is True

    only_warnings = ValidationResult(
        errors=(),
        warnings=(Issue("warning", "near_duplicates", "1 pair", ()),),
    )
    assert bool(only_warnings) is True


def test_validation_result_falsy_when_errors_present():
    result = ValidationResult(
        errors=(Issue("error", "watertight", "hole", ()),),
        warnings=(),
    )
    assert bool(result) is False


def test_format_report_groups_by_check():
    result = ValidationResult(
        errors=(
            Issue("error", "watertight", "hole at face 42", (("face", 42),)),
            Issue("error", "watertight", "hole at face 51", (("face", 51),)),
            Issue("error", "interface", "T-junction", (("face", 99),)),
        ),
        warnings=(Issue("warning", "near_duplicates", "1 pair", ()),),
    )
    report = result.format_report()
    # Errors before warnings, grouped by check name.
    assert "watertight" in report
    assert "interface" in report
    assert "near_duplicates" in report
    error_idx = report.index("ERRORS")
    warning_idx = report.index("WARNINGS")
    assert error_idx < warning_idx
    assert report.count("[watertight]") == 1
    assert report.count("[interface]") == 1
    assert report.count("[near_duplicates]") == 1


def test_format_report_empty_result_is_clean():
    result = ValidationResult(errors=(), warnings=())
    report = result.format_report()
    assert "no issues" in report.lower()
