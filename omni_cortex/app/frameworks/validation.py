"""
Framework Registry Validation

Validates framework definitions for completeness and correctness.
Run this during development to catch issues early.
"""

import re
import sys
from dataclasses import dataclass

from .registry import FRAMEWORKS, FrameworkCategory, FrameworkDefinition


@dataclass
class ValidationIssue:
    """A validation issue found in a framework definition."""

    framework_name: str
    severity: str  # "error", "warning", "info"
    field: str
    message: str


class FrameworkValidator:
    """Validates framework definitions for completeness and quality."""

    def __init__(self):
        self.issues: list[ValidationIssue] = []

    def validate_all(self) -> list[ValidationIssue]:
        """Validate all frameworks in the registry."""
        self.issues = []

        for name, definition in FRAMEWORKS.items():
            self._validate_framework(name, definition)

        return self.issues

    def _validate_framework(self, name: str, definition: FrameworkDefinition):
        """Validate a single framework definition."""
        # Name validation
        if definition.name != name:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="name",
                    message=f"Definition name '{definition.name}' doesn't match registry key '{name}'",
                )
            )

        if not re.match(r"^[a-z_]+$", name):
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="name",
                    message=f"Name '{name}' must be lowercase with underscores only",
                )
            )

        # Display name validation
        if not definition.display_name:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="display_name",
                    message="Display name is empty",
                )
            )

        # Description validation
        if not definition.description:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="description",
                    message="Description is empty",
                )
            )
        elif len(definition.description) < 20:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="warning",
                    field="description",
                    message=f"Description is very short ({len(definition.description)} chars)",
                )
            )

        # Best_for validation
        if not definition.best_for:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="best_for",
                    message="best_for list is empty",
                )
            )
        elif len(definition.best_for) < 2:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="warning",
                    field="best_for",
                    message=f"best_for has only {len(definition.best_for)} entries (recommend 3+)",
                )
            )

        # Vibes validation
        if not definition.vibes:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="vibes",
                    message="vibes list is empty",
                )
            )
        elif len(definition.vibes) < 5:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="warning",
                    field="vibes",
                    message=f"vibes has only {len(definition.vibes)} entries (recommend 10+)",
                )
            )

        # Check for duplicate vibes
        if definition.vibes:
            unique_vibes = {v.lower() for v in definition.vibes}
            if len(unique_vibes) < len(definition.vibes):
                self.issues.append(
                    ValidationIssue(
                        framework_name=name,
                        severity="warning",
                        field="vibes",
                        message="Contains duplicate vibes (case-insensitive)",
                    )
                )

        # Steps validation
        if definition.steps:
            if len(definition.steps) < 2:
                self.issues.append(
                    ValidationIssue(
                        framework_name=name,
                        severity="info",
                        field="steps",
                        message=f"Only {len(definition.steps)} steps defined (typical: 3-5)",
                    )
                )
        else:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="info",
                    field="steps",
                    message="No steps defined (will use fallback)",
                )
            )

        # Complexity validation
        if definition.complexity not in ["low", "medium", "high"]:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="complexity",
                    message=f"Invalid complexity '{definition.complexity}' (must be low/medium/high)",
                )
            )

        # Task type validation
        valid_task_types = [
            "unknown",
            "debug",
            "implement",
            "refactor",
            "test",
            "optimize",
            "architecture",
            "security",
            "docs",
            "planning",
            "algorithm",
            "adaptive",
            "context",
            "api",
            "compute",
            "quality",
            "iterative",
            "agent",
            "requirements",
        ]
        if definition.task_type not in valid_task_types:
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="warning",
                    field="task_type",
                    message=f"Unusual task_type '{definition.task_type}'",
                )
            )

        # Category validation
        if not isinstance(definition.category, FrameworkCategory):
            self.issues.append(
                ValidationIssue(
                    framework_name=name,
                    severity="error",
                    field="category",
                    message="Category must be a FrameworkCategory enum",
                )
            )

    def get_summary(self) -> dict[str, int]:
        """Get count of issues by severity."""
        summary = {"error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            summary[issue.severity] += 1
        return summary

    def print_report(self):
        """Print a formatted validation report."""
        if not self.issues:
            print("✅ All frameworks passed validation!")
            return

        summary = self.get_summary()
        print("\n📊 Validation Summary:")
        print(f"   Errors: {summary['error']}")
        print(f"   Warnings: {summary['warning']}")
        print(f"   Info: {summary['info']}")
        print(f"   Total: {len(self.issues)}")

        # Group by severity
        for severity in ["error", "warning", "info"]:
            severity_issues = [i for i in self.issues if i.severity == severity]
            if severity_issues:
                icon = "🔴" if severity == "error" else "🟡" if severity == "warning" else "🔵"
                print(f"\n{icon} {severity.upper()}S:")
                for issue in severity_issues:
                    print(f"   {issue.framework_name:30} [{issue.field:15}] {issue.message}")


def validate_registry() -> tuple[bool, list[ValidationIssue]]:
    """
    Validate the framework registry.

    Returns:
        (is_valid, issues) - True if no errors, list of all issues
    """
    validator = FrameworkValidator()
    issues = validator.validate_all()

    has_errors = any(issue.severity == "error" for issue in issues)

    return not has_errors, issues


if __name__ == "__main__":
    # Run validation when executed directly
    validator = FrameworkValidator()
    validator.validate_all()
    validator.print_report()

    summary = validator.get_summary()
    if summary["error"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
