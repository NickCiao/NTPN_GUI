"""Regression tests for print statement elimination.

Ensures production code uses logging instead of print() and
model.summary() without print_fn.
"""

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_print_calls(file_path: Path) -> list[tuple[int, str]]:
    """Find print() calls in a file (ignoring comments and strings).

    Returns list of (line_number, line_text) tuples.
    """
    refs = []
    with open(file_path) as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.lstrip()
            # Skip comments
            if stripped.startswith('#'):
                continue
            # Skip string definitions (docstrings, etc.)
            if stripped.startswith(('"""', "'''", '"', "'")):
                continue
            # Match print( at start of statement or after whitespace
            if re.search(r'\bprint\s*\(', stripped):
                refs.append((lineno, line.rstrip()))
    return refs


def _find_bare_model_summary(file_path: Path) -> list[tuple[int, str]]:
    """Find bare model.summary() calls without print_fn argument.

    Returns list of (line_number, line_text) tuples.
    """
    refs = []
    with open(file_path) as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.lstrip()
            # Skip comments and docstrings
            if stripped.startswith('#'):
                continue
            if stripped.startswith(('"""', "'''", '"', "'")):
                continue
            # Match model.summary() without print_fn as a statement
            if re.search(r'\.summary\(\s*\)', stripped):
                refs.append((lineno, line.rstrip()))
    return refs


class TestNoPrintStatements:
    """Production code should use logging, not print()."""

    PRODUCTION_FILES = [
        'ntpn/data_loaders.py',
        'ntpn/data_service.py',
        'ntpn/model_service.py',
        'ntpn/visualization_service.py',
        'ntpn/ntpn_utils.py',
        'ntpn/point_net.py',
        'ntpn/point_net_utils.py',
    ]

    def test_no_print_in_production_code(self):
        """No print() calls in production code."""
        all_refs = {}
        for rel_path in self.PRODUCTION_FILES:
            filepath = PROJECT_ROOT / rel_path
            if filepath.exists():
                refs = _find_print_calls(filepath)
                if refs:
                    all_refs[rel_path] = refs

        assert all_refs == {}, (
            "Found print() calls in production code (use logger instead):\n"
            + "\n".join(
                f"  {name}:\n" + "\n".join(f"    L{n}: {l}" for n, l in refs)
                for name, refs in all_refs.items()
            )
        )


class TestNoBareModelSummary:
    """model.summary() should use print_fn or be wrapped in logger."""

    def test_no_bare_summary_in_point_net(self):
        """point_net.py should not have bare model.summary() calls."""
        filepath = PROJECT_ROOT / 'ntpn' / 'point_net.py'
        refs = _find_bare_model_summary(filepath)
        assert refs == [], (
            f"Found bare model.summary() in point_net.py (use logger or print_fn):\n"
            + "\n".join(f"  L{n}: {l}" for n, l in refs)
        )

    def test_no_bare_summary_in_services(self):
        """Service modules should not have bare model.summary() calls."""
        service_files = [
            'ntpn/model_service.py',
            'ntpn/visualization_service.py',
        ]
        all_refs = {}
        for rel_path in service_files:
            filepath = PROJECT_ROOT / rel_path
            if filepath.exists():
                refs = _find_bare_model_summary(filepath)
                if refs:
                    all_refs[rel_path] = refs

        assert all_refs == {}, (
            "Found bare model.summary() calls:\n"
            + "\n".join(
                f"  {name}:\n" + "\n".join(f"    L{n}: {l}" for n, l in refs)
                for name, refs in all_refs.items()
            )
        )
