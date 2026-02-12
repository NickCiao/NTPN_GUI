"""Regression tests for StateManager migration completeness.

These tests ensure that all application code routes state through
StateManager rather than directly accessing st.session_state.
"""

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_session_state_refs(file_path: Path) -> list[tuple[int, str]]:
    """Find non-comment st.session_state references in a file.

    Returns list of (line_number, line_text) tuples.
    """
    refs = []
    with open(file_path) as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.lstrip()
            # Skip comments
            if stripped.startswith('#'):
                continue
            if 'st.session_state' in line:
                refs.append((lineno, line.rstrip()))
    return refs


class TestNoSessionStateInAppCode:
    """Ensure no direct st.session_state usage in application code."""

    def test_ntpn_utils_no_session_state(self):
        """ntpn_utils.py must not reference st.session_state."""
        filepath = PROJECT_ROOT / 'ntpn' / 'ntpn_utils.py'
        refs = _find_session_state_refs(filepath)
        assert refs == [], (
            f"Found {len(refs)} st.session_state ref(s) in ntpn_utils.py:\n"
            + "\n".join(f"  L{n}: {l}" for n, l in refs)
        )

    def test_page_files_no_session_state(self):
        """Page files must not reference st.session_state (excluding comments)."""
        pages_dir = PROJECT_ROOT / 'pages'
        all_refs = {}
        for py_file in sorted(pages_dir.glob('*.py')):
            refs = _find_session_state_refs(py_file)
            if refs:
                all_refs[py_file.name] = refs

        assert all_refs == {}, (
            "Found st.session_state refs in page files:\n"
            + "\n".join(
                f"  {name}:\n" + "\n".join(f"    L{n}: {l}" for n, l in refs)
                for name, refs in all_refs.items()
            )
        )

    def test_state_manager_is_sole_session_state_user(self):
        """Only state_manager.py should reference st.session_state in the ntpn package."""
        ntpn_dir = PROJECT_ROOT / 'ntpn'
        violations = {}
        for py_file in sorted(ntpn_dir.glob('*.py')):
            if py_file.name == 'state_manager.py':
                continue
            refs = _find_session_state_refs(py_file)
            if refs:
                violations[py_file.name] = refs

        assert violations == {}, (
            "Found st.session_state refs outside state_manager.py:\n"
            + "\n".join(
                f"  {name}:\n" + "\n".join(f"    L{n}: {l}" for n, l in refs)
                for name, refs in violations.items()
            )
        )
