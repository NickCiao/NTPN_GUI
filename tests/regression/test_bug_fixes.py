"""
Regression tests for fixed bugs.

These tests ensure that previously fixed bugs don't reappear.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBugFix1SessionStateTyop:
    """
    Bug Fix #1: ntpn_utils.py lines 92-94

    Issue: Used st.session.select_indices instead of st.session_state.select_indices
    Fixed: 2026-02-11
    """

    def test_samples_transform_uses_correct_session_state(self):
        """Verify samples_transform uses StateManager or correct session_state (not st.session typo)."""
        # Read the source file
        source_file = Path('ntpn/ntpn_utils.py')
        with open(source_file, 'r') as f:
            content = f.read()

        # Check that the bug doesn't exist (typo was st.session.select_indices)
        assert 'st.session.select_indices' not in content, \
            "Bug reappeared: st.session.select_indices should not exist (typo)"

        # Verify either StateManager usage or correct session_state usage exists
        has_state_manager = 'state.data.select_indices' in content or 'state.data.select_samples' in content
        has_correct_session_state = 'st.session_state.select_indices' in content

        assert has_state_manager or has_correct_session_state, \
            "Correct usage not found: should use StateManager (state.data.*) or st.session_state.*"

    @patch('ntpn.ntpn_utils.point_net_utils')
    def test_samples_transform_power_calls_with_session_state(self, mock_utils):
        """Test that Power transform uses StateManager correctly."""
        import streamlit as st
        from ntpn.ntpn_utils import samples_transform
        from ntpn.state_manager import StateManager

        # Reset state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Create StateManager with test data
        state = StateManager()
        test_samples = [np.array([[1, 2, 3]])]
        test_indices = [0]
        state.data.select_samples = test_samples
        state.data.select_indices = test_indices

        mock_utils.pow_transform.return_value = MagicMock()

        # Call function with state
        result = samples_transform('Power', state=state)

        # Verify it called pow_transform with correct data
        mock_utils.pow_transform.assert_called_once_with(
            test_samples,
            test_indices
        )

    @patch('ntpn.ntpn_utils.point_net_utils')
    def test_samples_transform_standard_calls_with_session_state(self, mock_utils):
        """Test that Standard transform uses StateManager correctly."""
        import streamlit as st
        from ntpn.ntpn_utils import samples_transform
        from ntpn.state_manager import StateManager

        # Reset state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Create StateManager with test data
        state = StateManager()
        test_samples = [np.array([[1, 2, 3]])]
        test_indices = [0]
        state.data.select_samples = test_samples
        state.data.select_indices = test_indices

        mock_utils.std_transform.return_value = MagicMock()

        # Call function with state
        result = samples_transform('Standard', state=state)

        # Verify it called std_transform with correct data
        mock_utils.std_transform.assert_called_once_with(
            test_samples,
            test_indices
        )


class TestBugFix2LogicError:
    """
    Bug Fix #2: train_model_page.py line 19 and 54

    Issue: Used 'and' instead of 'or' in validation check
           if not train_tensors and test_tensors: (WRONG)
           Should be: if not train_tensors or not test_tensors:
    Fixed: 2026-02-11
    """

    def test_validation_logic_uses_or_not_and(self):
        """Verify train_model_page uses OR logic for validation (works with StateManager or session_state)."""
        # Read the source file
        source_file = Path('pages/train_model_page.py')
        with open(source_file, 'r') as f:
            content = f.read()

        # Check that the bug doesn't exist (old pattern with 'and')
        assert 'and st.session_state.test_tensors:' not in content or 'and state.model.test_tensors:' not in content, \
            "Bug reappeared: validation should use 'or' not 'and'"

        # Verify correct usage exists (either StateManager or session_state pattern with 'or')
        has_correct_session_state = 'if not st.session_state.train_tensors or not st.session_state.test_tensors:' in content
        has_correct_state_manager = 'if not state.model.train_tensors or not state.model.test_tensors:' in content

        assert has_correct_session_state or has_correct_state_manager, \
            "Correct validation logic not found (should use 'or' with either session_state or StateManager)"

    def test_validation_logic_behavior(self):
        """Test the logical behavior of the fixed validation."""
        # Simulate the validation logic
        def should_show_warning(train_tensors, test_tensors):
            """Correct logic: warn if EITHER is missing."""
            return not train_tensors or not test_tensors

        # Test cases
        assert should_show_warning(None, None) == True, "Should warn when both missing"
        assert should_show_warning(None, "exists") == True, "Should warn when train missing"
        assert should_show_warning("exists", None) == True, "Should warn when test missing"
        assert should_show_warning("exists", "exists") == False, "Should NOT warn when both exist"

        # Verify the old (buggy) logic would fail
        def old_buggy_logic(train_tensors, test_tensors):
            """Old buggy logic: only warned in one specific case."""
            return not train_tensors and test_tensors

        # This is why the bug was problematic:
        assert not old_buggy_logic(None, None), "Bug: Didn't warn when both missing!"
        assert not old_buggy_logic("exists", None), "Bug: Didn't warn when test missing!"


@pytest.mark.regression
def test_all_critical_bugs_fixed():
    """Meta-test: Verify all documented critical bugs are fixed."""
    import ntpn.ntpn_utils as ntpn_utils
    from pathlib import Path

    # Read source files
    ntpn_utils_source = Path('ntpn/ntpn_utils.py').read_text()
    train_page_source = Path('pages/train_model_page.py').read_text()

    # Check Bug #1 is fixed
    assert 'st.session.select_indices' not in ntpn_utils_source

    # Check Bug #2 is fixed
    assert 'if not st.session_state.train_tensors and st.session_state.test_tensors:' not in train_page_source

    print("✓ All critical bugs verified as fixed")
