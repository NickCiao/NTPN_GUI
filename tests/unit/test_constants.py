"""Unit tests for ntpn.ntpn_constants module."""

from pathlib import Path

from ntpn import ntpn_constants


class TestConstantsExist:
    """All expected constants are defined."""

    def test_demo_data_paths(self):
        assert hasattr(ntpn_constants, 'demo_st_file')
        assert hasattr(ntpn_constants, 'demo_context_file')
        assert hasattr(ntpn_constants, 'dataset_name')

    def test_data_pipeline_constants(self):
        assert hasattr(ntpn_constants, 'DEFAULT_WINDOW_SIZE')
        assert hasattr(ntpn_constants, 'DEFAULT_WINDOW_STRIDE')
        assert hasattr(ntpn_constants, 'DEFAULT_NUM_NEURONS')
        assert hasattr(ntpn_constants, 'DEFAULT_TEST_SIZE')

    def test_model_architecture_constants(self):
        assert hasattr(ntpn_constants, 'DEFAULT_TRAJECTORY_LENGTH')
        assert hasattr(ntpn_constants, 'DEFAULT_LAYER_WIDTH')
        assert hasattr(ntpn_constants, 'DEFAULT_BATCH_SIZE')
        assert hasattr(ntpn_constants, 'DEFAULT_LEARNING_RATE')
        assert hasattr(ntpn_constants, 'DEFAULT_EPOCHS')
        assert hasattr(ntpn_constants, 'BATCH_NORM_MOMENTUM')
        assert hasattr(ntpn_constants, 'DROPOUT_RATE')
        assert hasattr(ntpn_constants, 'L2_REGULARIZATION')

    def test_visualization_constants(self):
        assert hasattr(ntpn_constants, 'DEFAULT_NUM_CRITICAL_SAMPLES')
        assert hasattr(ntpn_constants, 'DEFAULT_NUM_PLOT_EXAMPLES')
        assert hasattr(ntpn_constants, 'DEFAULT_PLOT_DIMENSIONS')
        assert hasattr(ntpn_constants, 'CRITICAL_SET_LAYER_NAME')

    def test_umap_constants(self):
        assert hasattr(ntpn_constants, 'UMAP_N_NEIGHBORS')
        assert hasattr(ntpn_constants, 'UMAP_MIN_DIST')

    def test_path_constants(self):
        assert hasattr(ntpn_constants, 'MODEL_SAVE_DIR')


class TestConstantTypes:
    """Constants have the correct types."""

    def test_int_constants_are_int(self):
        int_constants = [
            'DEFAULT_WINDOW_SIZE',
            'DEFAULT_WINDOW_STRIDE',
            'DEFAULT_NUM_NEURONS',
            'DEFAULT_TRAJECTORY_LENGTH',
            'DEFAULT_LAYER_WIDTH',
            'DEFAULT_BATCH_SIZE',
            'DEFAULT_EPOCHS',
            'DEFAULT_NUM_CRITICAL_SAMPLES',
            'DEFAULT_NUM_PLOT_EXAMPLES',
            'DEFAULT_PLOT_DIMENSIONS',
            'UMAP_N_NEIGHBORS',
        ]
        for name in int_constants:
            val = getattr(ntpn_constants, name)
            assert isinstance(val, int), f'{name} should be int, got {type(val)}'

    def test_float_constants_are_float(self):
        float_constants = [
            'DEFAULT_TEST_SIZE',
            'DEFAULT_LEARNING_RATE',
            'BATCH_NORM_MOMENTUM',
            'DROPOUT_RATE',
            'L2_REGULARIZATION',
            'UMAP_MIN_DIST',
        ]
        for name in float_constants:
            val = getattr(ntpn_constants, name)
            assert isinstance(val, float), f'{name} should be float, got {type(val)}'

    def test_string_constants_are_str(self):
        str_constants = [
            'CRITICAL_SET_LAYER_NAME',
            'MODEL_SAVE_DIR',
            'demo_st_file',
            'demo_context_file',
            'dataset_name',
        ]
        for name in str_constants:
            val = getattr(ntpn_constants, name)
            assert isinstance(val, str), f'{name} should be str, got {type(val)}'


class TestConstantValues:
    """Constants have sensible values."""

    def test_positive_sizes(self):
        assert ntpn_constants.DEFAULT_WINDOW_SIZE > 0
        assert ntpn_constants.DEFAULT_WINDOW_STRIDE > 0
        assert ntpn_constants.DEFAULT_NUM_NEURONS > 0
        assert ntpn_constants.DEFAULT_TRAJECTORY_LENGTH > 0
        assert ntpn_constants.DEFAULT_LAYER_WIDTH > 0
        assert ntpn_constants.DEFAULT_BATCH_SIZE > 0
        assert ntpn_constants.DEFAULT_EPOCHS > 0

    def test_test_size_in_range(self):
        assert 0.0 < ntpn_constants.DEFAULT_TEST_SIZE < 1.0

    def test_learning_rate_positive(self):
        assert ntpn_constants.DEFAULT_LEARNING_RATE > 0

    def test_dropout_in_range(self):
        assert 0.0 <= ntpn_constants.DROPOUT_RATE < 1.0

    def test_batch_norm_momentum_non_negative(self):
        assert ntpn_constants.BATCH_NORM_MOMENTUM >= 0.0

    def test_l2_regularization_positive(self):
        assert ntpn_constants.L2_REGULARIZATION > 0

    def test_critical_set_layer_name_is_activation(self):
        assert 'activation' in ntpn_constants.CRITICAL_SET_LAYER_NAME


class TestNoHardcodedActivation14:
    """No hardcoded 'activation_14' in service modules (should use constant)."""

    def test_no_hardcoded_in_visualization_service(self):
        source = Path('ntpn/visualization_service.py').read_text()
        assert "'activation_14'" not in source, (
            'visualization_service.py should use ntpn_constants.CRITICAL_SET_LAYER_NAME'
        )

    def test_no_hardcoded_in_ntpn_utils(self):
        source = Path('ntpn/ntpn_utils.py').read_text()
        assert "'activation_14'" not in source, 'ntpn_utils.py should use ntpn_constants.CRITICAL_SET_LAYER_NAME'
