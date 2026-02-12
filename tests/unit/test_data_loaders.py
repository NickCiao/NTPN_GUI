"""Unit tests for safe data loaders."""

import pytest
import numpy as np
from pathlib import Path
import warnings

from ntpn.data_loaders import (
    DataValidator,
    DataValidationError,
    DataLoadError,
    load_from_numpy,
    load_legacy_format,
    load_data_safe,
    convert_to_safe_format
)


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_spike_data_valid(self, sample_spike_data):
        """Test validation passes for valid spike data."""
        data, _ = sample_spike_data
        DataValidator.validate_spike_data(data)

    def test_validate_spike_data_not_list(self):
        """Test validation fails if data is not a list."""
        with pytest.raises(DataValidationError, match="must be a list"):
            DataValidator.validate_spike_data(np.array([[1, 2, 3]]))

    def test_validate_spike_data_empty_list(self):
        """Test validation fails for empty list."""
        with pytest.raises(DataValidationError, match="cannot be empty"):
            DataValidator.validate_spike_data([])

    def test_validate_spike_data_not_array(self):
        """Test validation fails if session is not numpy array."""
        with pytest.raises(DataValidationError, match="must be numpy array"):
            DataValidator.validate_spike_data([[1, 2, 3]])

    def test_validate_spike_data_wrong_dimensions(self):
        """Test validation fails for wrong array dimensions."""
        bad_data = [np.array([1, 2, 3])]
        with pytest.raises(DataValidationError, match="must be 2D"):
            DataValidator.validate_spike_data(bad_data)

    def test_validate_labels_valid(self, sample_spike_data):
        """Test validation passes for valid labels."""
        data, labels = sample_spike_data
        DataValidator.validate_labels(labels, data)

    def test_validate_labels_not_list(self):
        """Test validation fails if labels is not a list."""
        with pytest.raises(DataValidationError, match="must be a list"):
            DataValidator.validate_labels(np.array([1, 2, 3]))

    def test_validate_labels_empty_list(self):
        """Test validation fails for empty list."""
        with pytest.raises(DataValidationError, match="cannot be empty"):
            DataValidator.validate_labels([])

    def test_validate_labels_wrong_dimensions(self):
        """Test validation fails for wrong label dimensions."""
        bad_labels = [np.array([[1, 2], [3, 4]])]
        with pytest.raises(DataValidationError, match="must be 1D"):
            DataValidator.validate_labels(bad_labels)

    def test_validate_labels_length_mismatch(self, sample_spike_data):
        """Test validation fails when label length doesn't match data."""
        data, labels = sample_spike_data
        bad_labels = [np.array([0, 1]) for _ in labels]
        with pytest.raises(DataValidationError, match="does not match data time_bins"):
            DataValidator.validate_labels(bad_labels, data)


class TestLoadFromNumpy:
    """Tests for load_from_numpy function."""

    def test_load_from_numpy_success(self, tmp_path, sample_spike_data):
        """Test successful loading from NPZ files."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.npz"
        context_file = tmp_path / "context.npz"

        np.savez(st_file, raw_stbins=np.array(data, dtype=object))
        np.savez(context_file, context_label=np.array(labels, dtype=object))

        # Suppress dtype warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_data, loaded_labels = load_from_numpy(st_file, context_file, 'context_label')

        assert len(loaded_data) == len(data)
        assert len(loaded_labels) == len(labels)
        for orig, loaded in zip(data, loaded_data):
            np.testing.assert_array_equal(orig, loaded)

    def test_load_from_numpy_alternative_key(self, tmp_path, sample_spike_data):
        """Test loading with alternative 'data' key."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.npz"
        context_file = tmp_path / "context.npz"

        np.savez(st_file, data=np.array(data, dtype=object))
        np.savez(context_file, context_label=np.array(labels, dtype=object))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_data, loaded_labels = load_from_numpy(st_file, context_file, 'context_label')
            assert len(loaded_data) == len(data)

    def test_load_from_numpy_missing_file(self, tmp_path):
        """Test error when file doesn't exist."""
        st_file = tmp_path / "nonexistent.npz"
        context_file = tmp_path / "also_nonexistent.npz"

        with pytest.raises(DataLoadError, match="Failed to load NPZ file"):
            load_from_numpy(st_file, context_file, 'context_label')

    def test_load_from_numpy_missing_key(self, tmp_path, sample_spike_data):
        """Test error when expected key is missing."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.npz"
        context_file = tmp_path / "context.npz"

        np.savez(st_file, wrong_key=np.array(data, dtype=object))
        np.savez(context_file, context_label=np.array(labels, dtype=object))

        with pytest.raises(DataLoadError, match="No recognized spike data key"):
            load_from_numpy(st_file, context_file, 'context_label')

    def test_load_from_numpy_validates_data(self, tmp_path):
        """Test that loaded data is validated (wrapped in DataLoadError)."""
        st_file = tmp_path / "spike_data.npz"
        context_file = tmp_path / "context.npz"

        bad_data = [np.array([1, 2, 3])]
        labels = [np.array([0, 1, 2])]

        np.savez(st_file, raw_stbins=np.array(bad_data, dtype=object))
        np.savez(context_file, context_label=np.array(labels, dtype=object))

        # Validation errors are wrapped in DataLoadError
        with pytest.raises(DataLoadError, match="must be 2D"):
            load_from_numpy(st_file, context_file, 'context_label')


class TestLoadLegacyFormat:
    """Tests for load_legacy_format function."""

    def test_load_legacy_format_shows_warning(self, tmp_path, sample_spike_data):
        """Test that legacy loader shows security warning."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.p"
        context_file = tmp_path / "context.p"

        import pickle
        with open(st_file, 'wb') as f:
            pickle.dump({'raw_stbins': data}, f)
        with open(context_file, 'wb') as f:
            pickle.dump({'context_label': labels}, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_data, loaded_labels = load_legacy_format(st_file, context_file, 'context_label')

            assert len(w) == 1
            assert "SECURITY WARNING" in str(w[0].message)
            assert "pickle" in str(w[0].message).lower()

    def test_load_legacy_format_success(self, tmp_path, sample_spike_data):
        """Test successful loading from pickle files."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.p"
        context_file = tmp_path / "context.p"

        import pickle
        with open(st_file, 'wb') as f:
            pickle.dump({'raw_stbins': data}, f)
        with open(context_file, 'wb') as f:
            pickle.dump({'context_label': labels}, f)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded_data, loaded_labels = load_legacy_format(st_file, context_file, 'context_label')

        assert len(loaded_data) == len(data)
        assert len(loaded_labels) == len(labels)


class TestLoadDataSafe:
    """Tests for load_data_safe function."""

    def test_load_data_safe_prefers_npz(self, tmp_path, sample_spike_data):
        """Test that safe loader prefers NPZ over pickle."""
        data, labels = sample_spike_data

        st_file_pkl = tmp_path / "spike_data.p"
        context_file_pkl = tmp_path / "context.p"
        st_file_npz = tmp_path / "spike_data.npz"
        context_file_npz = tmp_path / "context.npz"

        import pickle
        with open(st_file_pkl, 'wb') as f:
            pickle.dump({'raw_stbins': data}, f)
        with open(context_file_pkl, 'wb') as f:
            pickle.dump({'context_label': labels}, f)

        modified_data = [arr * 2 for arr in data]
        np.savez(st_file_npz, raw_stbins=np.array(modified_data, dtype=object))
        np.savez(context_file_npz, context_label=np.array(labels, dtype=object))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_data, loaded_labels = load_data_safe(st_file_pkl, context_file_pkl, 'context_label')
            
            # Check that no SECURITY WARNING was issued (NPZ was used, not pickle)
            security_warnings = [warning for warning in w if "SECURITY WARNING" in str(warning.message)]
            assert len(security_warnings) == 0

        np.testing.assert_array_equal(loaded_data[0], modified_data[0])

    def test_load_data_safe_fallback_to_pickle(self, tmp_path, sample_spike_data):
        """Test fallback to pickle when NPZ not available."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.p"
        context_file = tmp_path / "context.p"

        import pickle
        with open(st_file, 'wb') as f:
            pickle.dump({'raw_stbins': data}, f)
        with open(context_file, 'wb') as f:
            pickle.dump({'context_label': labels}, f)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_data, loaded_labels = load_data_safe(st_file, context_file, 'context_label')
            assert len(w) == 1
            assert "SECURITY WARNING" in str(w[0].message)

    def test_load_data_safe_no_files(self, tmp_path):
        """Test error when no valid files exist."""
        st_file = tmp_path / "nonexistent.p"
        context_file = tmp_path / "also_nonexistent.p"

        with pytest.raises(DataLoadError, match="No valid data files found"):
            load_data_safe(st_file, context_file, 'context_label')


class TestConvertToSafeFormat:
    """Tests for convert_to_safe_format function."""

    def test_convert_to_safe_format_success(self, tmp_path, sample_spike_data):
        """Test successful conversion from pickle to NPZ."""
        data, labels = sample_spike_data

        st_file = tmp_path / "spike_data.p"
        context_file = tmp_path / "context.p"

        import pickle
        with open(st_file, 'wb') as f:
            pickle.dump({'raw_stbins': data}, f)
        with open(context_file, 'wb') as f:
            pickle.dump({'context_label': labels}, f)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_st, output_context = convert_to_safe_format(
                st_file, context_file, 'context_label', output_dir=tmp_path
            )

        assert output_st.exists()
        assert output_context.exists()
        assert output_st.suffix == '.npz'

        # Load and verify (suppress dtype warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_data, loaded_labels = load_from_numpy(output_st, output_context, 'context_label')
            assert len(loaded_data) == len(data)
            assert len(loaded_labels) == len(labels)


@pytest.mark.integration
def test_full_workflow(tmp_path, sample_spike_data):
    """Test complete workflow: save pickle, convert to NPZ, load safely."""
    data, labels = sample_spike_data

    st_file = tmp_path / "spike_data.p"
    context_file = tmp_path / "context.p"

    import pickle
    with open(st_file, 'wb') as f:
        pickle.dump({'raw_stbins': data}, f)
    with open(context_file, 'wb') as f:
        pickle.dump({'context_label': labels}, f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        convert_to_safe_format(st_file, context_file, 'context_label')

    # Load data and check for security warnings (not dtype warnings)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded_data, loaded_labels = load_data_safe(st_file, context_file, 'context_label')
        
        # Should have no SECURITY warnings (NPZ was used)
        security_warnings = [warning for warning in w if "SECURITY WARNING" in str(warning.message)]
        assert len(security_warnings) == 0

    assert len(loaded_data) == len(data)
    assert len(loaded_labels) == len(labels)
    for orig, loaded in zip(data, loaded_data):
        np.testing.assert_array_equal(orig, loaded)
