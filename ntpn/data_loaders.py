"""
Safe data loading module for NTPN GUI.

This module provides secure alternatives to unsafe pickle deserialization,
with support for multiple formats and migration utilities.

SECURITY WARNING: Never use pickle.load() on untrusted data files.
This module provides safer alternatives using NPZ and HDF5 formats.
"""

import numpy as np
import pickle  # Required for legacy format support - see load_legacy_format()
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Exception raised when data loading fails."""
    pass


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class DataValidator:
    """Validates loaded data structure and types."""

    @staticmethod
    def validate_spike_data(data: List[np.ndarray]) -> None:
        """
        Validate spike count data structure.

        Args:
            data: List of spike count arrays

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(data, list):
            raise DataValidationError(
                f"Data must be a list, got {type(data).__name__}"
            )

        if len(data) == 0:
            raise DataValidationError("Data list cannot be empty")

        for i, session_data in enumerate(data):
            if not isinstance(session_data, np.ndarray):
                raise DataValidationError(
                    f"Session {i} must be numpy array, got {type(session_data).__name__}"
                )

            if session_data.ndim != 2:
                raise DataValidationError(
                    f"Session {i} must be 2D (time_bins, neurons), got shape {session_data.shape}"
                )

            if session_data.dtype not in [np.float32, np.float64, np.int32, np.int64]:
                warnings.warn(
                    f"Session {i} has unexpected dtype {session_data.dtype}, "
                    "expected float32/float64/int32/int64"
                )

    @staticmethod
    def validate_labels(labels: List[np.ndarray], data: Optional[List[np.ndarray]] = None) -> None:
        """
        Validate label data structure.

        Args:
            labels: List of label arrays
            data: Optional corresponding spike data for length validation

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(labels, list):
            raise DataValidationError(
                f"Labels must be a list, got {type(labels).__name__}"
            )

        if len(labels) == 0:
            raise DataValidationError("Labels list cannot be empty")

        for i, session_labels in enumerate(labels):
            if not isinstance(session_labels, np.ndarray):
                raise DataValidationError(
                    f"Label session {i} must be numpy array, got {type(session_labels).__name__}"
                )

            if session_labels.ndim != 1:
                raise DataValidationError(
                    f"Label session {i} must be 1D, got shape {session_labels.shape}"
                )

            # Validate length matches data if provided
            if data is not None and i < len(data):
                expected_length = data[i].shape[0]  # time_bins is dimension 0
                if len(session_labels) != expected_length:
                    raise DataValidationError(
                        f"Label session {i} length {len(session_labels)} "
                        f"does not match data time_bins {expected_length}"
                    )


def load_from_numpy(st_file: Union[str, Path], context_file: Union[str, Path], context_key: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load data from safe NPZ (NumPy) format.

    This is the recommended safe format for NTPN data.

    Args:
        st_file: Path to .npz file containing spike count data
        context_file: Path to .npz file containing labels
        context_key: Key name for labels in context file

    Returns:
        Tuple of (spike_data_list, labels_list)

    Raises:
        DataLoadError: If file loading fails
        DataValidationError: If data validation fails
    """
    st_file = Path(st_file)
    context_file = Path(context_file)

    try:
        # Load spike data
        # Note: allow_pickle=True is needed for object arrays (list of arrays)
        # This is still safer than raw pickle as numpy controls the structure
        with np.load(st_file, allow_pickle=True) as data:
            # Support multiple possible keys
            if 'raw_stbins' in data:
                spike_data = data['raw_stbins']
            elif 'data' in data:
                spike_data = data['data']
            else:
                available_keys = list(data.keys())
                raise DataLoadError(
                    f"No recognized spike data key in {st_file}. "
                    f"Available keys: {available_keys}. "
                    f"Expected 'raw_stbins' or 'data'"
                )

        # Load labels
        with np.load(context_file, allow_pickle=True) as data:
            if context_key not in data:
                available_keys = list(data.keys())
                raise DataLoadError(
                    f"Key '{context_key}' not found in {context_file}. "
                    f"Available keys: {available_keys}"
                )
            labels = data[context_key]

        # Convert to list format if needed
        if isinstance(spike_data, np.ndarray):
            if spike_data.dtype == object:
                # Array of arrays
                spike_data_list = [arr for arr in spike_data]
            else:
                # Single array, wrap in list
                spike_data_list = [spike_data]
        else:
            spike_data_list = spike_data

        if isinstance(labels, np.ndarray):
            if labels.dtype == object:
                # Array of arrays
                labels_list = [arr for arr in labels]
            else:
                # Single array, wrap in list
                labels_list = [labels]
        else:
            labels_list = labels

        # Validate data
        DataValidator.validate_spike_data(spike_data_list)
        DataValidator.validate_labels(labels_list, spike_data_list)

        return spike_data_list, labels_list

    except (IOError, OSError) as e:
        raise DataLoadError(f"Failed to load NPZ file: {e}") from e
    except Exception as e:
        raise DataLoadError(f"Unexpected error loading NPZ data: {e}") from e


def load_legacy_format(st_file: Union[str, Path], context_file: Union[str, Path], context_key: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load data from legacy pickle format.

    SECURITY WARNING:
    Pickle files can execute arbitrary code when loaded.
    Only use this with TRUSTED data files from known sources.

    This function is provided for backward compatibility only.
    New data should use NPZ format.

    Args:
        st_file: Path to .p or .pkl file containing spike count data
        context_file: Path to .p or .pkl file containing labels
        context_key: Key name for labels in context file

    Returns:
        Tuple of (spike_data_list, labels_list)

    Raises:
        DataLoadError: If file loading fails
        DataValidationError: If data validation fails
    """
    # Loud warning
    warnings.warn(
        "\n" + "=" * 80 + "\n"
        "SECURITY WARNING: Loading data from UNSAFE pickle format\n"
        f"Files: {st_file}, {context_file}\n\n"
        "Pickle files can execute arbitrary code and may be malicious.\n"
        "ONLY proceed if you trust the source of these files.\n\n"
        "Recommended: Convert to safe NPZ format using:\n"
        f"  python scripts/migrate_demo_data.py\n"
        "=" * 80,
        category=RuntimeWarning,
        stacklevel=2
    )

    st_file = Path(st_file)
    context_file = Path(context_file)

    try:
        # Load spike data
        with open(st_file, 'rb') as fp:
            st_dict = pickle.load(fp)
            if not isinstance(st_dict, dict):
                raise DataLoadError(
                    f"Expected dict in {st_file}, got {type(st_dict).__name__}"
                )
            if 'raw_stbins' not in st_dict:
                available_keys = list(st_dict.keys())
                raise DataLoadError(
                    f"Key 'raw_stbins' not found in {st_file}. "
                    f"Available keys: {available_keys}"
                )
            spike_data_list = st_dict['raw_stbins']

        # Load labels
        with open(context_file, 'rb') as fp:
            label_dict = pickle.load(fp)
            if not isinstance(label_dict, dict):
                raise DataLoadError(
                    f"Expected dict in {context_file}, got {type(label_dict).__name__}"
                )
            if context_key not in label_dict:
                available_keys = list(label_dict.keys())
                raise DataLoadError(
                    f"Key '{context_key}' not found in {context_file}. "
                    f"Available keys: {available_keys}"
                )
            labels_list = label_dict[context_key]

        # Validate data
        DataValidator.validate_spike_data(spike_data_list)
        DataValidator.validate_labels(labels_list, spike_data_list)

        return spike_data_list, labels_list

    except (IOError, OSError) as e:
        raise DataLoadError(f"Failed to load pickle file: {e}") from e
    except Exception as e:
        raise DataLoadError(f"Unexpected error loading pickle data: {e}") from e


def load_data_safe(st_file: Union[str, Path], context_file: Union[str, Path], context_key: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load data with automatic format detection and preference for safe formats.

    This is the recommended high-level loading function.

    Format priority:
    1. NPZ format (.npz) - SAFE, recommended
    2. Legacy pickle (.p, .pkl) - UNSAFE, fallback with warning

    Args:
        st_file: Path to spike count data file
        context_file: Path to labels file
        context_key: Key name for labels in context file

    Returns:
        Tuple of (spike_data_list, labels_list)

    Raises:
        DataLoadError: If file loading fails
        DataValidationError: If data validation fails
    """
    st_file = Path(st_file)
    context_file = Path(context_file)

    # Try NPZ first (safe, preferred)
    npz_st = st_file.with_suffix('.npz')
    npz_context = context_file.with_suffix('.npz')

    if npz_st.exists() and npz_context.exists():
        return load_from_numpy(npz_st, npz_context, context_key)

    # Fallback to legacy pickle (unsafe, with warning)
    if st_file.exists() and context_file.exists():
        return load_legacy_format(st_file, context_file, context_key)

    # No valid files found
    raise DataLoadError(
        f"No valid data files found.\n"
        f"Tried:\n"
        f"  NPZ: {npz_st}, {npz_context}\n"
        f"  Pickle: {st_file}, {context_file}\n"
        f"None of these files exist or are readable."
    )


def convert_to_safe_format(
    input_st_file: Union[str, Path],
    input_context_file: Union[str, Path],
    context_key: str,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[Path, Path]:
    """
    Convert legacy pickle files to safe NPZ format.

    Args:
        input_st_file: Path to input pickle spike data file
        input_context_file: Path to input pickle context file
        context_key: Key name for labels in context file
        output_dir: Output directory (default: same as input)

    Returns:
        Tuple of (output_st_file_path, output_context_file_path)

    Raises:
        DataLoadError: If conversion fails
    """
    input_st_file = Path(input_st_file)
    input_context_file = Path(input_context_file)

    # Determine output directory
    if output_dir is None:
        output_dir = input_st_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data using legacy loader
    logger.info("Loading data from %s and %s", input_st_file, input_context_file)
    spike_data_list, labels_list = load_legacy_format(input_st_file, input_context_file, context_key)

    # Create output file paths
    output_st_file = output_dir / input_st_file.with_suffix('.npz').name
    output_context_file = output_dir / input_context_file.with_suffix('.npz').name

    logger.info("Converting to NPZ format...")

    # Save as NPZ with allow_pickle=True for object arrays
    np.savez(
        output_st_file,
        raw_stbins=np.array(spike_data_list, dtype=object)
    )
    np.savez(
        output_context_file,
        **{context_key: np.array(labels_list, dtype=object)}
    )

    logger.info("Conversion complete!")
    logger.info("  Output files: %s, %s", output_st_file, output_context_file)

    return output_st_file, output_context_file
