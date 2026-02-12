"""Unit tests for ntpn/analysis.py."""

import numpy as np
import pytest

from ntpn.analysis import (
    calc_cca_trajectories,
    cs_extract_uniques,
    cs_subsample,
    generate_cca_trajectories,
    generate_uniques_from_trajectories,
    generate_upper_sets,
    pca_cs_windowed,
    select_closest_trajectories,
)

# ---------------------------------------------------------------------------
# Fixtures — small synthetic data to keep tests fast
# ---------------------------------------------------------------------------


@pytest.fixture
def small_samples():
    """10 samples, 8 points, 3 dims."""
    np.random.seed(42)
    return np.random.randn(10, 8, 3).astype(np.float32)


@pytest.fixture
def small_cs(small_samples):
    """Critical sets matching small_samples shape."""
    np.random.seed(99)
    return np.random.randn(*small_samples.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


class TestPcaCsWindowed:
    def test_output_shapes_3d(self, small_cs, small_samples):
        mapped_cs, mapped_samples = pca_cs_windowed(small_cs, small_samples, dims=3)
        assert mapped_cs.shape == (10, 8, 3)
        assert mapped_samples.shape == (10, 8, 3)

    def test_output_shapes_2d(self, small_cs, small_samples):
        mapped_cs, mapped_samples = pca_cs_windowed(small_cs, small_samples, dims=2)
        assert mapped_cs.shape == (10, 8, 2)
        assert mapped_samples.shape == (10, 8, 2)

    def test_dims_parameter(self, small_cs, small_samples):
        for d in [1, 2, 3]:
            mapped_cs, mapped_samples = pca_cs_windowed(small_cs, small_samples, dims=d)
            assert mapped_cs.shape[2] == d
            assert mapped_samples.shape[2] == d


# ---------------------------------------------------------------------------
# CS processing
# ---------------------------------------------------------------------------


class TestCsExtractUniques:
    def test_returns_list_and_min_size(self, small_cs):
        # Add duplicates to test unique extraction
        cs = small_cs.copy()
        cs[:, 0, :] = cs[:, 1, :]  # duplicate point 1 -> point 0
        uniques_list, min_size = cs_extract_uniques(cs, cs_mean=8)
        assert isinstance(uniques_list, list)
        assert len(uniques_list) == cs.shape[0]
        assert min_size <= 8

    def test_handles_all_unique(self, small_cs):
        uniques_list, min_size = cs_extract_uniques(small_cs, cs_mean=8)
        assert len(uniques_list) == small_cs.shape[0]
        # All points unique => min_size should be 8
        assert min_size == 8


class TestCsSubsample:
    def test_correct_subsampling(self):
        cs_list = [
            np.random.randn(10, 3),
            np.random.randn(8, 3),
            np.random.randn(12, 3),
        ]
        result = cs_subsample(cs_list, min_size=8)
        for arr in result:
            assert arr.shape == (8, 3)

    def test_skips_small_entries(self):
        cs_list = [
            np.random.randn(5, 3),  # too small
            np.random.randn(10, 3),
        ]
        result = cs_subsample(cs_list, min_size=8)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# CCA alignment
# ---------------------------------------------------------------------------


class TestCalcCcaTrajectories:
    def test_output_shape(self, small_samples):
        exemplar = small_samples[0]
        example, aligned = calc_cca_trajectories(exemplar, small_samples, ex_index=0, ndims=3)
        assert aligned.shape == (10, 8, 3)

    def test_exemplar_index_set_to_dummy(self, small_samples):
        exemplar = small_samples[0]
        _, aligned = calc_cca_trajectories(exemplar, small_samples, ex_index=0, ndims=3)
        # The exemplar index row should be set to 10 (dummy value)
        np.testing.assert_array_equal(aligned[0], np.full((8, 3), 10.0))


class TestSelectClosestTrajectories:
    def test_returns_correct_count(self, small_samples):
        exemplar = small_samples[0]
        aligned = np.random.randn(10, 8, 3)
        result = select_closest_trajectories(exemplar, aligned, num_examples=3)
        assert result.shape[0] == 3
        assert result.shape[1:] == (8, 3)


class TestGenerateUniquesFromTrajectories:
    def test_fixed_mode(self, small_samples):
        exemplar = small_samples[0]
        trajectories = small_samples[1:4]
        point_set, all_points = generate_uniques_from_trajectories(exemplar, trajectories, mode='fixed', threshold=0.0)
        assert isinstance(point_set, list)
        assert len(point_set) > 0

    def test_dynamic_mode(self, small_samples):
        exemplar = small_samples[0]
        trajectories = small_samples[1:4]
        point_set, all_points = generate_uniques_from_trajectories(exemplar, trajectories, mode='dynamic')
        assert isinstance(point_set, list)


class TestGenerateCcaTrajectories:
    def test_returns_4_tuple(self, small_samples, small_cs):
        result = generate_cca_trajectories(small_samples, small_cs, num_examples=3)
        assert len(result) == 4
        samples_ex, cs_ex, samples_aligned, cs_aligned = result
        assert len(samples_ex) == 3
        assert len(cs_ex) == 3
        assert len(samples_aligned) == 3
        assert len(cs_aligned) == 3


class TestGenerateUpperSets:
    @pytest.mark.timeout(60)
    def test_returns_2_tuple(self, small_samples, small_cs):
        raw_uppers, cs_uppers = generate_upper_sets(small_samples, small_cs, num_sets=2, upper_size=5, threshold=0.2)
        assert len(raw_uppers) == 2
        assert len(cs_uppers) == 2


# ---------------------------------------------------------------------------
# UMAP (slow tests)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestUmapWindowed:
    def test_output_shape(self, small_samples):
        from ntpn.analysis import umap_windowed

        result = umap_windowed(small_samples, dims=2, neighbours=5)
        assert result.shape == (10, 8, 2)


@pytest.mark.slow
class TestUmapCsWindowed:
    def test_output_shapes(self, small_cs, small_samples):
        from ntpn.analysis import umap_cs_windowed

        mapped_cs, mapped_samples = umap_cs_windowed(small_cs, small_samples, dims=2, neighbours=5)
        assert mapped_cs.shape == (10, 8, 2)
        assert mapped_samples.shape == (10, 8, 2)
