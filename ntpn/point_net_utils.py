"""
Utility functions for PointNet with binned spike counts and labels.

This module is a thin facade that re-exports functions from focused modules
for backward compatibility. All functions are now implemented in:
  - ntpn.data_processing: Data I/O, sampling, preprocessing, transforms, splitting
  - ntpn.analysis: Dimensionality reduction, CS processing, CCA alignment
  - ntpn.plotting: All matplotlib visualization functions

@author: proxy_loken
"""

# Re-export from data_processing (legacy pickle functions preserved for backward compat)
# Re-export from analysis
from ntpn.analysis import calc_cca_trajectories as calc_cca_trajectories
from ntpn.analysis import cs_extract_uniques as cs_extract_uniques
from ntpn.analysis import cs_subsample as cs_subsample
from ntpn.analysis import generate_cca_trajectories as generate_cca_trajectories
from ntpn.analysis import generate_uniques_from_trajectories as generate_uniques_from_trajectories
from ntpn.analysis import generate_upper_sets as generate_upper_sets
from ntpn.analysis import pca_cs_windowed as pca_cs_windowed
from ntpn.analysis import select_closest_trajectories as select_closest_trajectories
from ntpn.analysis import umap_cs_windowed as umap_cs_windowed
from ntpn.analysis import umap_windowed as umap_windowed
from ntpn.data_processing import augment as augment
from ntpn.data_processing import gen_permuted_data as gen_permuted_data
from ntpn.data_processing import load_data_pickle as load_data_pickle
from ntpn.data_processing import load_pickle as load_pickle
from ntpn.data_processing import pow_transform as pow_transform
from ntpn.data_processing import precut_noise as precut_noise
from ntpn.data_processing import remove_noise_cat as remove_noise_cat
from ntpn.data_processing import save_pickle as save_pickle
from ntpn.data_processing import select_samples as select_samples
from ntpn.data_processing import select_samples_cs as select_samples_cs
from ntpn.data_processing import split_balanced as split_balanced
from ntpn.data_processing import std_transform as std_transform
from ntpn.data_processing import subsample_dataset_3d_across as subsample_dataset_3d_across
from ntpn.data_processing import subsample_dataset_3d_within as subsample_dataset_3d_within
from ntpn.data_processing import subsample_neurons as subsample_neurons
from ntpn.data_processing import subsample_neurons_3d as subsample_neurons_3d
from ntpn.data_processing import train_test_gen as train_test_gen
from ntpn.data_processing import train_test_tensors as train_test_tensors
from ntpn.data_processing import unit_sphere as unit_sphere
from ntpn.data_processing import window_projection as window_projection
from ntpn.data_processing import window_projection_segments as window_projection_segments

# Re-export from plotting
from ntpn.plotting import cloud_to_lines as cloud_to_lines
from ntpn.plotting import load_image as load_image
from ntpn.plotting import plot_critical as plot_critical
from ntpn.plotting import plot_critical_umap2D as plot_critical_umap2D
from ntpn.plotting import plot_mean_critical as plot_mean_critical
from ntpn.plotting import plot_mean_upper as plot_mean_upper
from ntpn.plotting import plot_sample as plot_sample
from ntpn.plotting import plot_sample_segmented as plot_sample_segmented
from ntpn.plotting import plot_samples as plot_samples
from ntpn.plotting import plot_target_trajectory as plot_target_trajectory
from ntpn.plotting import plot_target_trajectory_grid as plot_target_trajectory_grid
from ntpn.plotting import plot_upper as plot_upper
from ntpn.plotting import plot_upper_bound as plot_upper_bound
from ntpn.plotting import trajectory_to_lines as trajectory_to_lines
