"""
Analysis functions for NTPN.

This module provides dimensionality reduction (UMAP, PCA), critical set
processing, and CCA trajectory alignment utilities extracted from
point_net_utils.py.

No Streamlit dependency.

@author: proxy_loken
"""

import numpy as np
import umap
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from ntpn.data_processing import subsample_neurons

# UMAP Dimensionality reduction functions


# function for mapping windowed samples into reduceed dimension using UMAP
def umap_windowed(data, dims=2, neighbours=30, m_dist=0.0):

    # flatten windows data (3D matrix) to 2D for fitting
    shape = np.shape(data)
    flat_data = np.reshape(data, (-1, shape[2]))

    u_model = umap.UMAP(n_neighbors=neighbours, n_components=dims, min_dist=m_dist).fit(flat_data)

    # transform samples
    mapped_data = np.zeros((shape[0], shape[1], dims))
    for i in range(shape[0]):
        mapped_data[i, :, :] = u_model.transform(data[i, :, :])

    return mapped_data


# function for mapping windowed critical sets and samples into reduced dimension using UMAP
def umap_cs_windowed(cs, samples, dims=2, neighbours=30, m_dist=0.0):

    # flatten windows data (3D matrix) to 2D for fitting
    shape = np.shape(samples)
    flat_data = np.reshape(samples, (-1, shape[2]))

    u_model = umap.UMAP(n_neighbors=neighbours, n_components=dims, min_dist=m_dist).fit(flat_data)

    # transform samples
    mapped_samples = np.zeros((shape[0], shape[1], dims))
    for i in range(shape[0]):
        mapped_samples[i, :, :] = u_model.transform(samples[i, :, :])

    # transform cs
    cs_shape = np.shape(cs)
    mapped_cs = np.zeros((cs_shape[0], cs_shape[1], dims))
    for i in range(cs_shape[0]):
        mapped_cs[i, :, :] = u_model.transform(cs[i, :, :])

    return mapped_cs, mapped_samples


# function for mapping windowed critical sets and samples into reduced dimension using PCA
def pca_cs_windowed(cs, samples, dims=2):

    # flatten windows data (3D matrix) to 2D for fitting
    shape = np.shape(samples)
    flat_data = np.reshape(samples, (-1, shape[2]))

    pca_model = PCA(n_components=dims).fit(flat_data)

    # transform samples
    mapped_samples = np.zeros((shape[0], shape[1], dims))
    for i in range(shape[0]):
        mapped_samples[i, :, :] = pca_model.transform(samples[i, :, :])

    # transform cs
    cs_shape = np.shape(cs)
    mapped_cs = np.zeros((cs_shape[0], cs_shape[1], dims))
    for i in range(cs_shape[0]):
        mapped_cs[i, :, :] = pca_model.transform(cs[i, :, :])

    return mapped_cs, mapped_samples


# CS Processing


# function for extracting unique points from critical set trajectories
# critical sets will contian many duplicates, this will return an in order set
# of unique points
# INPUTS: cs - critical sets (samples x points x neurons), cs_mean - mean unique count
# OUTPUTS: uniques_list - list of unique points for each sample, min_size - smallest unique point set
def cs_extract_uniques(cs, cs_mean):

    uniques_list = []
    min_size = cs_mean
    for i in np.arange(np.shape(cs)[0]):
        unique_points, indxs = np.unique(cs[i, :, :], return_index=True, axis=0)
        unique_order = cs[i, np.sort(indxs), :]
        min_size = np.min([np.shape(unique_order)[0], min_size])
        uniques_list.append(unique_order)

    return uniques_list, min_size


# function for subsampling critical set lists to min_size
# skips samples that have less than min_size points
def cs_subsample(cs_list, min_size):

    subs_list = []
    for i in np.arange(len(cs_list)):
        if np.shape(cs_list[i])[0] < min_size:
            continue
        uniques_sub = subsample_neurons(cs_list[i].T, sample_size=min_size, replace=False)
        subs_list.append(uniques_sub.T)

    return subs_list


# TRAJECTORY ALIGNMENT FUNCTIONS


# function to generate aligned trajectories for each of num_examples raw samples and critical sets
# returns a list of aligned trajectories matrices (one for each example) as well as the original examples
def generate_cca_trajectories(samples, cs, num_examples=5):

    # select num_examples random entries from samples/cs
    inds = np.arange(cs.shape[0])
    selection = np.random.choice(inds, num_examples, False)
    samples_out = samples[selection, :, :]
    cs_out = cs[selection, :, :]

    # for each example, generate a matrix of CCA aligned trajectories (all other samples/cs)
    samples_aligned_trajs = []
    cs_aligned_trajs = []
    samples_examples = []
    cs_examples = []
    for i in range(len(selection)):
        sample_example, sample_aligned = calc_cca_trajectories(samples_out[i], samples, selection[i])
        cs_example, cs_aligned = calc_cca_trajectories(cs_out[i], cs, selection[i])
        samples_aligned_trajs.append(sample_aligned)
        cs_aligned_trajs.append(cs_aligned)
        samples_examples.append(sample_example)
        cs_examples.append(cs_example)

    return samples_examples, cs_examples, samples_aligned_trajs, cs_aligned_trajs


# helper to calculate a set of aligned trajectories to an exemplar using cca
def calc_cca_trajectories(exemplar, samples, ex_index, ndims=3):

    aligned_trajs = np.zeros((samples.shape[0], samples.shape[1], ndims))
    for i in range(samples.shape[0]):
        if i == ex_index:
            # set to an arbitrary value to be ignored in later analysis
            aligned_trajs[i, :, :] = 10
            continue
        cca = CCA(n_components=ndims)
        aligned_trajs[i, :, :], example = cca.fit_transform(samples[i], exemplar)

    return example, aligned_trajs


# function to select num_examples closest (by distance) algined trajectories to the exemplar trajectory
# returns those trajectories
def select_closest_trajectories(exemplar, aligned_trajectories, num_examples):
    # calculate the distances between the examplar and all algined trajectories
    dists = pairwise_distances(
        np.reshape(aligned_trajectories, (aligned_trajectories.shape[0], -1)), np.reshape(exemplar, (1, -1))
    )
    dists = np.squeeze(dists)
    # get the indices of the num_examples smallest distances
    inds = np.argpartition(dists, num_examples)[:num_examples]
    # select those indices from the aligned_trajectories array
    select_trajs = aligned_trajectories[inds]

    return select_trajs


# function to generate a set of unique(or approximately unique via threshold) points from a set of aligned trajectories
# mode: fixed - takes a manually defined threshold, dynamic - calculates a threshold based on the statistics of the trajectories
# returns a list of those unique points
def generate_uniques_from_trajectories(exemplar, trajectories, mode='fixed', threshold=0.0):

    point_set = []
    traj_flat = np.reshape(trajectories.copy(), (-1, trajectories.shape[2]))
    all_points = np.concatenate((exemplar, traj_flat))
    # overlap = np.ones(len(all_points), dtype=bool)
    if mode == 'dynamic':
        # TODO: set threshold calculation to something sensible
        threshold = np.std(all_points)

    i = 0
    while i < len(all_points) - 1:
        # check for dummy values to be removed TODO: change the dummy value to nan or somethign else
        if all_points[i, 0] == 10:
            all_points = np.delete(all_points, i, axis=0)
            continue
        # calculate distances between current point and remaining points in the set
        dists = pairwise_distances(all_points[i + 1 :, :], np.reshape(all_points[i, :], (1, -1)))
        # get indices of dists that are below threshold
        overlap = np.argwhere(np.squeeze(dists) <= threshold)
        # correct overlap indices to apply to all_points array
        overlap = overlap + i
        # remove points from the set based on the overlap indices
        all_points = np.delete(all_points, overlap, axis=0)
        # add current point to point_set list (sanity check: point_set and all_points should match at end)
        point_set.append(all_points[i])
        # increment i
        i += 1

    return point_set, all_points


# all in one function to generate upper-bound sets using aligned(by CCA) CS
# INPUTS: samples - raw trajectories, cs - critical sets (already subsampled ideally), num_sets - number of uppers to create
# upper_size - number of trajectories to build upper-bound shape from, threshold - distance for pruning points
def generate_upper_sets(samples, cs, num_sets=3, upper_size=50, threshold=0.2):

    raw_examples, cs_examples, raw_aligned, cs_aligned = generate_cca_trajectories(samples, cs, num_examples=num_sets)

    raw_uppers = []
    cs_uppers = []

    for i in range(num_sets):
        ind_arr = np.arange(raw_aligned[i].shape[0])
        rand_inds = np.random.choice(ind_arr, upper_size, replace=False)
        raw_aligned_sub = raw_aligned[i][rand_inds, :, :]
        point_list, point_arr = generate_uniques_from_trajectories(
            raw_examples[i], raw_aligned_sub, threshold=threshold
        )
        raw_uppers.append(point_arr)
        ind_arr = np.arange(cs_aligned[i].shape[0])
        rand_inds = np.random.choice(ind_arr, upper_size, replace=False)
        cs_aligned_sub = cs_aligned[i][rand_inds, :, :]
        point_list, point_arr = generate_uniques_from_trajectories(cs_examples[i], cs_aligned_sub, threshold=threshold)
        cs_uppers.append(point_arr)

    return raw_uppers, cs_uppers
