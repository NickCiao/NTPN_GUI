"""
Data processing functions for NTPN.

This module provides data I/O, sampling, preprocessing, transforms,
and train/test splitting utilities extracted from point_net_utils.py.

No Streamlit dependency.

@author: proxy_loken
"""

import pickle
from typing import Any

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, power_transform

try:
    from gtda.time_series import SlidingWindow
except ImportError:
    # Simple SlidingWindow replacement if giotto-tda is not available
    class SlidingWindow:
        def __init__(self, size=3, stride=1):
            self.size = size
            self.stride = stride

        def fit_transform_resample(self, X, y=None):
            """Create sliding windows from time series data.

            Expects X with shape (n_features, n_timepoints) and transposes internally
            to match giotto-tda behavior of (n_timepoints, n_features).
            Returns windows of shape (n_windows, window_size, n_features).
            """
            X = np.array(X)
            if y is not None:
                y = np.array(y)

            # Assume input is (neurons/features, time) - transpose to (time, features)
            if X.ndim == 2:
                X = X.T

            n_timepoints = X.shape[0]
            n_samples = (n_timepoints - self.size) // self.stride + 1

            # Create windows: shape (n_windows, window_size, n_features)
            windows = np.array([X[i * self.stride : i * self.stride + self.size] for i in range(n_samples)])

            if y is not None:
                # Take the last label in each window
                y_windows = np.array([y[i * self.stride + self.size - 1] for i in range(n_samples)])
                return windows, y_windows
            return windows


from sklearn.model_selection import train_test_split


# Pickling/Unpickling Data (legacy API - prefer NPZ format for new code)
def save_pickle(data: Any, filename: str) -> None:

    pickle.dump(data, open(filename, 'wb'))


def load_pickle(filename: str) -> Any:
    with open(filename, 'rb') as fp:
        l_data = pickle.load(fp)

    return l_data


# DATA LOADING FUNCTIONS


# Function to load dataset and labelset
# Now uses safe data loading with automatic format detection
# Prefers NPZ format, falls back to pickle with security warning
# TODO: branch to accomodate multilabelset loading
def load_data_pickle(st_file: str, label_file: str, label_name: str) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
    """
    Load spike data and labels from files.

    IMPORTANT: This function now uses safe data loading that prefers NPZ format.
    If you have old pickle files, run: python scripts/migrate_demo_data.py

    Args:
        st_file: Path to spike data file (.npz or .p)
        label_file: Path to label file (.npz or .p)
        label_name: Key name for labels in label file

    Returns:
        Tuple of (spike_data_list, label_list)
    """
    from ntpn.data_loaders import load_data_safe

    return load_data_safe(st_file, label_file, label_name)


# SAMPLING / SUBSAMPLING DATA FUNCTIONS


# function to extract a set of samples rom a given class for testing/visualisation
# INPUT: X: data array(3D, select along axis=0), Y: class labels,  num_samples: number of samples to extract (should match batch size),
# class_label: label of class to extract
def select_samples(
    X: npt.NDArray, Y: npt.NDArray, num_samples: int, class_label: int, return_index: bool = False
) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:

    inds = np.argwhere(class_label == Y)
    inds = np.squeeze(inds)
    selection = np.random.choice(inds, num_samples, False)
    samples = X[selection, :, :]
    if return_index:
        return samples, selection
    return samples


# function to extract a random subset of samples and accompanying critical sets
def select_samples_cs(cs, samples, num_samples, return_index=False):

    inds = np.arange(samples.shape[0])
    selection = np.random.choice(inds, num_samples, False)
    samples_out = samples[selection, :, :]
    cs_out = cs[selection, :, :]
    if return_index:
        return cs_out, samples_out, selection
    return cs_out, samples_out


# subsample neurons within each session, then stack each session and labels into a single dataset
def subsample_dataset_3d_within(stbin_list, labels_list, sample_size=32, replace=True):

    out_list = []
    for i in range(len(stbin_list)):
        out = subsample_neurons_3d(stbin_list[i], sample_size, replace)
        out_list.append(out)

    out_neurons = np.concatenate(out_list)
    out_labels = np.concatenate(labels_list)

    return out_neurons, out_labels


# subsample neurons across sessions. Assemble a dataset of fixed size (per label)
# Assumes the labels for each session contain the same set of labels
def subsample_dataset_3d_across(stbin_list, labels_list, num_samples=2000, sample_size=32, replace=True):

    labels = np.unique(labels_list[0])
    all_neurons = []
    all_labels = []
    for i in labels:
        # select all samples for each label from all sessions
        curr_neurons = []
        for j in range(len(stbin_list)):
            curr_samples = select_samples(stbin_list[j], labels_list[j], num_samples, i)
            curr_neurons.append(curr_samples)

        curr_neurons = np.concatenate(curr_neurons, axis=1)
        all_neurons.append(curr_neurons)
        all_labels.append(np.zeros(num_samples) + i)

    all_neurons = np.concatenate(all_neurons)
    all_labels = np.concatenate(all_labels)

    out_neurons = subsample_neurons_3d(all_neurons, sample_size, replace)

    return out_neurons, all_labels


def subsample_neurons(stbin, sample_size=32, replace=True):

    ind_arr = np.arange(np.size(stbin, 1))

    rand_inds = np.random.choice(ind_arr, sample_size, replace)

    out_neurons = stbin[:, rand_inds].copy()

    return out_neurons


def subsample_neurons_3d(stbin, sample_size=32, replace=True):

    ind_arr = np.arange(np.size(stbin, 1))

    rand_inds = np.random.choice(ind_arr, sample_size, replace)

    out_neurons = stbin[:, rand_inds, :].copy()

    return out_neurons


# DATA PERMUTATION / AUGMENTATION


def gen_permuted_data(neurons, labels, direction='width', samples=10):

    out_neurons = []

    if direction == 'width':
        for i in range(samples):
            sample_neurons = subsample_neurons(neurons, sample_size=32, replace=True)
            out_neurons.append(sample_neurons)

        neuron_array = np.hstack(out_neurons)
        labels_array = labels

    elif direction == 'length':
        for i in range(samples):
            sample_neurons = subsample_neurons(neurons, sample_size=32, replace=True)
            out_neurons.append(sample_neurons)

        neuron_array = np.vstack(out_neurons)
        labels_array = np.repeat(labels, samples, axis=0)

    return neuron_array, labels_array


# Function to augment data points by adding jitter
# TODO
def augment(points, label):
    # TODO change params for neural data
    points += tf.random.uniform(points.shape, -0.05, 0.05, dtype=tf.float64)

    points = tf.random.shuffle(points)

    return points, label


# function to generate a unit sphere of points to test upper bound shapes for classes
def unit_sphere():

    lim = 0.577
    all_points = []

    for x in np.linspace(-lim, lim, 16):
        for y in np.linspace(-lim, lim, 16):
            for z in np.linspace(-lim, lim, 16):
                all_points.append([x, y, z])

    all_points = np.array(all_points)
    # reshape to match input dims: 16^3 = 4096 points, 4096 / 32 = 128 batches
    all_points = np.reshape(all_points, (128, 32, 3))

    return all_points


# DATA PREPROCESSING / TRANSFORMS


# Helper pre-processing function to cut out noise periods from labelsets and
# accompanying numerical data
# labels and num_data need to be the same length, asumes labels are a vector
# assumes noise is labelled as -1 in labels
def precut_noise(labels, num_data, noise_label=-1):

    # get mask of non-noise labels
    mask = labels != noise_label
    outlabels = labels[mask].copy()

    # check for dim of numerical data
    if num_data.ndim == 1:
        outdata = num_data[mask].copy()
    else:
        outdata = num_data[mask, :].copy()

    return outlabels, outdata


# Remove noise from the portion of the dataset/labels in selection
# selection should be a list of indices of sessions to include
def remove_noise_cat(stbin_list, label_list, selection, noise_label=-1):

    if len(selection) <= 1:
        label_cut, st_cut = precut_noise(label_list, stbin_list, noise_label)
        return st_cut, label_cut

    stcut_list = []
    label_cut_list = []
    for i in selection:
        label_cut, st_cut = precut_noise(label_list[i], stbin_list[i], noise_label)
        stcut_list.append(st_cut)
        label_cut_list.append(label_cut)

    return stcut_list, label_cut_list


def pow_transform(stbin_list, selection):

    if len(selection) <= 1:
        X_pow = power_transform(stbin_list)
        return X_pow

    X_pow_list = []
    for i in selection:
        X_pow_list.append(power_transform(stbin_list[i]))

    return X_pow_list


def std_transform(stbin_list, selection):

    if len(selection) <= 1:
        X_std = StandardScaler().fit_transform(stbin_list)
        return X_std

    X_std_list = []
    for i in selection:
        std = StandardScaler()
        X_std = std.fit_transform(stbin_list[i])
        X_std_list.append(X_std)

    return X_std_list


# Project the 1-d time series of stbin into an N-d time series by windowing adjecent time bins, also adjusts the labels
# The window size is equivalent to the output dimension, ie a window of 3 produces a '3D' projection
# stride can be adjusted to subsample during the projection (stride of 1 will have N-1 overlap between windows)
def window_projection(
    stbin_list: list[npt.NDArray],
    label_list: list[npt.NDArray],
    selection: list[int],
    window_size: int = 3,
    stride: int = 1,
) -> tuple[list[npt.NDArray], list[npt.NDArray]]:

    if len(selection) <= 1:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(stbin_list, label_list)
        return X_sw, Y_sw

    X_sw_list = []
    Y_sw_list = []
    for i in selection:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(stbin_list[i], label_list[i])
        X_sw = np.swapaxes(X_sw, 1, 2)
        X_sw_list.append(X_sw)
        Y_sw_list.append(Y_sw)

    return X_sw_list, Y_sw_list


# helper for behavioural segmentation. Windows the behavioural labels to match the stbin windowing in the function above
def window_projection_segments(
    behav_list: list[npt.NDArray],
    label_list: list[npt.NDArray],
    selection: list[int],
    window_size: int = 3,
    stride: int = 1,
) -> tuple[list[npt.NDArray], list[npt.NDArray]]:

    if len(selection) <= 1:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(behav_list, label_list)
        return X_sw, Y_sw

    X_sw_list = []
    Y_sw_list = []
    for i in selection:
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, Y_sw = SW.fit_transform_resample(behav_list[i], label_list[i])

        X_sw_list.append(X_sw)
        Y_sw_list.append(Y_sw)

    return X_sw_list, Y_sw_list


# DATASET SPLITTING


# Helper to call sklearn's train_test_split to generate training and test matrices
def train_test_gen(Xs, Ys, test_size=0.2, stratify=False):

    if stratify:
        X_train, X_val, Y_train, Y_val = train_test_split(Xs, Ys, test_size=test_size, stratify=Ys)
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(Xs, Ys, test_size=test_size)

    return X_train, X_val, Y_train, Y_val


# helper to convert train, test datasets into tensorflow tensors for use in NN training/testing
# uses tensorflow functions to convert to tensors, batch, and augment the training set (optional)
def train_test_tensors(X_train, X_val, Y_train, Y_val, augment=True, batch_size=8):

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    # augment and shuffle
    # TODO: update augmentation
    if augment:
        train_dataset = train_dataset.shuffle(len(X_train)).map(augment).batch(batch_size, drop_remainder=True)
    else:
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.shuffle(len(X_val)).batch(batch_size, drop_remainder=True)

    return train_dataset, test_dataset


def split_balanced(data, target, test_size=0.2):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size < 1:
        n_test = np.round(len(target) * test_size)
    else:
        n_test = test_size
    n_train = max(0, len(target) - n_test)
    n_train_per_class = max(1, int(np.floor(n_train / len(classes))))
    n_test_per_class = max(1, int(np.floor(n_test / len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class + n_test_per_class) > np.sum(target == cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class / (n_train_per_class + n_test_per_class) * np.sum(target == cl)))
            ixs.append(
                np.r_[
                    np.random.choice(np.nonzero(target == cl)[0][:splitix], n_train_per_class),
                    np.random.choice(np.nonzero(target == cl)[0][splitix:], n_test_per_class),
                ]
            )
        else:
            ixs.append(
                np.random.choice(np.nonzero(target == cl)[0], n_train_per_class + n_test_per_class, replace=False)
            )

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class : (n_train_per_class + n_test_per_class)] for x in ixs])

    X_train = data[ix_train, :, :]
    X_test = data[ix_test, :, :]
    y_train = target[ix_train]
    y_test = target[ix_test]

    return X_train, X_test, y_train, y_test
