"""
Plotting functions for NTPN.

This module provides all matplotlib visualization functions extracted from
point_net_utils.py.

No Streamlit dependency.

@author: proxy_loken
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# VISUALISATION FUNCTIONS (Plotting to accompany layer visualisation generation)


# helper for creating a 3d line collection from a trajectory for plotting
def trajectory_to_lines(traj, width=2, alpha=1, cmap='viridis'):

    # reshape trajectory to be (numlines) x (points per line) x (dims: x, y, z)
    points = traj.copy()
    points = points.reshape(-1, 1, 3)
    # then concat into segments
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # generate a norm for the range of values for colour (0:length) if using index
    norm = plt.Normalize(0, traj.shape[0])
    lines = Line3DCollection(segments, cmap=cmap, norm=norm, alpha=alpha)
    # set the values for colour mapping
    lines.set_array(np.arange(traj.shape[0]))
    lines.set_linewidth(width)

    return lines


# helper for creatnig a 3d line collection from an unordered point cloud for plotting
def cloud_to_lines(cloud, width=2, alpha=1, cmap='viridis'):
    # reshape point cloud to be (numlines) x (points per line) x (dims: x, y, z)
    points = cloud.copy()
    points = points.reshape(-1, 1, 3)
    # then concat into segments
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = Line3DCollection(segments, cmap=cmap, alpha=alpha)
    lines.set_linewidth(width)

    return lines


# plot select samples
# TODO: add error on missing mode return
def plot_samples(samples, num_samples, mode='scatter'):

    fig, axs = plt.subplots(num_samples, subplot_kw=dict(projection='3d'))
    col_map = plt.get_cmap('viridis')

    for i in range(num_samples):
        xs = samples[i][:, 0]
        ys = samples[i][:, 1]
        zs = samples[i][:, 2]
        c = np.arange(samples.shape[1])
        if mode == 'scatter':
            sax = axs[i].scatter(xs, ys, zs, c=c, cmap=col_map)
            fig.colorbar(sax, ax=axs)
        elif mode == 'line':
            c = np.linspace(0, 1, samples.shape[1])
            # zip version, deprecated
            # for begin, end in zip(samples[i][:-1], samples[i][1:]):
            # x, y, z = zip(begin, end)
            # colour = c[i]
            # sax = axs[i].plot(x,y,z, c=f'C{c[i]}')
            for j in range(samples.shape[1] - 1):
                sax = axs[i].plot(xs[j : j + 2], ys[j : j + 2], zs[j : j + 2], c=plt.cm.viridis(c[j]))
                # norm = plt.Normalize(0,samples.shape[1])
                # fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=col_map), ax = axs)

        else:
            return 0

    # fig.colorbar(sax, ax = axs)

    return fig


# plot a single sample as a 3d scatter or line, coloured by index in the trajectory
def plot_sample(sample, mode='scatter', trajectory=True):

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    if mode == 'scatter':
        xs = sample[:, 0]
        ys = sample[:, 1]
        zs = sample[:, 2]
        if trajectory:
            c = np.arange(sample.shape[0])
            sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis')
            fig.colorbar(sax, ax=ax)
        else:
            c = zs
            sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    elif mode == 'line':
        if trajectory:
            lc = trajectory_to_lines(sample, width=2, cmap='viridis')
        else:
            lc = cloud_to_lines(sample, width=2, cmap='viridis')
        lines = ax.add_collection(lc)
        ax.set_xlim(np.min(sample[:, 0]), np.max(sample[:, 0]))
        ax.set_ylim(np.min(sample[:, 1]), np.max(sample[:, 1]))
        ax.set_zlim(np.min(sample[:, 2]), np.max(sample[:, 2]))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        if trajectory:
            fig.colorbar(lines, ax=ax)
    else:
        return 0

    return fig


# plot a segmented sample with each point labelled
def plot_sample_segmented(sample, labels, remove_noise=False):

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    if remove_noise:
        mask = labels != -1
        sample = sample[mask]
        labels = labels[mask]

    xs = sample[:, 0]
    ys = sample[:, 1]
    zs = sample[:, 2]

    c = labels

    sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    fig.colorbar(sax, ax=ax)

    return fig


# plot a reference trajectory with comparative trajectories as overlapping points
def plot_target_trajectory(target, comps, lines=False):

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # get line collection for target trajectory
    lc = trajectory_to_lines(target, width=2, cmap='viridis')
    lines = ax.add_collection(lc)
    ax.set_xlim(np.min(target[:, 0]), np.max(target[:, 0]))
    ax.set_ylim(np.min(target[:, 1]), np.max(target[:, 1]))
    ax.set_zlim(np.min(target[:, 2]), np.max(target[:, 2]))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(lines, ax=ax)

    if np.ndim(comps) == 2:
        if lines:
            clc = trajectory_to_lines(comps, width=5, alpha=0.3, cmap='viridis')
            clines = ax.add_collection(clc)
        xs = comps[:, 0]
        ys = comps[:, 1]
        zs = comps[:, 2]
        c = np.arange(comps.shape[0])
        sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis', alpha=0.6)

    else:
        for i in range(comps.shape[0]):
            if lines:
                clc = trajectory_to_lines(comps[i], width=5, alpha=0.3, cmap='viridis')
                clines = ax.add_collection(clc)
            xs = comps[i][:, 0]
            ys = comps[i][:, 1]
            zs = comps[i][:, 2]
            c = np.arange(comps.shape[1])
            sax = ax.scatter(xs, ys, zs, c=c, cmap='viridis', alpha=0.6)
    plt.show()
    return fig


# plot a reference trajectory alongside comparative trajectories
def plot_target_trajectory_grid(target, comps, lines=False):

    fig = plt.figure()
    num_comps = np.shape(comps)[0]
    # define gridspec for figure
    gs = fig.add_gridspec(1, 1, right=0.65)
    # define main axis
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    lc = trajectory_to_lines(target, width=2, cmap='viridis')
    lines = ax.add_collection(lc)
    ax.set_xlim(np.min(target[:, 0]), np.max(target[:, 0]))
    ax.set_ylim(np.min(target[:, 1]), np.max(target[:, 1]))
    ax.set_zlim(np.min(target[:, 2]), np.max(target[:, 2]))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(lines, ax=ax, location='left')

    gs2 = fig.add_gridspec(num_comps, 1, left=0.7)
    for i in range(num_comps):
        ax_comp = fig.add_subplot(gs2[i, 0], projection='3d')
        if lines:
            clc = trajectory_to_lines(comps[i], width=3, alpha=0.6, cmap='viridis')
            clines = ax_comp.add_collection(clc)
        xs = comps[i][:, 0]
        ys = comps[i][:, 1]
        zs = comps[i][:, 2]
        c = np.arange(comps.shape[1])
        sax = ax_comp.scatter(xs, ys, zs, c=c, cmap='viridis', alpha=0.6)
        ax_comp.grid(False)
        ax_comp.set_xticks([])
        ax_comp.set_yticks([])
        ax_comp.set_zticks([])

    plt.show()
    return fig


# plot critical points
def plot_critical(cs, num_samples, samples):
    # TODO: try changing to plotly interactive plots

    fig, axs = plt.subplots(num_samples, 2, sharex=True, sharey=True, subplot_kw=dict(projection='3d'))

    for i in range(num_samples):
        # plot original input points
        ax_orig = axs[i, 0]
        xs = samples[i][:, 0]
        ys = samples[i][:, 1]
        zs = samples[i][:, 2]
        c = np.arange(samples.shape[1])
        orig = ax_orig.scatter(xs, ys, zs, c=c, cmap=plt.get_cmap('viridis'))
        ax_orig.grid(False)

        # plot associated critical points
        ax_crit = axs[i, 1]
        # remove duplicates before plotting
        cs_min = np.unique(cs[i], axis=0)
        xs = cs_min[:, 0]
        ys = cs_min[:, 1]
        zs = cs_min[:, 2]
        # c = np.arange(cs.shape[1])
        crit = ax_crit.scatter(xs, ys, zs)
        ax_crit.grid(False)

    # add plot wide attributes here
    fig.colorbar(orig, ax=axs[:, 0], pad=0.2, shrink=0.6)

    return fig


# plot upper points
def plot_upper(ups, num_samples, samples):
    # TODO: try changing to plotly interactive plots
    fig = plt.figure()
    axs = plt.subplots(num_samples, 2, sharex=True, sharey=True, subplot_kw=dict(projection='3d'))

    for i in num_samples:
        # plot original input points
        ax_orig = axs[i * 2]
        xs = samples[i][:, 0]
        ys = samples[i][:, 1]
        zs = samples[i][:, 2]
        ax_orig.scatter(xs, ys, zs)

        # plot associated critical points
        ax_crit = axs[(i * 2) + 1]
        xs = ups[i][:, 0]
        ys = ups[i][:, 1]
        zs = ups[i][:, 2]
        ax_crit.scatter(xs, ys, zs)

    # add plot wide attributes here

    return fig


# plot an upper bound shape generated by extracting unique(thresholded) points from a set of aligned CS
# TODO: define additional modes for alternate visualisations
def plot_upper_bound(point_array, mode='scatter', colour_mode='height', cmap='viridis'):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    xs = point_array[:, 0]
    ys = point_array[:, 1]
    zs = point_array[:, 2]

    if colour_mode == 'height':
        c = zs
    elif colour_mode == 'distance':
        lengths = np.linalg.norm(point_array, axis=1)
        c = lengths / np.max(lengths)

    if mode == 'scatter':
        ax.scatter(xs, ys, zs, c=c, cmap=cmap)
    elif mode == 'scatter shell':
        # use the distance from 0 to define the alpha of each point(length of the vector described by the point)
        lengths = np.linalg.norm(point_array, axis=1)
        alphas = lengths / np.max(lengths)
        ax.scatter(xs, ys, zs, c=c, cmap=cmap, alpha=alphas)
    else:
        return 0

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig


# plot mean critical set
def plot_mean_critical():
    # TODO
    return


# plot mean upper set
def plot_mean_upper():
    # TODO
    return


def plot_critical_umap2D(cs, num_samples, samples):

    fig, axs = plt.subplots(num_samples, 2, sharex=True, sharey=True)

    for i in range(num_samples):
        # plot original input points
        ax_orig = axs[i, 0]
        xs = samples[i][:, 0]
        ys = samples[i][:, 1]
        ax_orig.scatter(xs, ys)

        # plot associated critical points
        ax_crit = axs[i, 1]
        xs = cs[i][:, 0]
        ys = cs[i][:, 1]
        ax_crit.scatter(xs, ys)

    return fig


# MISC UTILITY FUNCTIONS


def load_image(img_path):
    # TODO: grab frame from video using vid utils
    return skio.imread(img_path)
