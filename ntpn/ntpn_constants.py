#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants and configuration parameters for the NTPN application.

This module defines default file paths, hyperparameters, and configuration
values used across the application. All magic numbers should be defined here.

@author: proxy_loken
"""

# DEMO DATA: A Small dataset of binned neuron ensemble firing and emotional context labels

# Demo session name
dataset_name: str = 'DEMO'

# Binned Neuron firing
# Note: Application will auto-detect and prefer .npz files
demo_st_file: str = 'data/demo_data/raw_stbins.p'

# Emotional Context labels
demo_context_file: str = 'data/demo_data/context_labels.p'

# --- Data Pipeline ---
DEFAULT_WINDOW_SIZE: int = 32
DEFAULT_WINDOW_STRIDE: int = 8
DEFAULT_NUM_NEURONS: int = 11
DEFAULT_TEST_SIZE: float = 0.2

# --- Model Architecture ---
DEFAULT_TRAJECTORY_LENGTH: int = 32
DEFAULT_LAYER_WIDTH: int = 32
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_LEARNING_RATE: float = 0.02
DEFAULT_EPOCHS: int = 5
BATCH_NORM_MOMENTUM: float = 0.0
DROPOUT_RATE: float = 0.3
L2_REGULARIZATION: float = 0.001

# --- Visualization ---
DEFAULT_NUM_CRITICAL_SAMPLES: int = 100
DEFAULT_NUM_PLOT_EXAMPLES: int = 5
DEFAULT_PLOT_DIMENSIONS: int = 3
CRITICAL_SET_LAYER_NAME: str = 'activation_14'

# --- UMAP ---
UMAP_N_NEIGHBORS: int = 30
UMAP_MIN_DIST: float = 0.0

# --- Paths ---
MODEL_SAVE_DIR: str = 'models/'
