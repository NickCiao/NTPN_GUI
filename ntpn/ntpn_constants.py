#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants and configuration parameters for the NTPN application.

This module defines default file paths and dataset names for the application.

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


