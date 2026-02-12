#!/usr/bin/env python3
"""
Created on Tue Feb 11 19:10:27 2025

@author: proxy_loken
"""

import numpy as np
import streamlit as st

from ntpn import ntpn_constants
from ntpn.state_manager import get_state_manager
from ntpn.visualization_service import draw_cs_plots, generate_critical_sets


def generate_sets():
    # Get state manager
    state = get_state_manager()

    # sidebar section
    st.sidebar.write('Generate Critical Sets and Upper-bounds')

    # main section
    if not state.model.train_tensors or not state.model.test_tensors or not state.model.ntpn_model:
        st.warning('Ensure Training and Test Sets, and a NTPN Model are Defined')

    cs_tab, ub_tab = st.tabs(['Critical Sets', 'Upper-bounds'])

    with cs_tab:
        st.markdown('### Critical Sets')

        # Pulls default values for form fields from state of the current dataset

        critical_sets_form = st.form(key='critical_sets_form')

        critical_sets_number_classes = critical_sets_form.number_input(
            label='Number of Classes', min_value=1, max_value=100, value=len(np.unique(state.data.sub_labels))
        )
        critical_sets_number_samples = critical_sets_form.number_input(
            label='Number of Samples', min_value=10, max_value=1000, value=ntpn_constants.DEFAULT_NUM_CRITICAL_SAMPLES
        )

        critical_sets_submit = critical_sets_form.form_submit_button(label='Generate Critical Sets')

        if critical_sets_submit:
            generate_critical_sets(critical_sets_number_classes, critical_sets_number_samples, state=state)

    with ub_tab:
        st.markdown('### Upper-Bounds')
        st.warning('Not Yet Implemented')

    return


def plotting():
    # Get state manager
    state = get_state_manager()

    plotting_options = ['Critical Sets', 'Aligned CS', 'Trajectory Space', 'Upper-bound Shapes']

    plotting_selection = st.sidebar.selectbox(label='Plotting Type', options=plotting_options)

    if plotting_selection == 'Critical Sets':
        cs_plotting_form = st.sidebar.form(key='cs_plotting_form')

        cs_plotting_algo = cs_plotting_form.radio(
            label='Dimensionality Reduction Algorithm', options=['PCA', 'UMAP'], horizontal=True
        )
        cs_plotting_examples = cs_plotting_form.number_input(
            label='Number of Examples', min_value=1, max_value=10, value=ntpn_constants.DEFAULT_NUM_PLOT_EXAMPLES
        )
        cs_plotting_dims = cs_plotting_form.number_input(
            label='Number of Dimensions', min_value=2, max_value=3, value=ntpn_constants.DEFAULT_PLOT_DIMENSIONS
        )
        cs_plotting_classes = cs_plotting_form.number_input(
            label='Number of Classes', min_value=1, max_value=100, value=len(np.unique(state.data.sub_labels))
        )

        cs_plotting_submit = cs_plotting_form.form_submit_button(label='Generate Plots')

        if cs_plotting_submit:
            draw_cs_plots(cs_plotting_algo, cs_plotting_examples, cs_plotting_dims, cs_plotting_classes, state=state)

    elif plotting_selection == 'Aligned CS':
        st.sidebar.warning('Not Yet Implemented')

    else:
        st.sidebar.warning('Not Yet Implemented')

    # display_options = ['Density Tree','Embedding Density','Clusters over Embedding','Smoothed Clusters']
    # top_clustering_display = st.sidebar.selectbox(label='Top Display', key='top_clustering_display', options=display_options)
    # st.session_state.top_clustering_display_options = st.empty()
    # bottom_clustering_display = st.sidebar.selectbox(label='Bottom Display', key='bottom_clustering_display', options=display_options, index=2)
    # st.session_state.bottom_clustering_display_options = st.empty()

    if not state.viz.cs_ub_plots:
        st.warning('Generate a Plot')
    else:
        for fig in state.viz.cs_ub_plots:
            st.pyplot(fig)

    return


def ntpn_visualisations_main():
    # Get state manager
    state = get_state_manager()

    st.sidebar.markdown('# Visualise Critical Sets and Upper-Bounds')
    st.sidebar.markdown('Current Dataset: ' + state.data.dataset_name)

    vs_mode_select = st.sidebar.radio(label='Select Mode', options=['Generate Sets', 'Plotting'], horizontal=True)

    if vs_mode_select == 'Generate Sets':
        generate_sets()
    elif vs_mode_select == 'Plotting':
        plotting()

    # Main section
    if not state.model.train_tensors or not state.model.test_tensors or not state.model.ntpn_model:
        st.warning('Ensure Training and Test Sets, and a NTPN Model are Defined')

    return


ntpn_visualisations_main()
