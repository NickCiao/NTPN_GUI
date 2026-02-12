#!/usr/bin/env python3
"""
Created on Tue Feb 11 19:05:16 2025

@author: proxy_loken
"""

import streamlit as st

from ntpn import ntpn_constants
from ntpn.data_service import create_train_test, create_trajectories, samples_transform, session_select
from ntpn.model_service import save_model
from ntpn.state_manager import get_state_manager


def dataset_details_import_data():
    # Get state manager
    state = get_state_manager()

    # sidebar section
    st.sidebar.write('Import a Dataset from a file or files')

    # main section
    st.write('# Current Dataset: ', state.data.dataset_name)
    st.write('---')

    # tabs processing
    samples_tab, labels_tab, preprocessing_tab, trajectories_tab, train_test_tab = st.tabs(
        ['Samples', 'Labels', 'Pre-processing', 'Trajectories', 'Training and Validation']
    )

    with samples_tab:
        st.markdown('### Samples')
        st.warning('Not Yet Implemented')

        # TODO: implement form

        samples_load_form = st.form(key='samples_load_form')

        samples_load_button = samples_load_form.form_submit_button(label='Load Samples')
        # TODO dataset name, file format selector, filepicker, additional details
        if samples_load_button:
            pass
            # TODO: call the load function

    with labels_tab:
        st.markdown('### Labels')
        st.warning('Not Yet Implemented')

        # TODO: Implement form

        labels_load_form = st.form(key='labels_load_form')

        labels_load_button = labels_load_form.form_submit_button(label='Load Labels')
        # TODO: file format selector, filepicker, additional details

        if labels_load_button:
            pass
            # TODO: call the load function

    with preprocessing_tab:
        st.markdown('### Dataset Pre-processing')
        preprocess_form = st.form(key='preprocess_form')

        # select sessions from dataset
        preprocess_session_select = preprocess_form.multiselect(
            label='Select Sessions from Dataset', options=[(idx) for idx, item in enumerate(state.data.dataset)]
        )
        # Trim noise category from samples and labels
        preprocess_trim_noise = preprocess_form.checkbox(label='Remove Noise Category', value=True)
        # TODO: add more transfrom options, change to list selector
        preprocess_transform_radio = preprocess_form.radio(label='Transform', options=['Raw', 'Power', 'Standard'])

        preprocess_submit = preprocess_form.form_submit_button(label='Pre-Process Dataset')

        if preprocess_submit:
            session_select(preprocess_session_select, preprocess_trim_noise, state=state)
            samples_transform(preprocess_transform_radio, state=state)

    with trajectories_tab:
        st.markdown('### Trajectories')
        trajectories_form = st.form(key='trajectories_form')

        # TODO: add dynamic bounds for window size and stride
        trajectories_window_size = trajectories_form.number_input(
            label='Trajectory Length', min_value=1, max_value=64, value=ntpn_constants.DEFAULT_WINDOW_SIZE
        )
        trajectories_window_stride = trajectories_form.number_input(
            label='Trajectory Stride', min_value=1, max_value=64, value=ntpn_constants.DEFAULT_WINDOW_STRIDE
        )
        # TODO: Dynamic check for min neurons in selected sessions in dataset
        trajectories_num_neurons = trajectories_form.number_input(
            label='Number of Neurons', min_value=2, max_value=64, value=ntpn_constants.DEFAULT_NUM_NEURONS
        )

        trajectories_submit = trajectories_form.form_submit_button(label='Create Trajectories')

        if trajectories_submit:
            create_trajectories(
                trajectories_window_size, trajectories_window_stride, trajectories_num_neurons, state=state
            )

    with train_test_tab:
        st.markdown('### Training and Validation')

        train_test_form = st.form(key='train_test_form')
        # TODO: Add option for fixed set, batch size, augmentation, etc.
        test_size = train_test_form.number_input(
            label='Proportion for Validation', min_value=0.1, max_value=0.9, value=ntpn_constants.DEFAULT_TEST_SIZE
        )
        train_test_form_submit = train_test_form.form_submit_button(label='Create Training and Validation Sets')

        if train_test_form_submit:
            create_train_test(test_size, state=state)


def dataset_details_import_model():
    # Get state manager
    state = get_state_manager()

    # sidebar section
    st.sidebar.write('Import a Model from a file')

    # main section
    model_name = state.model.model_name
    st.write('# Current Model: ', model_name)
    st.write('---')

    st.markdown(st.markdown('### Import a Model'))
    st.warning('Not Yet Implemented')

    model_import_form = st.form(key='model_import_form')
    # TODO: model name, filepicker, additional params
    model_import_submit = model_import_form.form_submit_button(label='Import Model')
    if model_import_submit:
        pass


def dataset_details_export():
    # Get state manager
    state = get_state_manager()

    # sidebar sectiion
    st.sidebar.write('Export a Model or Data')

    # Main section
    st.markdown('### Export')

    models_tab, datasets_tab = st.tabs(['Export Model', 'Export Data'])

    with models_tab:
        st.markdown('### Model Exporting')

        model_export_form = st.form(key='model_export_form')
        model_export_name = model_export_form.text_input(label='Model Name')
        model_export_submit = model_export_form.form_submit_button(label='Export Model')
        if model_export_submit:
            save_model(model_export_name, state=state)

    with datasets_tab:
        st.markdown('### Data Exporting')
        st.warning('Not Yet Implemented')


def dataset_details():
    # Get state manager
    state = get_state_manager()

    # sidebar section
    st.sidebar.markdown('# Dataset')
    st.sidebar.markdown('Current Dataset: ' + state.data.dataset_name)
    sd_mode_select = st.sidebar.radio(
        label='Select Mode', options=['Import Data', 'Import Model', 'Export'], horizontal=True
    )

    if sd_mode_select == 'Import Data':
        dataset_details_import_data()
    elif sd_mode_select == 'Import Model':
        dataset_details_import_model()
    elif sd_mode_select == 'Export':
        dataset_details_export()


dataset_details()
