#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:08:26 2025

@author: proxy_loken
"""

import streamlit as st
import numpy as np

from ntpn import ntpn_utils
from ntpn.state_manager import get_state_manager


def define_model():
    # Get state manager
    state = get_state_manager()

    # Sidebar section
    st.sidebar.write('Define a NTPN Model')
    # Main section
    if not state.model.train_tensors or not state.model.test_tensors:
        st.warning('Ensure Training and Test Sets are Defined before Creating/Training a Model')


    # Define Model Form
    define_model_form = st.form(key='define_model_form')
    define_model_form.write('Define a NTPN Model')
    # Params: trajectory length, nubmer of classes, units (layer width), dims(number of neurons)
    # TODO: refine min and max value checks
    define_model_trajectory_length = define_model_form.number_input(label='Trajectory Length', min_value=1, max_value=64, value=32)
    define_model_layer_width = define_model_form.number_input(label='Layer Width', min_value=8, max_value=64, value=32)
    define_model_trajectory_dimension = define_model_form.number_input(label='Trajectory Dimension (Neurons)', min_value=1, max_value=100, value=(state.data.sub_samples.shape)[2])
    define_model_number_classes = define_model_form.number_input(label='Number of Classes', min_value=1, max_value=100, value=len(np.unique(state.data.sub_labels)))

    define_model_submit = define_model_form.form_submit_button(label='Create Model')

    if define_model_submit:
        ntpn_utils.create_model(define_model_trajectory_length, define_model_number_classes, define_model_layer_width, define_model_trajectory_dimension, state=state)

    # Model Summary Display
    # TODO: Container to clean up the visuals
    if state.model.ntpn_model:
        state.model.ntpn_model.summary(print_fn=lambda x: st.text(x))
    else:
        st.warning('Import or Create a Model')


    return



def train_model():
    # Get state manager
    state = get_state_manager()

    # sidebar section
    st.sidebar.write('Train the NTPN Model')
    # Main section
    if not state.model.train_tensors or not state.model.test_tensors:
        st.warning('Ensure Training and Test Sets are Defined before Creating/Training a Model')

    compile_tab, train_tab = st.tabs(['Compile Model', 'Train Model'])

    with compile_tab:
        st.markdown('### Compile')

        compile_model_form = st.form(key='compile_model_form')
        # TODO: Add selectable loss, optmizers, and metrics

        compile_model_form.write('Loss: Sparse Cataegorical')
        compile_model_form.write('Metrics: Sparse Crossentropy')
        compile_model_form.write('Optimizer: ADAM')
        compile_learning_rate = compile_model_form.number_input(label='Learning Rate', min_value=0.001, max_value = 0.9, value=0.02)
        compile_model_submit = compile_model_form.form_submit_button(label='Compile Model')

        if compile_model_submit:
            ntpn_utils.compile_model(learning_rate=compile_learning_rate, state=state)

    with train_tab:
        st.markdown('### Train')

        train_model_form = st.form(key='train_model_form')
        train_model_progress = train_model_form.checkbox(label='Show Progress', value=True)
        train_model_epochs = train_model_form.number_input(label='Number of Epochs', min_value=1, max_value=100, value=5)
        train_model_submit = train_model_form.form_submit_button(label='Train Model')

        if train_model_submit:
            ntpn_utils.train_model(train_model_epochs, view=train_model_progress, state=state)



    return


def train_model_main():
    # Get state manager
    state = get_state_manager()

    # sidebar section
    st.sidebar.markdown("# Define and Train a NTPN Model")
    st.sidebar.markdown("Current Dataset: "+ state.data.dataset_name)
    sd_mode_select = st.sidebar.radio(label='Select Mode', options=['Create Model','Train Model'], horizontal=True)
        
        
    if sd_mode_select == 'Create Model':
        define_model()
    elif sd_mode_select == 'Train Model':
        train_model()   
    
    
    
    return


train_model_main()