#!/usr/bin/env python3
"""
Created on Tue Feb 11 19:17:04 2025

@author: proxy_loken
"""

import streamlit as st

from ntpn import plotting
from ntpn.ntpn_utils import draw_image, initialise_session


def ntpn_landing_main():
    initialise_session()

    st.sidebar.success('Import a Model or Data to Begin')

    st.markdown('### Neural Trajectory Point Net')
    st.markdown('---')
    # Front Page Image
    title_image = plotting.load_image('images/ntpn_flowchart_captioned.png')
    draw_image(title_image, '', '')

    st.markdown('---')
    st.markdown('By')
    st.markdown('Adrian Lindsay (2025)')
    st.markdown('Seamans Lab at the University of British Columbia')


ntpn_landing_main()
