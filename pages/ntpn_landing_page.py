#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:17:04 2025

@author: proxy_loken
"""

import streamlit as st
import numpy as np

from ntpn import ntpn_utils
from ntpn import plotting



def ntpn_landing_main():
    ntpn_utils.initialise_session()
    
    st.sidebar.success('Import a Model or Data to Begin')
    
    
    st.markdown("### Neural Trajectory Point Net")
    st.markdown('---')
    # Front Page Image
    title_image = plotting.load_image('images/ntpn_flowchart_captioned.png')
    ntpn_utils.draw_image(title_image,'','')
    
    st.markdown('---')
    st.markdown("By")
    st.markdown("Adrian Lindsay (2025)")    
    st.markdown('Seamans Lab at the University of British Columbia')
    
    
    
    

ntpn_landing_main()