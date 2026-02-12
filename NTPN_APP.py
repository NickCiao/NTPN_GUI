#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:10:47 2025

@author: proxy_loken
"""

import streamlit as st

from ntpn import point_net_utils
from ntpn import point_net
from ntpn import ntpn_utils
from ntpn.state_manager import get_state_manager


def main():
    # Initialize centralized state manager
    state = get_state_manager()

    # Sync legacy session_state keys for backward compatibility
    # This ensures existing pages can still access old session_state keys
    state.sync_to_legacy()

    # Page Management
    landing_page = st.Page('pages/ntpn_landing_page.py', title='NTPN Application')
    import_page = st.Page('pages/import_and_load_page.py', title='Import Models and Data')
    training_page = st.Page('pages/train_model_page.py', title='Define and Train a Model')
    visualisations_page = st.Page('pages/ntpn_visualisations_page.py', title='View Critical Sets and Upper Bounds')
    mapper_page = st.Page('pages/mapper_page.py', title='Topology by Mapper')
    vrtda_page = st.Page('pages/vrtda_page.py', title='Topology by VR-Complexes')
    
    pg = st.navigation([landing_page,import_page,training_page, visualisations_page, mapper_page, vrtda_page])
    
    
    
    pg.run()




if __name__ == "__main__":
    main()