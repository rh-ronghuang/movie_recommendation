# Import necessary libraries
import streamlit as st
from pages import data_overview, feature_enhancement, recommendation_abstract, recommendation_demo

# Create a sidebar with page selection
page = st.sidebar.selectbox('Select a Page', ['Data Overview', 'Test Train Split Overview & Feature Enhancement', 'Recommendation Abstract', 'Recommendation Demo'])

# Conditionally display the selected page
if page == 'Data Overview':
    data_overview.show_page()
elif page == 'Test Train Split Overview & Feature Enhancement':
    feature_enhancement.show_page()
elif page == 'Recommendation Abstract':
    recommendation_abstract.show_page()
elif page == 'Recommendation Demo':
    recommendation_demo.show_page()
