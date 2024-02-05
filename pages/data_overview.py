# pages/data_overview.py
import streamlit as st
import pandas as pd

def load_data():
    movies_df = pd.read_csv('dataset/movies.csv')
    ratings_df = pd.read_csv('dataset/ratings.csv')
    return movies_df, ratings_df

def show_page():
    st.title('Data Overview')
    
    movies_df, ratings_df = load_data()
    
    # Display the head of each dataset
    st.header('Movies Dataset')
    st.write('First 5 rows of the Movies dataset:')
    st.dataframe(movies_df.head())

    st.header('Ratings Dataset')
    st.write('First 5 rows of the Ratings dataset:')
    st.dataframe(ratings_df.head())

    # Report the number of rows and columns for each dataset
    st.header('Dataset Dimensions')
    st.write(f'The Movies dataset contains {movies_df.shape[0]} rows and {movies_df.shape[1]} columns.')
    st.write(f'The Ratings dataset contains {ratings_df.shape[0]} rows and {ratings_df.shape[1]} columns.')
