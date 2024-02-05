# pages/data_overview.py
import streamlit as st
import pandas as pd

def show_page():
    st.title('Test Train Split Overview & Feature Enhancement')

    # Step 1: Load and display the original datasets
    movies_df = pd.read_csv('dataset/movies.csv')
    ratings_df = pd.read_csv('dataset/ratings.csv')

    st.header('Original Dataset')
    st.write('Showing the top 2 rows of the Movies dataset:')
    st.dataframe(movies_df.head(2))

    st.write('Showing the top 2 rows of the Ratings dataset:')
    st.dataframe(ratings_df.head(2))

    # Step 2: Load new datasets
    train_data = pd.read_csv('dataset/train_data.csv')
    test_data = pd.read_csv('dataset/test_data.csv')

    # Just to illustrate that datasets have been split according to the requirement
    st.header('Train and Test Data Split')
    st.write('This section is to verify that the training and testing datasets have been split evenly based on user IDs as required.')
    st.write('Showing the top 5 rows of the Train dataset:')
    st.dataframe(train_data.head(5))  
    st.write('Showing the top 5 rows of the Test dataset:')
    st.dataframe(test_data.head(5))  

    # Step 5: Explanation of Feature Engineering
    st.header('Explanation of Feature Engineering')
    st.markdown("""
    The following feature engineering steps were applied to the training dataset:

    1. **Title and Year Split:** The 'title' column was split into two columns: 'title' and 'year'. The year was extracted from the title and placed in its own column, making it easier to analyze movies by year.
    
    2. **Genres Conversion:** The 'genres' column, originally a string with genres separated by "|", was converted into a list of genres for each movie. This format is more suitable for analysis and processing of individual genres.
    
    3. **User Average Rating:** A new column, 'user_average_rating', was added to the training dataset. This column represents the average rating each user has given across all their reviews within the training set, providing insight into user rating behavior.
    
    4. **Movie Rated Count:** An additional column, 'movie_rated_cnt', was introduced to show the total number of ratings received by each movie in the training dataset. This can help identify popular movies or movies that may need more ratings.
    
    5. **Movie Average Rating:** Lastly, a 'movie_average_rating' column was added to reflect the average rating each movie received within the training dataset. This offers a direct measure of a movie's overall reception by users.
    """)

