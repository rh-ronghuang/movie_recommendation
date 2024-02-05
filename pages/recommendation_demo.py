# pages/data_overview.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model


# load model
model_path = 'saved_model/model.keras'
model = load_model(model_path)

# Global dictionaries
movieId_to_name = {}
ratings_by_user_train = {}
ratings_by_user_test = {}
movie_view_frequency = {}
userId_top_5_movies = {}
movies_emb = {}
userId_emb = {}

def load_dicts():
    global movieId_to_name, ratings_by_user_train, ratings_by_user_test, movie_view_frequency
    global userId_top_5_movies, movies_emb, userId_emb

    with open('saved_dicts/movieId_to_name.pkl', 'rb') as f:
        movieId_to_name = pickle.load(f)
    with open('saved_dicts/ratings_by_user_train.pkl', 'rb') as f:
        ratings_by_user_train = pickle.load(f)
    with open('saved_dicts/ratings_by_user_test.pkl', 'rb') as f:
        ratings_by_user_test = pickle.load(f)
    with open('saved_dicts/movie_view_frequency.pkl', 'rb') as f:
        movie_view_frequency = pickle.load(f)
    with open('saved_dicts/userId_top_5_movies.pkl', 'rb') as f:
        userId_top_5_movies = pickle.load(f)   
    with open('saved_dicts/movies_emb.pkl', 'rb') as f:
        movies_emb = pickle.load(f)
    with open('saved_dicts/userId_emb.pkl', 'rb') as f:
        userId_emb = pickle.load(f)

def recommend(userId, k):
    load_dicts()

    # userId rated # of movies in training dataset")
    cnt_ratings = len(ratings_by_user_train[userId])

    # find all unrated movies for userId
    unseen_movies_emb = []
    unseen_movieId = []
    for movieId, emb in movies_emb.items():
        if movieId not in set(ratings_by_user_train[userId]):
            unseen_movies_emb.append(emb)
            unseen_movieId.append(movieId)
    unseen_movies_emb = np.concatenate(unseen_movies_emb, axis=0)

    # prepare X for this user: [moive_embd, userId_emb]
    user_emb_broadcasted = np.tile(userId_emb[userId], (unseen_movies_emb.shape[0], 1))

    # final X embedding for userId and all unseen movies
    X_emb = np.hstack((unseen_movies_emb, user_emb_broadcasted))

    # predict scores for unseen movies
    unseen_ratings = model.predict(X_emb).flatten()

    # popularity for unseen movies (rated numbers)
    popularity = []
    for i, movieId in enumerate(unseen_movieId):

        if movieId not in movie_view_frequency:
            popularity.append(0)
        else:
            popularity.append(movie_view_frequency[movieId]) 

    # sort the predicted movies scores [and popularity] from high to low 
    combined_data = list(zip(unseen_movieId, unseen_ratings, popularity))
    # seen at least twice in train, then it can be recommended
    combined_data = [item for item in combined_data if item[2] >= 4]

    sorted_data = sorted(combined_data, key=lambda x: (-x[1], -x[2]))
    sorted_unseen_movieId = [item[0] for item in sorted_data]
    sorted_unseen_ratings = [item[1] for item in sorted_data]
    
    # if the top 5 recommendation is actually rated in test dataset, displayed below
    rec_in_test = [] # [movieId, predicted_score, true_score]

    # print the top k highest
    # Define the bin edges
    bin_edges = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    top_k_recommendations = [] # [[movieIds], [predicted_ratings]]
    for movieId, predicted_score in zip(sorted_unseen_movieId[:k], sorted_unseen_ratings[:k]):
        predicted_score = bin_edges[np.digitize(predicted_score, bin_edges) - 1] 
        top_k_recommendations.append([movieId,predicted_score])

        # if the recommendation is userId's test dataset
        if movieId in set(ratings_by_user_test[userId].keys()):
            rec_in_test.append([movieId, predicted_score, ratings_by_user_test[userId][movieId]])
        
    # check top5 highest rated movies for userId in test dataset
    X_top5_emb = []
    X_top5_movies = []
    for movieId in userId_top_5_movies[userId]:
        if len(X_top5_emb) == 5:
            break
        if movieId not in movies_emb:
            continue
        X_top5_emb.append(movies_emb[movieId])
        X_top5_movies.append(movieId)

    X_top5_emb = np.concatenate(X_top5_emb, axis=0)
    user_emb_broadcasted = np.tile(userId_emb[userId], (X_top5_emb.shape[0], 1))
    X_top5_emb = np.hstack((X_top5_emb, user_emb_broadcasted))

    # predict scores for top5 movies
    top_5_test = []
    X_top5_emb_predicted_ratings = model.predict(X_top5_emb).flatten()
    for i, movieId in enumerate(X_top5_movies):
        predicted_score = bin_edges[np.digitize(X_top5_emb_predicted_ratings[i], bin_edges) - 1] 
        top_5_test.append([movieId, predicted_score, ratings_by_user_test[userId][movieId]])    
    
    return top_k_recommendations, cnt_ratings, rec_in_test, top_5_test

def show_page():
    st.title('Recommendation Demo')
    user_id = st.number_input('Enter User ID (1-610)', min_value=1, max_value=610, value=1)

    if st.button('Get Recommendations'):
        top_k_recommendations, cnt_ratings, rec_in_test, top_5_test = recommend(user_id, 5)

        st.write(f"User ID {user_id} has rated {cnt_ratings} movies in the training dataset.")
        
        st.subheader('Top 5 Recommended Movies:')
        for movie_id, predicted_rating in top_k_recommendations:
            st.write(f"Movie ID: {movie_id}, Predicted Rating: {predicted_rating:.2f}")

        if rec_in_test:
            st.subheader(f'Recommended Movies are rated by User ID {user_id} in the Test Dataset:')
            for movie_id, predicted_score, actual_score in rec_in_test:
                st.write(f"Movie ID: {movie_id}, Predicted Rating: {predicted_score:.2f}, Actual Rating: {actual_score}")
        else:
            st.write("None of the top 5 recommended movies are in the test dataset.")

        st.subheader(f'Top 5 Movies Rated by User ID {user_id} in Test Dataset (by Rating and Popularity):')
        for movie_id, predicted_score, true_score in top_5_test:
            st.write(f"Movie ID: {movie_id}, Predicted Rating: {predicted_score if predicted_score != 'N/A' else 'Not Predicted'}, True Rating: {true_score}")