# pages/data_overview.py
import streamlit as st

def show_page():
    st.title('Recommendation abstract')
    st.header("Model Development")

    # Explanation of Neural Network Training
    st.markdown("""
    To predict movie scores, I trained a neural network that utilizes movie embeddings and user embeddings. This approach allows for personalized recommendations by understanding the intricate relationships between users and movies beyond mere ratings.
    """)

    # Model Architecture Explanation
    st.subheader("Model Architecture")
    st.markdown("""
    The model architecture is designed as follows:
    
    - A dense input layer with 64 units, followed by a LeakyReLU activation function and Dropout of 0.5 for regularization.
    - Batch normalization is applied to stabilize and accelerate the learning process.
    - This pattern is repeated for another layer.
    - The output layer consists of a single unit to predict the movie rating.
    
    The model is compiled with the Adam optimizer and mean squared error loss function.
    """)
    # st.image("images/train_validation_loss.png", caption="Training and Validation Loss")

    # Model Training and Validation Loss
    st.subheader("Model Training and Validation Loss")
    st.image("images/train_validation_loss.png", caption="Training and Validation Loss")

    # Evaluation on Test Dataset
    st.subheader("Model Evaluation")
    st.markdown("""
    After evaluation on the test dataset, the model achieved a Root Mean Squared Error (RMSE) of  1.0733. This metric helps in understanding the model's accuracy in predicting movie scores.
    """)

    # Recommendation Rationale
    st.header("Recommendation Rationale")
    st.markdown("""
    The rationale behind the recommendation system is as follows:
    
    1. For a given userId, identify all unseen movies within the training dataset.
    2. Predict scores for each of these unseen movies.
    3. Rank these movies based on the predicted scores from highest to lowest, and also consider the movie's popularity (number of ratings in the training dataset).
    4. Recommend the top 5
                
    This approach ensures that recommendations are both personalized and reflect movies' general appeal.
    """)