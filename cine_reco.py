import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load the movie dataset
def load_data():
    movies = pd.read_csv("movies.csv")  # Ensure the dataset contains 'title', 'description', and 'reviews'
    return movies

# Preprocess text data
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Train Sentiment Analysis Model
def train_sentiment_model(movies):
    movies['processed_reviews'] = movies['reviews'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(movies['processed_reviews'], movies['sentiment'], test_size=0.2, random_state=42)

    # Manually setting up the model pipeline
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_tfidf, y_train)

    return vectorizer, classifier

# Build the TF-IDF matrix for recommendations
def build_tfidf_matrix(movies):
    movies['processed_description'] = movies['description'].apply(preprocess)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies['processed_description'])
    return tfidf_matrix, movies

# Recommend movies based on user input
def recommend_movies(movie_title, movies, tfidf_matrix):
    if movie_title not in movies['title'].values:
        return "Movie not found. Please try another title."
    
    movie_index = movies.index[movies['title'] == movie_title][0]
    similarity_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix)
    similar_movies = list(enumerate(similarity_scores[0]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = [movies.iloc[i[0]]['title'] for i in similar_movies]
    return recommendations

if __name__ == "__main__":
    movies = load_data()
    vectorizer, sentiment_model = train_sentiment_model(movies)
    tfidf_matrix, movies = build_tfidf_matrix(movies)
    movie_title = input("Enter a movie title: ")
    recommendations = recommend_movies(movie_title, movies, tfidf_matrix)
    print("Recommended Movies:", recommendations)
