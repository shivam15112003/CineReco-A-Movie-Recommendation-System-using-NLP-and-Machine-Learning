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

# Downloading NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def load_data(file_path="movies_dataset_300.csv"):
    """Loads the movie dataset from a CSV file."""
    try:
        movies = pd.read_csv(file_path)
        return movies
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

def preprocess(text):
    """Preprocesses text by removing non-alphanumeric characters, lowercasing, and removing stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def train_sentiment_model(movies):
    """Trains a Naive Bayes sentiment analysis model using TF-IDF features."""
    movies['processed_reviews'] = movies['reviews'].apply(preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        movies['processed_reviews'], movies['sentiment'], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    classifier.fit(X_train_tfidf, y_train)
    
    return vectorizer, classifier

def build_tfidf_matrix(movies):
    """Builds a TF-IDF matrix for movie descriptions."""
    movies['processed_description'] = movies['description'].apply(preprocess)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies['processed_description'])
    return vectorizer, tfidf_matrix

def recommend_movies(movie_title, movies, tfidf_matrix):
    """Recommends similar movies based on cosine similarity of TF-IDF vectors."""
    if movie_title not in movies['title'].values:
        return "Movie not found. Please try another title."
    
    movie_index = movies.index[movies['title'] == movie_title][0]
    similarity_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix)[0]
    
    # Create list of indices and similarity scores excluding the input movie itself
    similar_movies = [
        (i, score) for i, score in enumerate(similarity_scores) if i != movie_index
    ]
    
    # Sort by similarity score
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[:5]
    
    recommendations = [movies.iloc[i[0]]['title'] for i in similar_movies]
    return recommendations


if __name__ == "__main__":
    movies = load_data()
    if movies is not None:
        # Deduplicate titles here
        movies = movies.drop_duplicates(subset=['title'])
        
        vectorizer, sentiment_model = train_sentiment_model(movies)
        vectorizer, tfidf_matrix = build_tfidf_matrix(movies)
        
        movie_title = input("Enter a movie title: ")
        recommendations = recommend_movies(movie_title, movies, tfidf_matrix)
        print("Recommended Movies:", recommendations)


