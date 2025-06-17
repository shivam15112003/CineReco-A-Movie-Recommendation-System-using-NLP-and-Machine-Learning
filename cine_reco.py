import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Download stopwords only once
nltk.download('punkt')
nltk.download('stopwords')

# ------------------ Data Loading -------------------
def load_movie_data(file_path="movies_dataset_300.csv"):
    movies = pd.read_csv(file_path)
    movies = movies.drop_duplicates(subset=['title'])
    movies['title_lower'] = movies['title'].str.lower()
    return movies

def load_ratings_data(file_path="user_movie_ratings.csv"):
    ratings = pd.read_csv(file_path)
    user_item_matrix = ratings.pivot_table(index='user', columns='title', values='rating').fillna(0)
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return item_similarity_df

# ------------------ Preprocessing -------------------
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ------------------ Deep Learning Content Embeddings -------------------
def build_sbert_embeddings(movies):
    print("Generating SBERT embeddings (this may take few seconds)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, good quality
    movies['processed_description'] = movies['description'].apply(preprocess)
    embeddings = model.encode(movies['processed_description'].tolist())
    return model, embeddings

def content_based_similarity_sbert(movie_title, movies, embeddings):
    movie_title_lower = movie_title.lower()
    if movie_title_lower not in movies['title_lower'].values:
        return None
    movie_index = movies.index[movies['title_lower'] == movie_title_lower][0]
    similarity_scores = cosine_similarity([embeddings[movie_index]], embeddings)[0]
    return similarity_scores

# ------------------ Hybrid Fusion -------------------
def hybrid_similarity_fusion(movie_title, movies, embeddings, item_similarity_df, alpha=0.5, n=5):
    cb_sim = content_based_similarity_sbert(movie_title, movies, embeddings)
    
    if cb_sim is None:
        print("Movie not found in content-based dataset.")
        return []
    
    if movie_title not in item_similarity_df.columns:
        print("Movie not found in collaborative filtering dataset.")
        return []

    cf_sim = item_similarity_df[movie_title].values

    # Normalize both scores to 0-1
    scaler = MinMaxScaler()
    cb_sim_norm = scaler.fit_transform(cb_sim.reshape(-1,1)).flatten()
    cf_sim_norm = scaler.fit_transform(cf_sim.reshape(-1,1)).flatten()

    final_similarity = alpha * cb_sim_norm + (1 - alpha) * cf_sim_norm

    # Sort top N results excluding self-recommendation
    movie_indices = np.argsort(final_similarity)[::-1]
    recommendations = []
    for idx in movie_indices:
        if movies.iloc[idx]['title'] != movie_title:
            recommendations.append((movies.iloc[idx]['title'], final_similarity[idx]))
        if len(recommendations) >= n:
            break

    return recommendations

# ------------------ Main -------------------
if __name__ == "__main__":
    movies = load_movie_data()
    item_similarity_df = load_ratings_data()
    sbert_model, embeddings = build_sbert_embeddings(movies)

    movie_title = input("Enter a movie title: ")

    recommendations = hybrid_similarity_fusion(movie_title, movies, embeddings, item_similarity_df, alpha=0.5, n=5)

    if recommendations:
        print("\n🎯 Top 5 Deep Learning Hybrid Recommendations:")
        for rec, score in recommendations:
            print(f"- {rec} (Score: {score:.4f})")
