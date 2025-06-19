import pandas as pd
import numpy as np
import re
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Download NLTK resources
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
    print("Generating SBERT embeddings (this may take a few seconds)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
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

    # Normalize both scores
    scaler = MinMaxScaler()
    cb_sim_norm = scaler.fit_transform(cb_sim.reshape(-1,1)).flatten()
    cf_sim_norm = scaler.fit_transform(cf_sim.reshape(-1,1)).flatten()

    final_similarity = alpha * cb_sim_norm + (1 - alpha) * cf_sim_norm

    # Sort and get top N excluding self
    movie_indices = np.argsort(final_similarity)[::-1]
    recommendations = []
    for idx in movie_indices:
        if movies.iloc[idx]['title'] != movie_title:
            recommendations.append((movies.iloc[idx]['title'], final_similarity[idx]))
        if len(recommendations) >= n:
            break

    return recommendations

# ------------------ Evaluation Metrics -------------------
def precision_at_k(recommended, relevant, k=5):
    recommended_at_k = recommended[:k]
    hits = sum([1 for item in recommended_at_k if item in relevant])
    return hits / k

def ndcg_at_k(recommended, relevant, k=5):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum([1 / math.log2(i + 2) for i in range(ideal_hits)])
    return dcg / idcg if idcg > 0 else 0.0

# ------------------ Main -------------------
if __name__ == "__main__":
    # Load and prepare data
    movies = load_movie_data()
    item_similarity_df = load_ratings_data()
    sbert_model, embeddings = build_sbert_embeddings(movies)

    # Input
    movie_title = input("Enter a movie title you liked: ").strip()
    relevant_movies_input = input("Enter other movies you liked (comma-separated): ")
    relevant_movies = [title.strip() for title in relevant_movies_input.split(",")]

    # Generate recommendations
    recommendations = hybrid_similarity_fusion(movie_title, movies, embeddings, item_similarity_df, alpha=0.5, n=5)

    # Display
    if recommendations:
        print("\n🎯 Top 5 Deep Learning Hybrid Recommendations:")
        for rec, score in recommendations:
            print(f"- {rec} (Score: {score:.4f})")

        recommended_titles = [rec[0] for rec in recommendations]

        # Evaluation
        p_at_5 = precision_at_k(recommended_titles, relevant_movies, k=5)
        ndcg_5 = ndcg_at_k(recommended_titles, relevant_movies, k=5)

        print(f"\n📊 Evaluation (based on your liked movies):")
        print(f"Precision@5: {p_at_5:.2f}")
        print(f"NDCG@5: {ndcg_5:.2f}")
    else:
        print("No recommendations could be generated.")
