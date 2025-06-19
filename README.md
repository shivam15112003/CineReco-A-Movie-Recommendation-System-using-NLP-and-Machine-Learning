# 🎬 CineReco: Deep Learning Hybrid Movie Recommendation System

## 📌 Overview
CineReco is an AI-powered movie recommendation system that combines **deep learning-based semantic similarity** (using Sentence-BERT) with **collaborative filtering** to deliver highly accurate movie recommendations. This hybrid system leverages both movie content descriptions and user behavior to provide robust, personalized suggestions.

## 🚀 Features

- ✅ NLP-based Text Preprocessing for better feature extraction.
- ✅ Case-Insensitive Movie Title Matching: users can input movie titles in any case (e.g., avatar, AVATAR, Avatar) and still get accurate results.
- ✅ Title Deduplication: ensures recommendations are not repeated for the same movie.
- ✅ Accurate Title Output: returns recommended movies with original, correctly cased titles.
- ✅ Excludes Self-Recommendation: filters out recommending the movie itself.
- ✅ Exception Handling for missing datasets and invalid inputs.
- 🔎 **Deep Learning Content-Based Filtering**: Uses SBERT (`sentence-transformers`) for semantic similarity on movie descriptions.
- 🤝 **Collaborative Filtering**: Incorporates user rating data to enhance recommendations based on user behavior patterns.
- ⚙ **Hybrid Fusion Model**: Combines both similarity scores into a unified ranking using normalized cosine similarity.
- 🔬 **Preprocessing with NLP**: Efficient text cleaning using NLTK for better semantic encoding.
- 🔢 **Cosine Similarity Fusion**: Balanced scoring between content and collaborative models.
- ⚠ **Cold Start Friendly**: Works even if user ratings are limited, due to strong content-based model.
- 📊 **Evaluation Support**:Calculates Precision@5 and NDCG@5 based on user-provided liked movies to measure relevance and ranking quality.



## 🔧 Technologies Used
- math
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- sentence-transformers (SBERT)
- PyTorch (backend for transformers)

## 📂 Dataset

Two CSV files are used:

### 1️⃣ movies_dataset_300.csv

| title | description | review | sentiment |
|-------|-------------|--------|-----------|

### 2️⃣ user_movie_ratings.csv
| user | title | rating |
|------|-------|--------|

### 🧪 Evaluation Metrics
After generating recommendations, CineReco can evaluate accuracy based on user-specified liked movies:

**Precision@5**: How many of the top 5 suggestions were relevant.

**NDCG@5**: How well ranked the relevant movies were.

## 📂 Installation & Usage

### Clone the repository:

```bash
git clone https://github.com/shivam15112003/CineReco-A-Movie-Recommendation-System-using-NLP-and-Machine-Learning.git
cd CineReco-A-Movie-Recommendation-System-using-NLP-and-Machine-Learning
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the program:

```bash
python cine_reco.py
```

### Usage:

- Enter any movie title (case-insensitive) when prompted.
- The system returns the Top 5 hybrid recommendations.
- Evaluation: Precision@5 and NDCG@5

## 📈 Future Enhancements

- Replace collaborative filtering with full Neural Collaborative Filtering (NCF).
- Add DNN-based sentiment classifier for reviews.
- Integrate streaming APIs for real-world data.
- Build a web-based GUI (Flask / FastAPI + React).

---

## Note:

The system now uses a helper lowercase column `title_lower` internally to handle case-insensitive matching, while preserving the original movie titles for output display.

