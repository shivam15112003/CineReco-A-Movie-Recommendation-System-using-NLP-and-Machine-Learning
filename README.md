# 🎬 CineReco: A Movie Recommendation System using NLP & Machine Learning

## 📌 Overview

CineReco is an AI-powered movie recommendation system that uses Natural Language Processing (NLP) and Machine Learning to provide personalized recommendations. It also includes sentiment analysis to evaluate movie reviews.

## 🚀 Features

- ✅ Personalized Movie Recommendations using TF-IDF & Cosine Similarity.
- ✅ Sentiment Analysis using Naïve Bayes classifier on movie reviews.
- ✅ NLP-based Text Preprocessing for better feature extraction.
- ✅ **Case-Insensitive Movie Title Matching**: users can input movie titles in any case (e.g., `avatar`, `AVATAR`, `Avatar`) and still get accurate results.
- ✅ **Title Deduplication**: ensures recommendations are not repeated for the same movie.
- ✅ **Accurate Title Output**: returns recommended movies with original, correctly cased titles.
- ✅ **Excludes Self-Recommendation**: filters out recommending the movie itself.
- ✅ Exception Handling for missing datasets and invalid inputs.

## 🔧 Technologies Used

- Python
- Pandas & NumPy
- Scikit-Learn (ML & NLP models)
- NLTK (Text Processing)

## 📂 Installation & Usage

### Clone the repository:

```bash
git clone https://github.com/shivam15112003/CineReco-A-Movie-Recommendation-System-using-NLP-and-Machine-Learning.git
cd CineReco-A-Movie-Recommendation-System-using-NLP-and-Machine-Learning
