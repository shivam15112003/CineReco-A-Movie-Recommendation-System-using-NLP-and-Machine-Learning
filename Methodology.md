## 📌 Methodology

## 1️⃣ Data Collection & Preprocessing
- Collected movie data including **titles, descriptions, and user reviews**.
- Cleaned text by removing **Lowercasing, stopwords, special characters, and tokenizing words** using NLTK.
- Added **deduplication** step to ensure each movie title appears only once.
- Created a **lowercase helper column (title_lower)** for case-insensitive title matching, while preserving original title casing for display.
- Prepared user-item ratings dataset for collaborative filtering.

## 2️⃣ Content-Based Filtering (Deep Learning Powered)
- **Applied Sentence-BERT (SBERT) (all-MiniLM-L6-v2 model from sentence-transformers)** to generate dense 768-dimensional semantic embeddings from preprocessed movie descriptions.
- Computed pairwise **cosine similarity** between all movie embeddings to measure content similarity.
- This allowed capturing **deep contextual relationships** between movies beyond simple keyword matching.

## 3️⃣ Collaborative Filtering
- Constructed **user-item matrix** from user ratings data.
- Computed cosine similarity between items (movies) based on user rating vectors.
- This captured similarity based on how **users rated different movies**.

## 4️⃣ Hybrid Similarity Fusion
- Both similarity scores (Content-Based & Collaborative Filtering) were **normalized using MinMaxScaler (range 0 to 1)**.
- Final similarity score calculated as weighted average: **Final Score = 𝛼 ⋅ Content Similarity + (1 − 𝛼)**
- Default weight: **α = 0.5 (equal importance to both models)**
- Generated top-5 recommendations by **ranking final fused similarity scores**.

## 4️⃣ User Interaction & Output
- Accepts user input for any **movie title (case-insensitive)**.
- Internally matches the input against a **lowercased normalized title** column to ensure robust search.
- Retrieves and displays recommendations using the movie’s original title casing for better presentation.
- Displays the Top-5 most similar movies based on the **final hybrid similarity score**.
- Handles invalid or unmatched titles gracefully by prompting appropriate error messages.
- Supports evaluation using:
           **Precision@5** — relevance of recommended movies
           **NDCG@5** — ranking quality of the results based on user-provided liked movies


This methodology ensures an **efficient, scalable, and personalized movie recommendation system** using **AI,Deep Learning and NLP techniques**.

