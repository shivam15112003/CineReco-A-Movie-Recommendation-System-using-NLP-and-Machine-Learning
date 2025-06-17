📌 Methodology

## 1️⃣ Data Collection & Preprocessing
- Collected movie data including **titles, descriptions, and user reviews**.
- Cleaned text by removing **stopwords, special characters, and tokenizing words** using NLTK.
- Applied **TF-IDF Vectorization** to convert movie descriptions into numerical representations.
- Added **deduplication** step to ensure each movie title appears only once.
- Created a **lowercase helper column (title_lower)** for case-insensitive title matching, while preserving original title casing for display.

## 2️⃣ Sentiment Analysis
- Labeled movie reviews as **positive or negative**.
- Used **Naïve Bayes classifier** with TF-IDF features to train a sentiment analysis model.
- Sentiment model trained using Scikit-Learn and NLTK-based preprocessing.
- Classified new movie reviews to enhance recommendation quality.

## 3️⃣ Movie Recommendation System
- Computed **cosine similarity** between movie descriptions to find similar movies.
- Designed an **NLP-powered recommendation model** based on user input.
- Ensures recommendations exclude recommending the movie itself **(self-recommendation filter)**.
- Returns **top 5 similar movies** for the given input title.
- Always displays the **original properly-cased movie titles** in output for a clean user experience.

## 4️⃣ User Interaction & Output
- Accepts user input for any **movie title (case-insensitive matching)**.
- Processes the title and retrieves **most relevant recommendations**.
- Displays **recommended movies** in **original correct casing** based on content similarity.
- Handles invalid movie inputs gracefully with appropriate messages.



This methodology ensures an **efficient, scalable, and personalized movie recommendation system** using **AI and NLP techniques**.

