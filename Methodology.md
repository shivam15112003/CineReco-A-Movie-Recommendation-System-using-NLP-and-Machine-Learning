📌 Methodology

## 1️⃣ Data Collection & Preprocessing
- Collected movie data including **titles, descriptions, and user reviews**.
- Cleaned text by removing **stopwords, special characters, and tokenizing words** using NLTK.
- Applied **TF-IDF Vectorization** to convert movie descriptions into numerical representations.

## 2️⃣ Sentiment Analysis
- Labeled movie reviews as **positive or negative**.
- Used **Naïve Bayes classifier** with TF-IDF features to train a sentiment analysis model.
- Classified new movie reviews to enhance recommendation quality.

## 3️⃣ Movie Recommendation System
- Computed **cosine similarity** between movie descriptions to find similar movies.
- Designed an **NLP-powered recommendation model** based on user input.
- Returns **top 5 similar movies** for the given input title.

## 4️⃣ User Interaction & Output
- Takes user input for a movie title.
- Processes the title and retrieves **most relevant recommendations**.
- Displays **recommended movies** based on content similarity.

This methodology ensures an **efficient, scalable, and personalized movie recommendation system** using **AI and NLP techniques**.

