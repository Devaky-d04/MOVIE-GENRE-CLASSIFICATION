# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the 'overview' column
X = vectorizer.fit_transform(df['overview'])
