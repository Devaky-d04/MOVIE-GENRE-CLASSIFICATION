# Preprocess the 'overview' column (plot summaries)
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase and remove stopwords
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
df['overview'] = df['overview'].apply(preprocess_text)
