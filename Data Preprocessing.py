# Load the dataset
df = pd.read_csv('tmdb_movies_data.csv')  # Replace with your dataset path

# Drop rows with missing plot or genres
df = df.dropna(subset=['overview', 'genres'])

# Process the 'genres' column into a list of genres
df['genres'] = df['genres'].apply(lambda x: x.split('|'))

# Print the column names to verify the dataset
print(df.columns)

# View the first few rows of the relevant columns
print(df[['original_title', 'overview', 'genres']].head())  # Replace 'original_title' if necessary
