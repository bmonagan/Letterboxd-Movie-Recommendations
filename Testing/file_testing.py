import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
target_movie_id = 524
# 1. Load metadata
# df = pd.read_csv('data/TMDB_movie_dataset.csv')
# df = df[['id', 'title', 'overview']]  # or whatever text field you want to vectorize

# 2. Vectorize
# vectorizer = TfidfVectorizer(max_features=5000)
# tfidf_matrix = vectorizer.fit_transform(df['overview'].fillna(''))

# 3. Save vectors and metadata
# # sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
# df[['id', 'title']].to_parquet("data/movie_metadata.parquet", index=False)

# 4. For lookup:
# Load both files
# tfidf_matrix = sparse.load_npz('tfidf_matrix.npz')
metadata = pd.read_parquet('data/movie_metadata.parquet')

# # To get title from a row index (e.g., idx)
# movie_id = metadata.iloc[idx]['id']
# movie_title = metadata.iloc[idx]['title']
row = metadata[metadata['id'] == target_movie_id].iloc[0]
target_movie_title = metadata.iloc[target_movie_id]
print(target_movie_title)
# This should be a string, e.g., "Casino"
print(type(target_movie_title), target_movie_title)

# print(row)
# print(metadata.head())