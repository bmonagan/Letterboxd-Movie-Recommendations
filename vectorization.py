import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz,load_npz 
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow # Required for pandas.to_parquet
import pyarrow.parquet as pq # Required for pandas.to_parquet


from helper_functions import parse_and_clean

df = pd.read_csv('data/TMDB_movie_dataset.csv')

selected_columns = ['title', 'id', 'overview', 'genres', 'keywords']
new_df = df[selected_columns].copy() # .copy() to avoid SettingWithCopyWarning

print(new_df.head())

#apply parsing/cleaning to genres and keywords
new_df['genres_cleaned'] = new_df['genres'].apply(parse_and_clean)
new_df['keywords_cleaned'] = new_df['keywords'].apply(parse_and_clean)

# Fill NaN values in 'overview' values
new_df['overview'] = new_df['overview'].fillna('')

# Create the new column by combining the text features
new_df['combined_features'] = new_df['overview'] + ' ' + \
                              new_df['genres_cleaned'] + ' ' + \
                              new_df['keywords_cleaned']

# Print the DataFrame with combined features
print("\nDataFrame after cleaning and combining features:")
print(new_df[['combined_features']].head())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit the vectorizer on the *entire* 'combined_features' column and transform it
tfidf_matrix = vectorizer.fit_transform(new_df['combined_features'])

# Print the shape of the TF-IDF matrix (number of documents, number of unique terms)
print(f"\nShape of TF-IDF matrix: {tfidf_matrix.shape}")

sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
df[['id', 'title']].to_parquet('movie_metadata.parquet', index=False)


