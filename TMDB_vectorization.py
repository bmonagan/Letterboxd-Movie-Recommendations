import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz,load_npz 
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow # Required for pandas.to_parquet
import pyarrow.parquet as pq # Required for pandas.to_parquet


from helper_functions import parse_and_clean

df = pd.read_csv('data/TMDB_movie_dataset.csv')

selected_columns = ['id', 'overview', 'genres', 'keywords']
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
print(new_df[['id', 'combined_features']].head())

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit the vectorizer on the *entire* 'combined_features' column and transform it
tfidf_matrix = vectorizer.fit_transform(new_df['combined_features'])

# Print the shape of the TF-IDF matrix (number of documents, number of unique terms)
print(f"\nShape of TF-IDF matrix: {tfidf_matrix.shape}")

# Get the feature names (i.e., the words in the vocabulary)
feature_names = vectorizer.get_feature_names_out()
print("\nFirst 10 feature names (vocabulary):")
print(feature_names[:10])

# To see the TF-IDF scores for a specific movie (e.g., the first movie):
# Convert the sparse matrix row to a dense array for easier viewing
first_movie_vector = tfidf_matrix[0].todense()

# Get the indices of non-zero (i.e., present) TF-IDF values for the first movie
feature_index = first_movie_vector.nonzero()[1]
tfidf_scores = [first_movie_vector[0, idx] for idx in feature_index]

# Create a dictionary of features and their scores for the first movie
feature_score_dict = {feature_names[i]: score for i, score in zip(feature_index, tfidf_scores)}

# Convert the sparse TF-IDF matrix to a dense NumPy array
dense_tfidf_matrix = tfidf_matrix.toarray()

# Create a DataFrame from the dense TF-IDF matrix, using feature_names as column headers
tfidf_df = pd.DataFrame(dense_tfidf_matrix, columns=feature_names)

tfidf_df.insert(0, 'id', new_df['id'].reset_index(drop=True))

# Save the combined DataFrame to a Parquet file(NOT SURE THIS IS THE DATA FORMAT LONG. SAW IT AND WANTED TO TRY OUT THE FORMAT)
output_parquet_filename = 'movie_vectors_with_ids.parquet'
tfidf_df.to_parquet(output_parquet_filename, index=False)
print(f"\nMovie IDs and TF-IDF vectors saved to '{output_parquet_filename}' (Parquet format)")
