import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
import itertools

#NLTK Resources ni 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#streaming
class DocumentStreamer(object):
    def __init__(self, df):
        self.df = df

    def __iter__(self):
        for index, row in self.df.iterrows():
            # Tokenize the text
            tokens = word_tokenize(row['text'])
            yield TaggedDocument(words=tokens, tags=[index])

# List of file paths
file_paths = [r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_first_5000_done.xlsx",
            r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_second_5000_done.xlsx",
             r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_third_5000_done.xlsx",
             r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_fourth_5000_done.xlsx",
             r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_fifth_5000_done.xlsx",
             r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_sixth_5000_done.xlsx",
             r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\for_scraping_seventh_5000_done.xlsx"]

# Initialize an empty dataframe
all_dataframes = []

# Load data from multiple Excel files
for file_path in file_paths:
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        all_dataframes.append(df)
    else:
        print(f"File not found: {file_path}")

df = pd.concat(all_dataframes, ignore_index=True)

# I-exclude ang rows kung asa ang "Title" column contains "Not Found" and the "Content" column is empty
df = df[(df['ScientificName'] != 'Not Found') & (df['Content'].notnull())]

df

# words to remove
words_to_remove = [ "plant", "plants", "specie", 'flower', 'reference', 'external', 'links', 'also', 'var', 'name', 'used', 'leaf', 'tree', 'rknuth', '-']

# Additional step: Remove duplicates from the DataFrame
df = df.drop_duplicates(subset=['ScientificName'])

# Preprocess the data and create TaggedDocument instances
documents = []
all_words = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer() #reduce a word to its base or dictionary form ( eg running to run or better to good)

# mag-efine og translation table to remove punctuation
translator = str.maketrans('', '', string.punctuation)

for index, row in df.iterrows():
    title = row['ScientificName']
    content = row['Content']
    # mag-combine sa title and content na columns for better representation
    combined_text = f"{title.lower()} {content.lower()}"
    
    #Tokenize the text and lemmatize
    words = [
        lemmatizer.lemmatize(word.lower()) 
        for word in nltk.word_tokenize(combined_text.translate(translator)) 
        if word.lower() not in stop_words
        #and word != 'species' and word != 'plant' and word != 'flowers'
    ]
    
    #Remove specific words
    for word_to_remove in words_to_remove:
        words = [word for word in words if word != word_to_remove.lower()]

    documents.append(TaggedDocument(words, [str(index)]))
    all_words.extend(words)

#Build a corpus from TaggedDocument instances
corpus = documents

corpus[1]

#i-determine ang number of corpus
n_docs = len(corpus)
print("Number of documents in the corpus:", n_docs)

#Build a dictionary mapping words to their frequencies
word_frequency_dict = Counter(all_words)

#Print the most frequent words
most_common_words = Counter(all_words).most_common(10)
print("\nMost Common Words:")
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")

# Train a Doc2Vec model
model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=10)
model.build_vocab(corpus)
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

#kuhaon ang inferred vectors for each document
#vectors = [model.dv[index] for index in range(len(df))]

vectors = [model.dv[str(index)] for index in df.index]

vectors_array = np.array(vectors)

'''PERFORM CLUSTER ANALYSIS'''

num_clusters = 1000
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vectors)

df['cluster'] = kmeans.labels_

labels = kmeans.labels_

from sklearn import metrics

metrics.silhouette_score(vectors, labels, metric='euclidean')

cluster_distribution = df['cluster'].value_counts()
print("Cluster Distribution:")
print(cluster_distribution)

cluster_dict = {}

# Populate the dictionary with cluster information
for cluster_id in range(num_clusters):
    cluster_titles = df[df['cluster'] == cluster_id]['ScientificName']
    cluster_dict[cluster_id] = list(cluster_titles)

# Function to search for a title and retrieve cluster information
def search_title(title):
    for cluster_id, titles in cluster_dict.items():
        if title in titles:
            return cluster_id, titles



def calculate_similarity(vectors):
    return cosine_similarity(vectors)

searched_title = "Oryza sativa" 
result = search_title(searched_title)

######para di magbalikbalik ang sciname

if result:
    cluster_id, titles = result
    print(f"Cluster ID: {cluster_id}")
    print(f"Titles in Cluster {cluster_id}:\n{titles}")

    # Get vectors and titles for the specific cluster
    cluster_vectors = vectors_array[df['cluster'] == cluster_id]
    cluster_titles = df[df['cluster'] == cluster_id]['ScientificName']

    # Calculate cosine similarity matrix
    similarity_matrix = calculate_similarity(cluster_vectors)

    # Collect and sort the similarity pairs
    similarity_pairs = []
    for i, title_i in enumerate(cluster_titles):
        for j, title_j in enumerate(cluster_titles):
            if i < j:
                similarity = similarity_matrix[i, j]
                similarity_pairs.append(((title_i, title_j), similarity))

    # Sort the pairs by similarity in descending order
    similarity_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get the top 50 titles most similar to the searched title
    top_similar_titles = []
    added_titles = set()  # Initialize a set to keep track of titles that have been added
    for pair, similarity in similarity_pairs[:50]:
        title_i, title_j = pair
        # Add the title that is not the searched title
        if title_i.lower() == searched_title.lower():
            if title_j not in added_titles:
                top_similar_titles.append((title_j, similarity))
                added_titles.add(title_j)
        else:
            if title_i not in added_titles:
                top_similar_titles.append((title_i, similarity))
                added_titles.add(title_i)

    # Print the top 50 titles with their similarity scores
    print(f"\nTop 10 Titles Similar to '{searched_title}':")
    for i, (title, similarity) in enumerate(top_similar_titles, start=1):
        print(f"{i}. {title} - Similarity: {similarity:.4f}")
else:
    print(f"Title '{searched_title}' not found in any cluster.")

import pandas as pd

scientific_names_df = pd.read_csv(r"C:\Users\USER\Downloads\FFAR NEW\top50_new_again.csv", encoding='latin1')


scientific_names = scientific_names_df['ScientificName'].tolist()

for searched_title in scientific_names:
    result = search_title(searched_title)

    if result:
        cluster_id, titles = result
#      print(f"Cluster ID for '{searched_title}': {cluster_id}")
#        print(f"Titles in Cluster {cluster_id}:\n{titles}")

        # Get vectors and titles for the specific cluster
        cluster_vectors = vectors_array[df['cluster'] == cluster_id]
        cluster_titles = df[df['cluster'] == cluster_id]['ScientificName']

        # Calculate cosine similarity matrix
        similarity_matrix = calculate_similarity(cluster_vectors)

        # Collect and sort the similarity pairs
        similarity_pairs = []
        for i, title_i in enumerate(cluster_titles):
            for j, title_j in enumerate(cluster_titles):
                if i < j:
                    similarity = similarity_matrix[i, j]
                    similarity_pairs.append(((title_i, title_j), similarity))

        # Sort the pairs by similarity in descending order
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)

        # Get the top 50 titles most similar to the searched title
        top_similar_titles = []
        added_titles = set()  # Initialize a set to keep track of titles that have been added
        for pair, similarity in similarity_pairs[:500]:
            title_i, title_j = pair
            # Add the title that is not the searched title
            if title_i.lower() == searched_title.lower():
                if title_j not in added_titles:
                    top_similar_titles.append((title_j, similarity))
                    added_titles.add(title_j)
            else:
                if title_i not in added_titles:
                    top_similar_titles.append((title_i, similarity))
                    added_titles.add(title_i)

        # Print the top 50 titles with their similarity scores
        print(f"\nTop 10 Titles Similar to '{searched_title}':")
        for i, (title, similarity) in enumerate(top_similar_titles, start=1):
            print(f"{i}. {title} - Similarity: {similarity:.4f}")
    else:
        print(f"Title '{searched_title}' not found in any cluster.")

import pandas as pd

# Initialize a dictionary to store similarity scores
similarity_dict = {}

# Create a list to store all unique scientific names
all_scientific_names = []

for searched_title in scientific_names:
    result = search_title(searched_title)

    if result:
        cluster_id, titles = result

        # Get vectors and titles for the specific cluster
        cluster_vectors = vectors_array[df['cluster'] == cluster_id]
        cluster_titles = df[df['cluster'] == cluster_id]['ScientificName']

        # Calculate cosine similarity matrix
        similarity_matrix = calculate_similarity(cluster_vectors)

        # Collect and sort the similarity pairs
        similarity_pairs = []
        for i, title_i in enumerate(cluster_titles):
            for j, title_j in enumerate(cluster_titles):
                if i < j:
                    similarity = similarity_matrix[i, j]
                    similarity_pairs.append(((title_i, title_j), similarity))

        # Sort the pairs by similarity in descending order
        similarity_pairs.sort(key=lambda x: x[1], reverse=True)

        # Get the top 50 titles most similar to the searched title
        top_similar_titles = []
        added_titles = set()  # Initialize a set to keep track of titles that have been added
        for pair, similarity in similarity_pairs[:50]:
            title_i, title_j = pair
            # Add the title that is not the searched title
            if title_i.lower() == searched_title.lower():
                if title_j not in added_titles:
                    top_similar_titles.append((title_j, similarity))
                    added_titles.add(title_j)
            else:
                if title_i not in added_titles:
                    top_similar_titles.append((title_i, similarity))
                    added_titles.add(title_i)

        # Store similarity scores in the dictionary
        similarity_dict[searched_title] = {title: similarity for title, similarity in top_similar_titles}

        # Update the list of all scientific names
        all_scientific_names.extend(cluster_titles)

    else:
        print(f"Title '{searched_title}' not found in any cluster.")

# Remove duplicates from the list of all scientific names
all_scientific_names = list(set(all_scientific_names))

# Create DataFrame with all scientific names as row names
similarity_df = pd.DataFrame(index=all_scientific_names)

# Add columns for the top 50 similar scientific names
for searched_title, similar_titles in similarity_dict.items():
    for similar_title, similarity in similar_titles.items():
        similarity_df.at[similar_title, searched_title] = similarity

# Save the DataFrame to an Excel file
similarity_df.to_excel(r"C:\Users\USER\Documents\Github\h4h-submit version\OUTPUT DATA OF MODELS\wiki_results_final_new.xlsx")

print("Results saved to wiki_results_final_new.xlsx file.")




