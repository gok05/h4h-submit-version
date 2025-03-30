import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging
# from nltk.tokenize import word_tokenize
from collections import Counter
# from sklearn.cluster import KMeans

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\FPIDatabase_RA v01_doc2vec.xlsx"

if os.path.exists(file_path):
    df = pd.read_excel(file_path)
else:
    print(f"File not found: {file_path}")
    exit()

df

words_to_remove = ["plant", "plants", "specie", 'flower', 'reference', 'external', 'links', 'also', 'var', 'x000bthe', 'cm', 'leaf', 'long', 'grows', 'used']

# Preprocess the data and create TaggedDocument instances
documents = []
all_words = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()  # reduce a word to its base or dictionary form

# Define a translation table to remove punctuation
translator = str.maketrans('', '', string.punctuation)

for index, row in df.iterrows():
    common_name = str(row['CommonNames']) if not pd.isnull(row['CommonNames']) else ''
    scientific_name = str(row['ScientificName']) if not pd.isnull(row['ScientificName']) else ''
    description = str(row['Description']) if not pd.isnull(row['Description']) else ''
    use = str(row['Use']) if not pd.isnull(row['Use']) else ''
    cultivation = str(row['Cultivation']) if not pd.isnull(row['Cultivation']) else ''
    distribution = str(row['Distribution']) if not pd.isnull(row['Distribution']) else ''
    status = str(row['Status']) if not pd.isnull(row['Status']) else ''

    # Combine columns for better representation
    combined_text = f"{description.lower()} {use.lower()} {cultivation.lower()} {distribution.lower()} {status.lower()}"
    
    #{scientific_name.lower()}
    # Tokenize the text and lemmatize
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in nltk.word_tokenize(combined_text.translate(translator))
        if word.lower() not in stop_words and not any(char.isdigit() for char in word)
  
    ]

    # Remove specific words
    for word_to_remove in words_to_remove:
        words = [word for word in words if word != word_to_remove.lower()]

    documents.append(TaggedDocument(words, [str(index)]))
    all_words.extend(words)

corpus = documents

corpus[3]

n_docs = len(corpus)
print("Number of documents in the corpus:", n_docs)

word_frequency_dict = Counter(all_words)

most_common_words = Counter(all_words).most_common(10)
print("\nMost Common Words:")
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")

model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=50)
model.build_vocab(corpus)
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

vectors = [model.dv[index] for index in range(len(df))]

vectors

###############GENSIM

from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [" ".join(doc.words) for doc in corpus]

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

dictionary = corpora.Dictionary([doc.split() for doc in documents])

corpus_gensim = [dictionary.doc2bow(doc.split()) for doc in documents]

# Define the number of topics
n_topics = 500

# Build the LDA model
lda_model = models.LdaModel(corpus_gensim, num_topics=n_topics, id2word=dictionary)

#calculating model perplexity

perplexity = lda_model.log_perplexity(corpus_gensim)

print(perplexity)

from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=[all_words], dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)

# Print the topics
print("\nTopics:")
for topic_id, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {topic_id}: {topic}")

topic_documents = {i: [] for i in range(n_topics)}

# Search for documents belonging to each topic based on the scientific name
for doc_index, doc in enumerate(corpus):
    # Get the topic distribution for the document
    doc_topics = lda_model[corpus_gensim[doc_index]]
    # Sort the topics by their probability in the document
    sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
    # Check if there are topics associated with the document
    if sorted_topics:
        # Get the most dominant topic ID
        dominant_topic_id = sorted_topics[0][0]
        # Append the document to the corresponding topic
        topic_documents[dominant_topic_id].append(doc)

# Print documents for each topic
for topic_id, documents in topic_documents.items():
    print(f"\nDocuments for Topic {topic_id}:")
    for doc_tuple in documents:
        scientific_name = doc_tuple[0]
        description = doc_tuple[1]
        print(f" - {scientific_name}: {description}")

# Group documents by topics
topics_documents = {i: [] for i in range(n_topics)}
for doc_index, doc_topics in enumerate(lda_model.get_document_topics(corpus_gensim)):
    if doc_topics:  # Check if the document has a non-empty topic distribution
        dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
        topics_documents[dominant_topic].append(doc_index)

# Show the contents of documents for each topic
for topic_id, document_indices in topics_documents.items():
    print(f"\nDocuments for Topic {topic_id}:")
    for doc_index in document_indices[:5]:
        if 0 <= doc_index < len(corpus):
            print(f"Document Index: {doc_index}")
            print(f"Document Content: {corpus[doc_index]}")  # Print the original content of the document
            print("="*50)



# Scientific name to search
search_scientific_name = "Malus pumila"

# Find the topic of the document associated with the searched scientific name
search_topic = None
for topic, document_indices in topics_documents.items():
    for doc_index in document_indices:
        # Get the original content from the DataFrame
        row = df.iloc[doc_index]
        scientific_name = row['ScientificName']
        if scientific_name == search_scientific_name:
            search_topic = topic
            search_doc_index = doc_index  # Store the index of the searched document
            break
    if search_topic is not None:
        break

from scipy.sparse import csr_matrix

# List to store similarity scores
similarity_scores = []

# Search for the scientific name and determine its topic
topic_id = None
for topic, document_indices in topics_documents.items():
    for doc_index in document_indices:
        # Get the original content from the DataFrame
        row = df.iloc[doc_index]
        scientific_name = row['ScientificName']
        
        # If the scientific name matches, determine the topic and break the loop
        if scientific_name == search_scientific_name:
            topic_id = topic
            break
    if topic_id is not None:
        break

# If the scientific name is found
if topic_id is not None:
    # Get the document index of the searched scientific name
    search_doc_index = topics_documents[topic_id][0] 
    
    # Convert topic distribution vectors to dense arrays
    search_vector = lda_model[corpus_gensim[search_doc_index]]
    search_vector_dense = [prob for _, prob in search_vector]
    
    # Compute cosine similarity between the searched document and all other documents
    for doc_index in range(len(corpus)):
        if doc_index != search_doc_index:  # Exclude the searched document itself
            # Convert topic distribution vectors to dense arrays
            doc_vector = lda_model[corpus_gensim[doc_index]]
            doc_vector_dense = [prob for _, prob in doc_vector]
            
            # Compute cosine similarity if the dimensions are compatible
            if len(search_vector_dense) == len(doc_vector_dense):
                similarity_score = cosine_similarity([search_vector_dense], [doc_vector_dense])[0][0]
                similarity_scores.append((doc_index, similarity_score))
    
    # Sort similarity scores based on similarity score
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 most similar documents
    print(f"Top 10 Most Similar Documents to '{search_scientific_name}':")
    for i, (doc_index, similarity_score) in enumerate(similarity_scores[:10], 1):
        # Get the original content from the DataFrame
        row = df.iloc[doc_index]
        other_scientific_name = row['ScientificName']
        common_names = row['CommonNames']
        description = row['Description']
        
        # Print document details
        print(f"\nDocument {i}")
        print(f"Scientific Name: {other_scientific_name}")
        print(f"Common Names: {common_names}")
        print(f"Description: {description}")
        print(f"Similarity Score: {similarity_score}")
        print("="*50)
else:
    print(f"The scientific name '{search_scientific_name}' was not found in any document.")

# from scipy.sparse import csr_matrix

# Function to search using a scientific name and find similar scientific names within the same topic
def search_scientific_name(lda_model, corpus_gensim, dictionary, df, scientific_name):
    # Search for the scientific name in the DataFrame
    scientific_name = scientific_name.lower()
    scientific_name_row = df[df['ScientificName'].str.lower() == scientific_name]
    
    if not scientific_name_row.empty:
        scientific_name_index = scientific_name_row.index[0]
        
        # Determine the topic of the searched scientific name
        query_vector = lda_model[corpus_gensim[scientific_name_index]]
        query_topic = max(query_vector, key=lambda x: x[1])[0]
        
        # Retrieve other scientific names within the same topic
        topic_documents = topics_documents[query_topic]
        similar_names = []
        similarity_scores = []
        
        for doc_index in topic_documents:
            if doc_index != scientific_name_index:
                other_name = df.iloc[doc_index]['ScientificName']
                other_vector = lda_model[corpus_gensim[doc_index]]
                
                # Convert sparse matrices to dense arrays
                query_vector_dense = csr_matrix(query_vector)
                other_vector_dense = csr_matrix(other_vector)
                
                similarity = cosine_similarity(query_vector_dense, other_vector_dense)[0][0]
                
                similar_names.append(other_name)
                similarity_scores.append(similarity)
                
        return query_topic, similar_names, similarity_scores
    
    else:
        return None, None, None

import pandas as pd

scientific_names_df = pd.read_csv(r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\top50_new_again.csv", encoding='latin1')

results_df = pd.DataFrame(columns=['Searched Scientific Name', 'Topic', 'Similar Scientific Names', 'Similarity Scores'])

for index, row in scientific_names_df.iterrows():
    search_query = row['ScientificName'] 

    topic, similar_names, similarity_scores = search_scientific_name(lda_model, corpus_gensim, dictionary, df, search_query)

    if topic is not None:
        combined_results = list(zip(similar_names, similarity_scores))

        # Sort combined results by similarity scores
        sorted_results = sorted(combined_results, key=lambda x: x[1], reverse=True)

        # Append the results 
        results_df = results_df.append({'Searched Scientific Name': search_query,
                                        'Topic': topic,
                                        'Similar Scientific Names': [name for name, _ in sorted_results],
                                        'Similarity Scores': [score for _, score in sorted_results]}, 
                                        ignore_index=True)
    else:
        # Handle case where scientific name is not found
        results_df = results_df.append({'Searched Scientific Name': search_query,
                                        'Topic': None,
                                        'Similar Scientific Names': [],
                                        'Similarity Scores': []}, 
                                        ignore_index=True)

# Extract unique 'Similar Scientific Names'
unique_names = set()
for names_list in results_df['Similar Scientific Names']:
    unique_names.update(names_list)

# Create a new DataFrame to store transposed results
transposed_df = pd.DataFrame(index=unique_names)

# Transpose the data
for index, row in results_df.iterrows():
    for name, score in zip(row['Similar Scientific Names'], row['Similarity Scores']):
        transposed_df.loc[name, row['Searched Scientific Name']] = score

output_excel_path = r"C:\Users\USER\Documents\Github\h4h-submit version\OUTPUT DATA OF MODELS\LDA_FPI_top_results_new.xlsx"
transposed_df.to_excel(output_excel_path)

print(output_excel_path)



