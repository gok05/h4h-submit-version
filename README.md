# About
This readme file contains two sections. The first section describes the model building process including the datasets used during model building. The second section contains information about the installation

# Model building

# manual ranking of crops with available information about their functionality (stabilizer, emulsifier, and thickening agent) as found in FOODB

1. conduct search in foodb
2. define criteria used for ranking (<list the criteria here>)
3. score each crop
4. rank

## Embedding and topicmodeling
## Input Data
FPIDatabase_RA v01_doc2vec.csv - input data for creating the doc2vec model (Description) and topic modeling (Description, Use, Cultivation, Distribution, and Status)

Collated Data (Filtered)_Preprocessed (1).xlsx - input data for heirarchical clustering

for_scraping.xlsx - input data for doc2vec



## training files
FPI_DOC2VEC_TOPIC_MODELING_GENSIM_LDA.py - python code for doc2vec and lda

-load the input data
- preprocessing
- document embedding (doc2vec)
- topic modeling (lda)
- retrieve vectors from doc2vec
- retrieve topic distribution for each document

## Output data/ results
The output data from training were used for the predictive modeling

- LDA_FPI_top_results_new.xlsx - contains similarity scores (using results from lda) between the top 50 crops (obtained from manual ranking)
  



## model files
- doc2vec model (fpi_doc2vec_gensim.pkl)
- lda model (fpi_lda_model.pkl)

## validation files

## Clustering
Three types of cluster models were 

1. clustering based on nutrients using FPI (heirarchical)
2. clustering based on nutrients from different sources (collated) (heirarchical)
3. clustering based on wikipedia (doc2vec+ kmeans)

## Input data
Nutrition data
Collated Data (Filtered)_Preprocessed (1).xlsx

## clustering model
- load the data input
- preprocess
- harmonize and rank the nutrients
- perform clustering

# Model output

# data output

cluster_results_final_new.xlsx

# Create docker image
docker build -t python h4h_automated .