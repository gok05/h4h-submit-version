import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\FPIDatabase_Krakv01Corr (1).xlsx"
df = pd.read_excel(file_path)

print(df.head())

# Replace missing values with zeros
df.fillna(0, inplace=True)

df

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_features = scaler.fit_transform(df[['EnergyKcal', 'EnergyKJ', 'Moisture', 'Protein', 'Ironmg', 'Zincmg', 'ProVitAug', 'VitCmg']])

Z = linkage(scaled_features, method='ward', metric='euclidean')

threshold = 2
clusters = fcluster(Z, t=threshold, criterion='distance')

df['Cluster'] = clusters

print(clusters)

len(set(clusters))

df.query('Cluster == 10')

df.head()

len(clusters)

import numpy as np
import pandas as pd

# List of scientific names to search for
scientific_names_to_search = [
"Apium graveolens",
"Solanum tuberosum",
"Malus domestica",
"Musa × paradisiaca",
"Capsicum annuum",
"Oryza sativa",
"Zea mays",
"Spinacia oleracea",
"Brassica oleracea var. capitata",
"Daucus carota",
"Cocos nucifera",
"Phoenix dactylifera",
"Linum usitatissimum",
"Citrus limon",
"Citrullus lanatus",
#"Allium spp.",
"Capsicum annuum",
"Pisum sativum",
"Glycine max",
"Vigna radiata",
"Avena sativa",
"Carica papaya",
"Prunus persica",
"Ananas comosus",
#"Citrus spp.",
"Allium cepa",
"Phaseolus vulgaris",
"Fagopyrum esculentum",
"Brassica oleracea var. capitata",
"Lens culinaris",
"Citrus reticulata",
"Cenchrus americanus",
"Cucumis melo",
"Arachis hypogea",
"Brassica rapa",
"Triticum aestivum",
"Brassica oleracea var. capitata",
"Cichorium intybus",
#"Cucurbita spp.",
"Solanum melongena",
"Prunus domestica",
"Foeniculum vulgare",
"Ficus carica",
"Rheum rhabarbarum",
"Psidium guajava",
"Armoracia rusticana",
"Actinidia chinensis",
"Lactuca sativa",
"Abelmoschus esculentus",
"Pyrus communis",
"Pistacia vera",
"Secale cereale",
"Ipomoea batatas"
]


# List of all scientific names in the dataset
all_scientific_names = df['ScientificName Corrected'].tolist()

# Create a dictionary to store the cluster label for each scientific name
cluster_labels = {}
for name in all_scientific_names:
    row_index = df[df['ScientificName Corrected'] == name].index
    if len(row_index) > 0:
        row_index = row_index[0]
        cluster_labels[name] = df.loc[row_index, 'Cluster']

# Create a results matrix
results_matrix = []

# Iterate over each scientific name in the dataset
for scientific_name_row in all_scientific_names:
    row = []
    # Find the cluster label for the current scientific name in the row
    cluster_label_row = cluster_labels[scientific_name_row]
    # Iterate over each searched scientific name
    for scientific_name_col in scientific_names_to_search:
        # Find the cluster label for the current searched scientific name
        cluster_label_col = cluster_labels[scientific_name_col]
        # Check if both scientific names belong to the same cluster
        if cluster_label_row == cluster_label_col:
            row.append(1)
        else:
            row.append(0)
    results_matrix.append(row)

# Convert the results matrix to a DataFrame
results_df = pd.DataFrame(results_matrix, columns=scientific_names_to_search, index=all_scientific_names)

# Filter rows with all 0 values
results_df_filtered = results_df[(results_df != 0).any(axis=1)]

# Display the filtered results
print(results_df_filtered)


# Save the results DataFrame to an Excel file
results_excel_path = r"C:\Users\USER\Documents\Github\h4h-submit version\OUTPUT DATA OF MODELS\fpi_cluster_results_final_new.xlsx"
results_df_filtered.to_excel(results_excel_path)

print(f"Results saved to {results_excel_path}")



