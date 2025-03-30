import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
#import seaborn as sns

file_path = r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\Mean_Final_Verified.xlsx"
df = pd.read_excel(file_path)

print(df.head())

# Replace missing values with zeros
df.fillna(0, inplace=True)

df

features = ['ENERCOS(kcal)', 'WATER(g)', 'PROTCNT(g)', 'PROT-(g)', 'FAT(g)', 'FATCE(g)', 'FAT-(g)', 
            'CHOAVLDF(g)', 'CHOAVL(g)', 'CHOCDF(g)', 'SUGAR(g)', 'FIBC(g)', 'ASH(g)', 'FE(mg)', 
            'VITA_RAE(mcg)', 'VITA(mcg)', 'VITA-(mcg)', 'VITA-(IU)', 'DM(g)', 'PROTCNP(g)', 
            'CHOAVL-(g)', 'FIBTGLCS(g)', 'ID(mcg)', 'FACID(g)', 'FASAT(g)', 'FATRN(g)', 'AAT-(mg)', 
            'SUGAR-(g)', 'PROTCNA(g)', 'PROTLBL(g)', 'CHOLBL(g)', 'ENEREU(kcal)', 'PROTCCN(g)', 
            'FATRN(mg)', 'ENERCT(kcal)', 'FASAT(mg)']


data_rounded = df.round(decimals=2)

X = data_rounded[features]

X['CHOLBL(g)'] = X['CHOAVLDF(g)'].fillna(X['CHOCDF(g)']).fillna(X['CHOAVL(g)']).fillna(X['CHOAVL-(g)']).fillna(0)

X['FASAT(mg)'] *= 1000

X['FASAT(g)'] = X['FASAT(mg)'].fillna(0)

X['FAT(g)'] = X['FATCE(g)'].fillna(X['FAT-(g)']).fillna(0)

X['FATRN(mg)'] = X['FATRN(mg)'] * 1000

X['FATRN(g)'] = X['FATRN(mg)'].fillna(0)

X['FIBC(g)'] = X['FIBTGLCS(g)'].fillna(0)

X['PROTLBL(g)'] = X['PROTCNP(g)'].fillna(X['PROTCNA(g)']).fillna(X['PROTCNT(g)']).fillna(X['PROTCCN(g)']).fillna('PROT-(g)').fillna(0)

X['SUGAR(g)'] = X['SUGAR-(g)'].fillna(0)

X['VITA-(IU)'] = X['VITA-(IU)'] / 3.33

X['VITA(mcg)'] = X['VITA-(mcg)'].fillna(X['VITA-(IU)']).fillna(0)

X['ENEREU(kcal)'] = X['ENERCOS(kcal)'].fillna(X['ENERCT(kcal)']).fillna(0)


X.drop(['CHOAVLDF(g)','CHOCDF(g)','CHOAVL(g)','CHOAVL-(g)', 'FASAT(mg)', 'FATCE(g)', 'FAT-(g)', 'FATRN(mg)', 'PROTCNP(g)', 'PROTCNA(g)', 'PROTCNT(g)', 'PROTCCN(g)', 'PROT-(g)',
       'SUGAR-(g)', 'VITA-(mcg)', 'VITA-(IU)', 'ENERCOS(kcal)', 'ENERCT(kcal)', 'WATER(g)', 'VITA_RAE(mcg)', 'ASH(g)', 'FE(mg)', 'ID(mcg)', 'FACID(g)', 'DM(g)', 'FIBTGLCS(g)'], axis=1, inplace=True)

X.fillna(0, inplace=True)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_features = scaler.fit_transform(X)

Z = linkage(scaled_features, method='ward', metric='euclidean')

threshold = 2
clusters = fcluster(Z, t=threshold, criterion='distance')

df['Cluster'] = clusters

print(clusters)

len(set(clusters))

df.query('Cluster == 1')

df.head()

len(clusters)

plt.figure(figsize=(15, 8))
dendrogram(Z, labels=df['Species/Subspecies for Project'].values, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('Distance')
plt.xlabel('Food Names')
plt.xticks(rotation=90)
plt.show()

#import numpy as np
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
"Allium spp.",
"Capsicum annuum",
"Pisum sativum",
"Glycine max",
"Vigna radiata",
"Avena sativa",
"Carica papaya",
"Prunus persica",
"Ananas comosus",
"Citrus spp.",
"Allium cepa",
"Phaseolus vulgaris",
"Fagopyrum esculentum",
"Brassica oleracea var. capitata",
"Lens culinaris",
"Citrus × aurantium",
"Cenchrus americanus",
"Cucumis melo",
"Arachis hypogea",
"Brassica rapa",
"Triticum aestivum",
"Brassica oleracea var. capitata",
"Cichorium intybus",
"Cucurbita spp.",
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
all_scientific_names = df['Species/Subspecies for Project'].tolist()

# Create a dictionary to store the cluster label for each scientific name
cluster_labels = {}
for name in all_scientific_names:
    row_index = df[df['Species/Subspecies for Project'] == name].index
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
results_excel_path = r"C:\Users\USER\Documents\Github\h4h-submit version\OUTPUT DATA OF MODELS\cluster_results_final_new.xlsx"
results_df_filtered.to_excel(results_excel_path)

print(f"Results saved to {results_excel_path}")



