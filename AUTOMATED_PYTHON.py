import pandas as pd

# Read Excel files
df = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/OUTPUT DATA OF MODELS/wiki_results_final_new.xlsx")
df2 = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/OUTPUT DATA OF MODELS/LDA_FPI_top_results_new.xlsx")
df3 = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/OUTPUT DATA OF MODELS/cluster_results_final_new.xlsx")
df4 = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/OUTPUT DATA OF MODELS/fpi_cluster_results_final_new.xlsx")

# List of species to exclude
species_to_exclude = [
    "Apium graveolens", "Solanum tuberosum", "Malus domestica", "Musa × paradisiaca", "Capsicum annuum",
    "Oryza sativa", "Zea mays", "Spinacia oleracea", "Brassica oleracea var. capitata", "Daucus carota",
    "Cocos nucifera", "Phoenix dactylifera", "Linum usitatissimum", "Citrus limon", "Citrullus lanatus",
    "Allium spp.", "Capsicum annuum", "Pisum sativum", "Glycine max", "Avena sativa", "Carica papaya",
    "Prunus persica", "Ananas comosus", "Citrus sinensis", "Allium cepa", "Phaseolus vulgaris", "Fagopyrum esculentum",
    "Brassica oleracea", "Citrus reticulata", "Cucumis melo", "Arachis hypogea", "Brassica rapa", "Triticum aestivum",
    "Brassica oleracea var. capitata", "Cichorium intybus", "Cucurbita spp.", "Solanum melongena", "Prunus domestica",
    "Foeniculum vulgare", "Ficus carica", "Rheum rhabarbarum", "Psidium guajava", "Armoracia rusticana",
    "Actinidia chinensis", "Lactuca sativa", "Abelmoschus esculentus", "Pyrus communis", "Pistacia vera",
    "Secale cereale"
]

# Filter dataframes
dfs = [df, df2, df3, df4]
dfs = [df[df['CropSpecies'].notna() & ~df['CropSpecies'].isin(species_to_exclude)] for df in dfs]

# Replace NA values with 0
for df in dfs:
    df.fillna(0, inplace=True)

# Multiply numeric columns by weights
weights = [0.15, 0.15, 0.35, 0.35]
for i, df in enumerate(dfs):
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols] * weights[i]

# Merge data
df_merged = pd.concat(dfs).groupby("CropSpecies").sum().reset_index()
df_merged.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/merged_new.xlsx", index=False)

# Assign weights
df_weights = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/weights.xlsx")
df_weights['Score'] = ((len(df_weights) * 2) - df_weights['Rank'] + 1)
df_weights['Weight'] = df_weights['Score'] / df_weights['Score'].sum()
df_weights['Rank_U'] = df_weights['U_UN'].rank(ascending=False)
df_weights['Score_U'] = (len(df_weights) - df_weights['Rank_U'] + 1)
df_weights['Weight_U'] = df_weights['Score_U'] / df_weights['Score_U'].sum()
df_weights['Weight / 2'] = (df_weights['Weight'] + df_weights['Weight_U']) / 2
df_weights['Final_Rank'] = df_weights['Weight / 2'].rank(ascending=False)
df_weights.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/Final_Rank_new.xlsx", index=False)

# Transpose weights
df_transposed = df_weights[['Crops', 'Weight / 2']].set_index('Crops').T

df_transposed.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/Transposed_new.xlsx", index=False)

# Merge weights with original data
df_merged = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/merged_new.xlsx")
df_weights = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/Transposed_new.xlsx")
df_merged.iloc[0, 1:] = df_weights.iloc[0, :]
df_merged.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/merged_updated_new.xlsx", index=False)

# Multiply values by weights
df_updated = pd.read_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/merged_updated_new.xlsx")
weights = df_updated.iloc[0, 1:].values
df_values = df_updated.iloc[1:].copy()
df_values.iloc[:, 1:] = df_values.iloc[:, 1:].apply(lambda x: x * weights, axis=1)
df_values.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/AUTOMATED/MULTIPLIED.xlsx", index=False)

# Compute sum per CropSpecies
df_values['SUM'] = df_values.iloc[:, 1:].sum(axis=1)
df_values.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/SUM_new.xlsx", index=False)

# Merge common names
df_collated = pd.read_excel("C:/Users/USER/Desktop/H4H/Collated Data (Filtered).xlsx")
df_values = df_values.merge(df_collated[['CropSpecies', 'CommonName']], on='CropSpecies', how='left')
df_values = df_values.drop_duplicates(subset='CropSpecies', keep='first')
df_values.to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTIVE MODEL PROCESS OUTPUTS/commonname_new.xlsx", index=False)

# Sort by SUM and export top 20
df_sorted = df_values.sort_values(by='SUM', ascending=False)
df_sorted[['CropSpecies', 'CommonName_x', 'SUM']].to_excel("C:/Users/USER/Documents/Github/h4h-submit version/PREDICTED CROPS/Final_Rank_UnCommon_Crops_Top 20_new.xlsx", index=False)



