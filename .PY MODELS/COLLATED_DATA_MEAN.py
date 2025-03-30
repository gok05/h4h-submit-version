#!/usr/bin/env python
# coding: utf-8


import pandas as pd

file_path = r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\Collated Data (Filtered)_Preprocessed (1).xlsx"
df = pd.read_excel(file_path)

df

sci_name = "Malus domestica"

filtered_df = df[df['Species/Subspecies Corrected'] == sci_name]

summary_stats = filtered_df.describe() #excludes missing values in the calculation

print("Filtered Data for", sci_name)
print(filtered_df)
print("\nSummary Statistics:")
print(summary_stats)

mean_values = filtered_df.mean()

print("Mean Values for", sci_name)
print(mean_values)

mean_by_species_type = df.groupby(['Species/Subspecies for Project', 'Plant Part - Unknown', 'Plant Part - Whole', 'Plant Part - Root', 'Plant Part - Shoot', 'Plant Part - Sprout', 'Plant Part - Stem / Trunk / Stalk / Petiole', 'Plant Part - Leaf', 'Plant Part - Flower', 'Plant Part - Fruit', 'Plant Part - Gum / Sap', 'Plant Part - Grain / Seed / Nut', 'Plant Part Config']).agg({'Foodname in English': 'first', 
                                                                       'ENERCOS(kcal)': 'mean',
                                                                       'WATER(g)': 'mean',
                                                                       'PROTCNT(g)': 'mean',
                                                                       'PROT-(g)': 'mean',
                                                                       'FAT(g)': 'mean',
                                                                       'FATCE(g)': 'mean',
                                                                       'FAT-(g)': 'mean',
                                                                       'CHOAVLDF(g)': 'mean',
                                                                       'CHOAVL(g)': 'mean',
                                                                       'CHOCDF(g)': 'mean',
                                                                       'SUGAR(g)': 'mean',
                                                                       'FIBC(g)': 'mean',
                                                                       'ASH(g)': 'mean',
                                                                       'FE(mg)': 'mean',
                                                                       'VITA_RAE(mcg)': 'mean',
                                                                       'VITA(mcg)': 'mean',
                                                                       'VITA-(mcg)': 'mean',
                                                                       'VITA-(IU)': 'mean',
                                                                       'DM(g)': 'mean',
                                                                       'PROTCNP(g)': 'mean',
                                                                       'CHOAVL-(g)': 'mean',
                                                                       'FIBTGLCS(g)': 'mean',
                                                                       'ID(mcg)': 'mean',
                                                                       'FACID(g)': 'mean',
                                                                       'FASAT(g)': 'mean',
                                                                       'FATRN(g)': 'mean',
                                                                       'AAT-(mg)': 'mean',
                                                                       'SUGAR-(g)': 'mean',
                                                                       'PROTCNA(g)': 'mean',
                                                                       'PROTLBL(g)': 'mean',
                                                                       'CHOLBL(g)': 'mean',
                                                                       'ENEREU(kcal)': 'mean',
                                                                       'PROTCCN(g)': 'mean',
                                                                       'FATRN(mg)': 'mean',
                                                                       'ENERCT(kcal)': 'mean',
                                                                       'FASAT(mg)': 'mean'
                                                                       
                                                                      })

print("Mean Values by Species/Subspecies and Identifiers:")
print(mean_by_species_type)

mean_by_species_type.to_excel(r"C:\Users\USER\Documents\Github\h4h-submit version\INPUT DATA FOR MODELS\Mean_Final_Verified.xlsx")




