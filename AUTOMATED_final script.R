library(readxl)
library(writexl)
library(openxlsx)
library(dplyr)
library(tibble)
library(tidyr)
library(conflicted)
########conflicted#########################MERGED#################################################
df <- read_excel("C:/Users/USER/Downloads/FFAR NEW/THREE DATA/wiki_results_final_new.xlsx")
df2 <- read_excel("C:/Users/USER/Downloads/FFAR NEW/THREE DATA/LDA_FPI_top_results_new.xlsx")
df3 <- read_excel("C:/Users/USER/Downloads/FFAR NEW/THREE DATA/cluster_results_final_new.xlsx")
df4 <- read_excel("C:/Users/USER/Downloads/FFAR NEW/THREE DATA/fpi_cluster_results_final_new.xlsx")

# List of species to exclude
species_to_exclude <- c(
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
  #"Vigna radiata",
  "Avena sativa",
  "Carica papaya",
  "Prunus persica",
  "Ananas comosus",
  "Citrus sinensis",
  "Allium cepa",
  "Phaseolus vulgaris",
  "Fagopyrum esculentum",
  "Brassica oleracea",
  #"Lens culinaris",
  "Citrus reticulata",
  #"Cenchrus americanus",
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
  "Secale cereale"
  #"Ipomoea batatas"
)

df = filter(df, !(CropSpecies %in% species_to_exclude))
df2 = filter(df2, !(CropSpecies %in% species_to_exclude))
df3 = filter(df3, !(CropSpecies %in% species_to_exclude))
df4 = filter(df4, !(CropSpecies %in% species_to_exclude))

df[is.na(df)] <- 0
df2[is.na(df2)] <- 0
df3[is.na(df3)] <- 0
df4[is.na(df4)] <- 0


# Multiply NUMERIC values by weights
df_numeric_cols <- colnames(df)[sapply(df, is.numeric)]
df[df_numeric_cols] <- df[df_numeric_cols] * 0.15
df2[df_numeric_cols] <- df2[df_numeric_cols] * 0.15
df3[df_numeric_cols] <- df3[df_numeric_cols] * 0.35
df4[df_numeric_cols] <- df4[df_numeric_cols] * 0.35

merged_data <- bind_rows(df, df2, df3, df4) %>%
  group_by(CropSpecies) %>%
  summarise(across(everything(), ~sum(.)))

merged_file <- "C:/Users/USER/Downloads/FFAR NEW/merged_new.xlsx"
write_xlsx(merged_data, path = merged_file)

#########################ASSIGN WEIGHTS#########################################
df <- read_excel("C:/Users/USER/Downloads/FFAR NEW/weights.xlsx")

df$Score <- ((nrow(df) * 2) - df$Rank + 1)

sum_scores <- sum(df$Score)

df$Weight <- df$Score / sum_scores

df$Rank_U <- rank(-df$U_UN)

df$Score_U <- ((nrow(df)) - df$Rank_U + 1)

sum_scores_U <- sum(df$Score_U)

df$Weight_U <- (df$Score_U) / sum_scores_U

df$Sum <- df$Weight + df$Weight_U

df$`Weight / 2` <- df$Sum / 2

df$Final_Rank <- rank(-df$`Weight / 2`)

output_file <- "C:/Users/USER/Downloads/FFAR NEW/Final_Rank_new.xlsx"
write_xlsx(df, path = output_file)

##########################TRANSPOSED WEIGHTS#####################################

df_transposed <- as.data.frame(t(df[, c("Crops", "Weight / 2")]))

colnames(df_transposed) <- df_transposed[1, ]

df_transposed <- df_transposed[-1, ]

output_file <- "C:/Users/USER/Downloads/FFAR NEW/Transposed_new.xlsx"
write_xlsx(df_transposed, path = output_file)

##########################MERGED THE WEIGHTS TO THE OG DATA######################

merged <- read_excel("C:/Users/USER/Downloads/FFAR NEW/merged_new.xlsx")

# empty data frame
empty_row <- data.frame(matrix(ncol = ncol(merged), nrow = 1))
colnames(empty_row) <- colnames(merged)  # Set column names

# Combine empty row and OG data drame
merged <- rbind(empty_row, merged)

weights <- read_excel("C:/Users/USER/Downloads/FFAR NEW/Transposed_new.xlsx")

# Insert the weights data into the second row of the merged data
merged[1, 2:ncol(merged)] <- weights[1,]

output_file <- ("C:/Users/USER/Downloads/FFAR NEW/merged_updated_new.xlsx")
write_xlsx(merged, path = output_file)


#########################MULTIPLY THE VALUES TO WEIGHTS##########################
df <- read_excel("C:/Users/USER/Downloads/FFAR NEW/merged_updated_new.xlsx")

# Get the weights from the first row
weights <- as.numeric(df[1, -1])  # Exclude CropSpecies column
df_values <- df[-1, ]
df_values[, -1] <- sapply(df_values[, -1], as.numeric)

# Multiply each column by the corresponding weight
result_df <- df_values
for (i in 2:ncol(df_values)) {
  result_df[, i] <- df_values[, i] * weights[i-1]
}

# Add the CropSpecies column back
result_df$CropSpecies <- df$CropSpecies[-1]

result_df[is.na(result_df)] <- 0

output_file <- "C:/Users/USER/Downloads/FFAR NEW/MULTIPLIED.xlsx"
write_xlsx(result_df, path = output_file)


##########################FIND THE SUM OF EACH CROPSPECIES#######################

result_df$SUM <- rowSums(result_df[, -1])

print(result_df)

output_file <- "C:/Users/USER/Downloads/FFAR NEW/SUM_new.xlsx"
write_xlsx(result_df, path = output_file)



######################ADD COMMON NAME###########################################
# Insert an empty column named "CommonName" after the "CropSpecies" column
result_df <- cbind(result_df[, 1], CommonName = NA, result_df[, -1])

print(result_df)


collated_df <- read_excel("C:/Users/USER/Downloads/Collated Data (Filtered).xlsx")

# Merge the result_df with the collated_df based on CropSpecies
merged_df <- merge(result_df, collated_df, by.x = "CropSpecies", by.y = "CropSpecies", all.x = TRUE)

# Replace the existing CommonName values with the new ones
merged_df$CommonName.x <- ifelse(!is.na(merged_df$CommonName.y), merged_df$CommonName.y, merged_df$CommonName.x)

# Remove the unnecessary columns
merged_df <- merged_df[, -which(names(merged_df) %in% c("CommonName.y"))]

# Rename the CommonName column
names(merged_df)[names(merged_df) == "CommonName.x"] <- "CommonName"

# Remove duplicate entries keeping only the first occurrence
unique_merged_df <- merged_df[!duplicated(merged_df$CropSpecies), ]

# Display the updated dataframe
print(unique_merged_df)

output_file <- "C:/Users/USER/Downloads/FFAR NEW/commonname_new.xlsx"
write_xlsx(unique_merged_df, path = output_file)
########################SORT########################################################
unique_merged_df <- unique_merged_df[order(-unique_merged_df$SUM), ]
print(unique_merged_df)
head(unique_merged_df, n = 20)

output_file <- "C:/Users/USER/Downloads/FFAR NEW/Final_Rank_UnCommon_Crops_Top 20_new.xlsx"
write_xlsx(unique_merged_df %>% select(CropSpecies, CommonName, SUM), path = output_file)
#write_xlsx(unique_merged_df, path = output_file)

############################MATCHED NUTRIENTS#################################################
# Read the Excel files
data_copy <- read_excel("C:/Users/USER/Downloads/FFAR NEW/Mean_Final_Verified - Copy.xlsx")
data_copy2 <- read_excel("C:/Users/USER/Downloads/FFAR NEW/Final_Rank_UnCommon_Crops_Top 20_new.xlsx")

# Example: Summing up SUM values for each CropSpecies
data_copy2_agg <- data_copy2 %>%
  group_by(CropSpecies) %>%
  summarise(SUM = sum(SUM))

# Lookup and extract nutrient values from data_copy
matched_nutrients <- data_copy2_agg %>%
  inner_join(data_copy, by = "CropSpecies") %>%
  select(CropSpecies, SUM, everything()) %>%
  rename(NUTRIENT_SUM = SUM) # Rename the SUM column to NUTRIENT_SUM for clarity

# Write the matched nutrients to a new Excel file
write_xlsx(matched_nutrients, "C:/Users/USER/Downloads/FFAR NEW/Matched.xlsx")







# Load necessary libraries
library(tidyverse)
library(readxl)

# Function to count non-missing values in a vector
count_non_missing <- function(x) {
  return(sum(!is.na(x)))
}

# Read the Excel files
data_copy <- read_excel("C:/Users/USER/Downloads/FFAR NEW/Mean_Final_Verified - Copy.xlsx")
data_copy2 <- read_excel("C:/Users/USER/Downloads/FFAR NEW/Final_Rank_Common_Crops_Top 20_new.xlsx")

# Ensure data_copy2 has a unique identifier for each row
data_copy2$id <- seq_len(nrow(data_copy2))

# Count non-missing values for each nutrient in data_copy2
data_copy2$non_missing_values <- apply(data_copy2[, grep("ENERCOS", names(data_copy2), value = TRUE)], 1, count_non_missing)

# Select one entry per crop species with the maximum number of non-missing values
selected_entries <- data_copy2 %>%
  group_by(CropSpecies) %>%
  slice_max(non_missing_values, with_ties = FALSE) %>%
  ungroup() %>%
  select(-non_missing_values, -id) # Remove helper columns

# Merge selected entries with original data_copy to get corresponding nutrient values
final_data <- left_join(selected_entries, data_copy, by = "CropSpecies")

# Save the final result to a new Excel file
write_xlsx(final_data, "C:/Users/USER/Downloads/FFAR NEW/Matched_new.xlsx")

# Print the final result to the console
print(final_data)

