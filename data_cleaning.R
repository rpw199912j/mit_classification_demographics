library(tidyverse)
library(stringr)
library(stringi)
library(janitor)

# load the csv files ------------------------------------------------------

individual_full_summary <- read_csv('data/unprocessed/material_conductivity_survey.csv') %>% 
  # No information was collected on email & name, so these columns are dropped
  select(-c('Collector ID','Email Address', 'First Name', 'Last Name', 'Custom Data 1')) %>% 
  clean_names(case = "snake")


# clean up the data set --------------------------------------------------------


# now we see that since the original csv files have the first two rows as the column heads, 
# and read_csv use the first row as the header, some columns do not have the name in the first row and are replaced with X..
# Get the positions / indices of columns that need to be replaced
replace_columns_boolean <- individual_full_summary %>% 
  colnames() %>% 
  str_detect('^x\\d{0,3}')

# find the columns need to be replaced
colnames(individual_full_summary)[replace_columns_boolean]

# new column names 
# select the first row and remove the column names
first_row <- slice(individual_full_summary, 1) %>% 
  unlist(use.names = FALSE)
new_column_names <- first_row[replace_columns_boolean]

# rename the columns with that start with X
names(individual_full_summary)[replace_columns_boolean] <- new_column_names
# drop the first row
individual_full_summary <- individual_full_summary[-1,] %>% 
  # remove columns completely composed of NA
  remove_empty("cols")

# see the column names of the first 14 columns
colnames(individual_full_summary[,1:14])

# examine the column type
glimpse(individual_full_summary)


demographics <- individual_full_summary[,1:14] %>% 
  # merge the "Graduate Student", "Postdoc", "Faculty", "Staff Scientist" columns
  unite("current_position", c("Graduate Student", "Postdoc", "Faculty", "Staff Scientist"), na.rm = TRUE, remove = TRUE)

# check if the merged columns has more than one value per observation
unique(demographics$highest_academic_degree_held)
unique(demographics$current_position)

# it seems that one person choose both postdoc and faculty as his/her current position, we keep faculty as the position
sum(demographics$current_position == "Postdoc_Faculty")
demographics$current_position[demographics$current_position == "Postdoc_Faculty"] <- "Faculty"

# join the score column from the quiz_summary tibble with the demographics tibble
demographics <- demographics %>% 
  # rename field_of_study and primary_research_type
  rename(materials_science = field_of_study, Experimental = primary_research_type) %>% 
  clean_names(case = "snake")



# When each respondent was asked to identify a compound as either an insulator or metal, the original csv recorded the responses as two columns 

# drop the columns on demographics
by_individual <- individual_full_summary[,-2:-14] %>%
  clean_names() %>% 
  group_by(respondent_id) %>% 
  group_split()

# check the first individual response dataframe in the list
test <- by_individual[[1]] %>% 
  clean_names() %>% 
  select(-c(last(colnames(.)))) %>% 
  clean_names()

test

# define the correct column names, formulas and labels
col_names <- c("formula", "predicted_label", "mit", "stoichiometry", "crystal_structure", "average_metal_oxygen_bond_distance", 
               "total_number_of_valence_electrons", "d_electron_count", "mass_density", "mean_electronegativity_of_elements_in_formula",
               "polarizability_of_the_compound", "standard_deviation_of_average_ionic_radius_of_elements", "crystal_field_splitting_energy", 
               "electronic_correlation", "other_please_specify")

correct_chemical_formulas <- c("LaRuO3", "LaFeO3",  "LaNiO3", "CaFeO3", "ReO3",   "MoO2",    
                              "MoO3",   "Ag2BiO3", "TiO",    "MnO",    "Ca2RuO4","NbO2",
                              "Sr2TiO4","Ti2O3",   "Cr2O3",  "KVO3",   "BaVS3",  "SrCrO3")

correct_labels <- c("metal",     "insulator", "metal",     "mit",       "metal", "metal",
                    "insulator", "mit",       "metal",     "insulator", "mit",   "mit",
                    "insulator", "mit",       "insulator", "insulator", "mit",   "metal")

# create a function to clean each sub-tibble
clean_tibble <- function(tibble){
  for (i in seq(1, 252, 14)) {
    temp_tibble <- tibble %>% 
      # select every 14 columns
      select(names(tibble)[i:(i+13)]) %>% 
      # get the formula
      mutate(formula = first(colnames(.))) %>% 
      # rename the first column as `metal`
      rename(metal = first(colnames(.))) %>% 
      select(formula, everything())
    
    # rename all the columns 
    names(temp_tibble) <- col_names
    
    if(i > 1){
      # after the first iteration, start binding tibbles by rows
      one_respondent <- bind_rows(one_respondent, temp_tibble)
    }else{
      one_respondent <- temp_tibble
    }
  }
  # change the formula into the correct naming convention
  one_respondent$formula <- correct_chemical_formulas
  one_respondent
}

# iterate the cleaning process over all dataframes in the list
for (tb_index in seq_along(by_individual)){
  tb <- by_individual[[tb_index]] %>% 
    clean_names() %>% 
    # drop the last last column
    select(-c(last(colnames(.)))) %>% 
    clean_names()
  
  # get the respondent id
  respondent <- tb$respondent_id
  
  tb <- tb %>% 
    select(-respondent_id) %>% 
    clean_tibble() %>% 
    add_column(respondent_id = rep(respondent, nrow(.))) %>% 
    add_column(true_label = correct_labels) %>% 
    select(respondent_id, formula, true_label, everything())
  
  if(tb_index > 1){
    # after the first iteration, start binding tibbles by rows
    all_respondents <- bind_rows(all_respondents, tb)
  }else{
    all_respondents <- tb
  }
}

all_respondents <- all_respondents %>% 
  # merge the response from the MIT or not question with that from the metal or insulator question
  mutate(predicted_label = case_when(
    true_label == "mit" & mit == "Yes" ~ "mit",
    TRUE ~ str_to_lower(predicted_label)
  )) %>% 
  # drop the MIT question response
  select(-mit)


# accuracy by individual
accuracy_by_individual <- all_respondents %>% 
  group_by(respondent_id) %>% 
  summarise(accuracy = mean(true_label == predicted_label)) %>% 
  arrange(desc(accuracy))

accuracy_by_individual

# accuracy by compound
all_respondents %>% 
  group_by(formula) %>% 
  summarise(accuracy = mean(true_label == predicted_label)) %>% 
  arrange(desc(accuracy))

# join the demographic data frame with the accuracy_by_individual data frame
demographics <- demographics %>% 
  left_join(accuracy_by_individual, by = "respondent_id") %>% 
  select(respondent_id, accuracy, everything())


# Store the cleaned csv files
write_csv(demographics, 'data/processed/demographics.csv')
write_csv(all_respondents, 'data/processed/all_respondents.csv')






