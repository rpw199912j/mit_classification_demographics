# MIT Classification Demographics
This repository is to accompany the supporting demographics analysis in the paper 
"A Database and Machine Learning Model to Identify Thermally Driven Metal-Insulator Transition Compounds".

# Research Question
This project involves the use of a survey sent out to around 200 people (e.g. materials scientists, physics, chemists) to see how well domain experts 
will do when classifying the Metal-Insulator Transition (MIT) compounds and to examine whether or not MIT classification is a trivial task for human experts.

# Workflow
## Data procurement
The survey contained 18 chemical compounds, including 6 metals, 7 insulators and 5 MIT compounds. 
For each compound, the respondents were asked to perform 3 tasks:  

1. classify the compound as either a metal or an insulator
2. determine if the compound exhibits MIT behavior
3. identify chemical and structural descriptors used in  determining  the  conductivity  class  of  the  materials.

A list of 11 descriptors were provided in the survey and respondents were asked to add in descriptors 
they used that were not already provided.

Please refer to [this section](https://github.com/rpw199912j/mit_classification_demographics#supporting-information) 
for the complete list of compounds and descriptors used in the survey.

## Data cleaning
After the [raw survey data](https://github.com/rpw199912j/mit_classification_demographics/tree/master/data/unprocessed/material_conductivity_survey.csv) 
was imported, a data cleaning process was carried out to tidy the dataset and to prepare it for 
later post-processing. This step used several packages from the [Tidyverse](https://www.tidyverse.org/) collection as implemented in the R programming language.

The data cleaning script used can be found [here](https://github.com/rpw199912j/mit_classification_demographics/tree/master/data_cleaning.R). 
Even better, you can immediately start a RStudio interface right in your web browser without installing any dependecies by clicking
on the `launch binder` icon below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_classification_demographics/master?urlpath=rstudio)

## Data analysis
After the data cleaning process was complete, the [processed datasets](https://github.com/rpw199912j/mit_classification_demographics/tree/master/data/processed/) were used to analyze the classification accuracy 
by different demographic groups. The descriptor usage was also analyzed to see what physical and chemical descriptors were used when
people were classifying MIT materials.

The Jupyter notebook used for analysis can be found [here](https://github.com/rpw199912j/mit_classification_demographics/tree/master/mit_classification_survey_analysis.ipynb). 
Just like before, you can also launch an interactive JupyterLab notebook right in your web browser by clicking on the `launch binder` icon below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rpw199912j/mit_classification_demographics/master?urlpath=lab/tree/mit_classification_survey_analysis.ipynb)
# Supporting Information
## Compounds
|Metals|Insulators|MITs|
|:-----|:---------|:---|
|LaRuO3|LaFeO3|CaFeO3|
|LaNiO3|MoO3|Ca2RuO4|
|ReO3|MnO|NbO2|
|MoO2|Sr2TiO4|Ti2O3|
|TiO|Cr2O3|BaVS3|
|SrCrO3|KVO3| |
| |Ag2BiO3| |'

## Descriptors
|Descriptors provided|
|:---------------|
|Stoichiometry|
|Crystal structure (e.g.  perovskite, rock salt, rutile)|
|Average metal-oxygen bond distance|
|Total number of valence electrons|
|d electron count|
|Mass density|
|Mean electronegativity of elements in formula|
|Polarizability of the compound|
|Standard deviation of average ionic radius of elements|
|Crystal field splitting energy|
|Electronic correlation|