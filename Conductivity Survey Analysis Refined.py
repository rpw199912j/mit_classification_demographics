
# coding: utf-8

# # Overview
# This notebook contains the demographic results of the survey and statistical analysis on the accuracy of different groups of respondents. (e.g. Current Position, Field of study, Primary Research Type, Academic Degree)
# 
# ## Import modules and load the data set 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matminer.figrecipes.plot import PlotlyFig
import seaborn as sns
sns.set()


# ## A first look at the data set
# The dataset was compiled from the raw survey results collected from the SurveyMonkey. The loaded dataframe only contains complete responses.

# In[2]:


df = pd.read_csv('~/Downloads/Survey_Demographic.csv')

print(df.head())
print()
print(df.info())

# Total number of respondents
tn = df.shape[0]
print('\nThe total number of respondent is ' + str(tn))


# As we can see from the information printed above, there are 52 complete responses from the survey. Excluding the accuracy column, we have 12 other features that can be grouped into 4 more general categories, which are 
# 1. Current Position
# 2. Field of study
# 3. Primary Research Type
# 4. Academic Degree
# 
# ## Overview of accuracy distribution

# In[3]:


print('The minimum accuracy is {n1}\nThe maximum accuracy is {n2}'.format(n1=min(df['Accuracy']),
                                                                          n2=max(df['Accuracy'])))

f = sns.distplot(df['Accuracy'], bins=[40, 50, 60, 70, 80, 90], kde=False)
plt.xlabel('Accuracy (%)', fontsize=24)
plt.ylabel('Number of people', fontsize=24)
plt.savefig('Accuracy_distribution.pdf', dpi=400)


# The histogram above shows that among the 52 respondents, the accuracy distribution is:
# * 40%-50% : 15
# * 50%-60% : 10
# * 60%-70% : 14
# * 70%-80% : 11
# * 80%-90% : 2

# In[4]:


print('The average accuracy is: {n1}% with a std of ± {n2}%'.format(
    n1=round(df.Accuracy.mean(), 2), n2=round(df.Accuracy.std(), 2)))
print('The median accuracy is: {}%'.format(round(df.Accuracy.median(), 2)))


# # Demographic and Statistical Analysis
# ## Group by academic degree
# The academic degree category has 3 features:
# * BA/BS: Bachelor of Arts / Bachelor of Science
# * MS: Master
# * PhD

# In[5]:


def IQR(lst):
    return np.percentile(lst, 75) - np.percentile(lst, 25)


# In[6]:


df_MS = df[df['Academic_degree'] == 'MS']
df_B = df[df['Academic_degree'] == 'BA/BS']
df_PhD = df[df['Academic_degree'] == 'PhD']


# The accuracy distributions for the 3 features above are displayed below. The red dashed line represents the overall average accuracy.

# In[7]:


def get_dimension(ax):
    x_bottom, x_top = ax.get_xlim()
    
    y_bottom, y_top = ax.get_ylim()
    return x_bottom, y_top


# In[8]:


sns.reset_defaults()


# In[9]:


def group_boxplot(lst_of_features_df, group_name, reset_xticklables, figure_size=(8, 6), font_size=20, top_text="# of\npeople:", top_text_scale_factor=1.015, dashed_line_value=df.Accuracy.mean()):
    # Convert a list of dataframes into a list of accuracies
    lst_of_features = [df.Accuracy for df in lst_of_features_df]

    # Set the plot size
    f, ax = plt.subplots(figsize=figure_size)
    fig = sns.boxplot(data=lst_of_features)

    # Get the x coordinate of the left frame and y coordinate of the top frame
    x_bottom, y_top = get_dimension(ax)
    y_top *= top_text_scale_factor

    # Draw a red dashed line across the plot, representing the overall mean accuracy by individual
    plt.axhline(dashed_line_value, color='r', linestyle='dashed', linewidth=2)
    plt.xlabel(group_name, size=font_size)
    plt.ylabel('Accuracy (%)', size=font_size)
    plt.text(x_bottom, y_top, top_text, size=font_size,
             horizontalalignment="right")

    # Label the number of people in each feature group
    for position in range(len(lst_of_features)):
        ax.text(position, y_top, str(
            len(lst_of_features[position])), size=font_size, horizontalalignment='center')
    ax.tick_params(labelsize=font_size)
    ax.set_xticklabels(reset_xticklables)


# In[10]:


group_boxplot([df_B, df_MS, df_PhD], "Academic degree", ["BA/BS", "MS", "PhD"])


# To compute the average accuracy for each feature, 3 separate dataframes are created, eaching containing only one of the three features above.

# In[11]:


def print_accuracy_report(lst_of_features_df, names):
    """
    Argument: list of dataframe
    Output: Print a summary of each feature's median accuracy
            along with interquartile range
    """
    for index, df in enumerate(lst_of_features_df):
        accuracies = df.Accuracy
        median_accuracy = accuracies.median()
        IQR_accuracy = IQR(accuracies)
        print("{name}_median_accuracy: {median}%, IQR: {IQR}%, Number_of_{name}: {count}".format(
            name=names[index], median=round(median_accuracy, 2), IQR=round(IQR_accuracy, 2), count=len(accuracies)))


# In[12]:


print_accuracy_report([df_B, df_MS, df_PhD], ["BA/BS", "MS", "PhD"])


# As seen from the results, the PhD category has both the highest accuracy and the largest number of people.
# 
# ## Group by primary research type
# The primary research type category has 3 features:
# * Experimental
# * Computational
# * Theoretical

# In[13]:


df_expe = df[df['Experimental'] == 1]
df_comp = df[df['Computational'] == 1]
df_theo = df[df['Theoretical'] == 1]


# The accuracy distributions for the 3 features above are displayed below.

# In[14]:


group_boxplot([df_expe, df_comp, df_theo], "Primary research type", [
              "Experimental", "Computational", "Theoretical"])


# Using the same procedures as the analysis on academic degree category, the average accuracies the number of people for the 3 features in primaty research type are printed below.

# In[15]:


print_accuracy_report([df_expe, df_comp, df_theo], [
                      "Experimental", "Computational", "Theoretical"])


# The _theoretical_ feature has the highest accuracy, but it also has the smallest number of people.
# 
# ## Group by field of study
# The field of study has 3 features:
# * Mat_sci
# * Physics
# * Chemistry

# In[16]:


df_mat_sci = df[df['Material_Science'] == 1]
df_physics = df[df['Physics'] == 1]
df_chem = df[df['Chemistry'] == 1]


# The accuracy distributions for the 3 features above are displayed below.

# In[17]:


group_boxplot([df_physics, df_mat_sci, df_chem], "Field of study", [
              "Physics", "Material Science", "Chemistry"])


# In[18]:


print_accuracy_report([df_physics, df_mat_sci, df_chem], [
    "Physics", "Material Science", "Chemistry"])


# ## Group by current position
# The current position category has 4 features:
# * Graduate
# * Postdoc
# * Faculty
# * Staff_scientist

# In[19]:


df_grad = df[df['Graduate'] == 1]
df_postdoc = df[df['Postdoc'] == 1]
df_faculty = df[df['Faculty'] == 1]
df_staff_scientist = df[df['Staff_Scientist'] == 1]


# The accuracy distributions for the 4 features above are displayed below.

# In[20]:


group_boxplot([df_grad, df_postdoc, df_faculty, df_staff_scientist],
              "Current position", ["Graduate", "Postdoc", "Faculty", 'Staff Scientist'])


# In[21]:


print_accuracy_report([df_grad, df_postdoc, df_faculty, df_staff_scientist], [
                      "Graduate", "Postdoc", "Faculty", 'Staff Scientist'])


# ## Graphical representation of the demographic data
# The accuracies of all 12 features are compiled into a list and reshaped.

# In[22]:


accuracy_list = [df_MS.Accuracy.median(), df_B.Accuracy.median(), df_PhD.Accuracy.median(), df_expe.Accuracy.median(), df_comp.Accuracy.median(), df_theo.Accuracy.median(), df_mat_sci.Accuracy.median(),
                 df_physics.Accuracy.median(), df_chem.Accuracy.median(), df_grad.Accuracy.median(), df_postdoc.Accuracy.median(), df_faculty.Accuracy.median(), df_staff_scientist.Accuracy.median()]
accuracy_list_np = np.reshape(accuracy_list, (1, -1))[0]


# The corresponding names of these features are also complile into a list.

# In[23]:


accuracy_name = ['MS', 'BA/BS', 'PhD', 'Experimental', 'Computational', 'Theoretical',
                 'Mat_sci', 'Physics', 'Chemistry', 'Graduate', 'Postdoc', 'Faculty', 'Staff_Scientist']
accuracy_name_np = np.array(accuracy_name)


# In[24]:


print(accuracy_list_np)


# Because the list containing all the accuracies are not yet ranked from the highest to the lowest, a sort function is used to get the indices of the accuracies values ranking from the highest to the lowest.

# In[25]:


indices = np.argsort(accuracy_list)[::-1]


# Reorganize the name list and the values list using the ranked indices.

# In[26]:


X = accuracy_name_np[indices]
Y = accuracy_list_np[indices]
print(X)


# The lists are then divided into 4 sub-lists, corresponding to the 4 categories mentioned in the previous section.

# In[27]:


X_position = X[[0, 2, 5, 10]]
X_research_type = X[[1, 7, 8]]
X_degree = X[[3, 9, 12]]
X_field = X[[4, 6, 11]]


# In[28]:


Y_position = Y[[0, 2, 5, 10]]
Y_research_type = Y[[1, 7, 8]]
Y_degree = Y[[3, 9, 12]]
Y_field = Y[[4, 6, 11]]


# In[29]:


pf = PlotlyFig(y_title='Average Accuracy (%)',
               title='Feature by type',
               fontsize=20,
               mode='notebook',
               ticksize=15)


# Plot the bar graph.
# 
# **Note:** The graphs are interactive. Move the cursor on top of each bar will show the specific value for the corresponding bar. Single-click the color label on the right to hide that category. Double-click the color label to display only that category.

# In[30]:


pf.bar(x=[X_position, X_research_type, X_degree, X_field],
       y=[Y_position, Y_research_type, Y_degree, Y_field],
       labels=['Current Position', 'Primary Research Type',
               'Highest Academic Degree', 'Field of Study'])


# The bar graph without dividing the name and accuracy lists into sub-lists.

# In[31]:


pf.bar(x=X, y=Y)


# # Compound Classification
# The survey also yields results on how well human researches classify the conductivity class of an unknown compounds and on what descriptors they use most or least often.
# ## Load the new dataset

# In[32]:


cc = pd.read_csv("~/Downloads/Compound_Classification_Complete.csv")

print(cc.head())
print()
print(cc.info())


# There are 18 compounds in total, 6 of which are metals, 6 of which are insulators, and the rest exhibit metal-to-insulator transition (MIT).
# ## Divide the dataframe into 3 conductivity class subsets.

# In[33]:


cc_metal = cc[cc['Label'] == 1]
cc_insulator = cc[cc['Label'] == 0]
cc_mit = cc[cc['Label'] == 2]


# The accuracy distributions for each conductivity class are displayed below.

# In[34]:


group_boxplot([cc_insulator*100, cc_metal*100, cc_mit*100],
              "", ["Insulator", "Metal", "MIT"], top_text="#:", dashed_line_value=cc.Accuracy.mean()*100)


# Compute the average accuracy for each subset.

# In[35]:


print('Overall median accuracy by compound: {n1}% ± {n2}%'.format(
    n1=round(cc.Accuracy.median()*100, 2), n2=round(IQR(cc.Accuracy)*100, 2)))
for name, accuracy_per_class in zip(["Insulator", "Metal", "MIT"], [cc_insulator.Accuracy, cc_metal.Accuracy, cc_mit.Accuracy]):
    print("{class_name}_median_accuracy: {accuracy}% ± {IQR}%".format(class_name=name, accuracy=round(
        accuracy_per_class.median()*100, 2), IQR=round(IQR(accuracy_per_class)*100, 2)))


# The insulator class has the best classification accuracy, while the metal has the lowest.

# ## Descriptor usage analysis
# Irrelevant columns are dropped. The remaining subset dataframes each contain 11 descriptor columns. The number in each column represents the number of times this descriptor are selected as an important feature in determining the conductivity class of the corresponding compound. The numbers of appearance as important feature are then summed for each conductivity class.

# In[36]:


metal_series = cc_metal.drop(['Formula', 'Accuracy', 'Metal',
                              'Insulator', 'MIT', 'Label'], axis=1).sum(axis=0).sort_values(ascending=False)

insulator_series = cc_insulator.drop(['Formula', 'Accuracy', 'Metal',
                                      'Insulator', 'MIT', 'Label'], axis=1).sum(axis=0).sort_values(ascending=False)

mit_series = cc_mit.drop(['Formula', 'Accuracy', 'Metal',
                          'Insulator', 'MIT', 'Label'], axis=1).sum(axis=0).sort_values(ascending=False)


# In[37]:


print('Metals:\n{}\n'.format(metal_series))
print('Insulators:\n{}\n'.format(insulator_series))
print('MITs:\n{}\n'.format(mit_series))


# For each descriptor in one conductivity class, the maximum number for it to selected for one compound is 52, since there are 52 respondents in this dataframe. In each conductivity class, there are 6 compounds, so the maximum number possible for a descriptor to be selected is $52 * 6 =  306$

# In[38]:


total_appearance = tn*6


# The number of appeareances for the 11 descriptors in each class are reorganized into lists, along with the corresponding labels.

# In[39]:


metal_index = metal_series.index.tolist()
metal_index_relabeled = [x+'_1' for x in metal_index]
metal_values = metal_series.values
metal_values_recalculated = [x/total_appearance*100 for x in metal_values]

insulator_index = insulator_series.index.tolist()
insulator_index_relabeled = [x+'_0' for x in insulator_index]
insulator_values = insulator_series.values
insulator_values_recalculated = [x/total_appearance*100 for x in insulator_series]

MIT_index = mit_series.index.tolist()
MIT_index_relabeled = [x+'_2' for x in MIT_index]
MIT_values = mit_series.values
MIT_values_recalculated = [x/total_appearance*100 for x in MIT_values]


# ### 3 most-often used descriptors

# 3 most-often used descriptors for each conductivity class are sliced from the original lists.

# In[40]:


n_descriptors = 3

X_metal_most = metal_index_relabeled[:n_descriptors]
Y_metal_most = metal_values_recalculated[:n_descriptors]

X_insulator_most = insulator_index_relabeled[:n_descriptors]
Y_insulator_most = insulator_values_recalculated[:n_descriptors]

X_mit_most = MIT_index_relabeled[:n_descriptors]
Y_mit_most = MIT_values_recalculated[:n_descriptors]


# In[41]:


pf_2 = PlotlyFig(y_title='Appearance Frequency (%)',
                 title='Descriptors used most often',
                 fontsize=20,
                 mode='notebook',
                 ticksize=15)


# In[42]:


pf_2.bar(x=[X_metal_most, X_insulator_most, X_mit_most],
         y=[Y_metal_most, Y_insulator_most, Y_mit_most],
         labels=['Metal', 'Insulator', 'MIT'])


# ### 3 least-often used descriptors

# 3 least-often used descriptors for each conductivity class are sliced from the original lists.

# In[43]:


X_metal_least = metal_index_relabeled[-n_descriptors:]
Y_metal_least = metal_values_recalculated[-n_descriptors:]

X_insulator_least = insulator_index_relabeled[-n_descriptors:]
Y_insulator_least = insulator_values_recalculated[-n_descriptors:]

X_mit_least = MIT_index_relabeled[-n_descriptors:]
Y_mit_least = MIT_values_recalculated[-n_descriptors:]


# In[44]:


pf_3 = PlotlyFig(y_title='Appearance Frequency (%)',
                 title='Descriptors used least often',
                 fontsize=20,
                 mode='notebook',
                 ticksize=15)


# In[45]:


pf_3.bar(x=[X_metal_least, X_insulator_least, X_mit_least],
         y=[Y_metal_least, Y_insulator_least, Y_mit_least],
         labels=['Metal', 'Insulator', 'MIT'])


# ### Descriptor usage by compound
# The compound classification dataframe is sorted using the values in the 'Accuracy' columns in descending order

# In[46]:


cc_sorted = cc.sort_values(by='Accuracy', axis=0, ascending=False)
print(cc_sorted.head(5))


# #### Descriptors usage for the 5 compounds with highest accuracy

# In[47]:


n_compound = 5

cc_sorted_5_most = cc_sorted.iloc[:n_compound, :]
cc_sorted_5_most_series = cc_sorted_5_most.drop(['Formula', 'Accuracy', 'Metal',
                                                 'Insulator', 'MIT', 'Label'], axis=1).sum(axis=0).sort_values(ascending=False)
print('{n_1} Most Accurate:\n{n_2}\n'.format(n_1=n_compound, n_2=cc_sorted_5_most_series))


# To convert the values in the series into percentage, the maximum number for each descriptor to be chosen is calculated as $52 \times 5 = 260$.

# In[48]:


total_appearance_2 = tn*5


# In[49]:


cc_sorted_5_most_series_index = cc_sorted_5_most_series.index.tolist()
cc_sorted_5_most_series_values = cc_sorted_5_most_series.values
cc_sorted_5_most_series_values_recalculated = [
    x/total_appearance_2*100 for x in cc_sorted_5_most_series_values]


# In[50]:


pf_5 = PlotlyFig(y_title='Appearance Frequency (%)',
                 title='Descriptors Usage Frequency of 5 Most Accurate Compounds',
                 fontsize=20,
                 mode='notebook',
                 ticksize=15)

pf_5.bar(x=cc_sorted_5_most_series_index,
         y=cc_sorted_5_most_series_values_recalculated)


# In[51]:


sns.reset_defaults()
x = range(11)
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(x, cc_sorted_5_most_series_values_recalculated)
plt.xticks(x, cc_sorted_5_most_series_index)
# Code modified from https://stackoverflow.com/questions/36220829/
# fine-control-over-the-font-size-in-seaborn-plots-for-academic-papers
for lable in ax.get_xticklabels():
    lable.set_ha("left")
    lable.set_rotation(-45)
ax.tick_params(labelsize=15)
plt.ylabel('Relative Frequency (%)', fontsize=15)
plt.title('Descriptor Usage of 5 Most Accurate Compounds', fontsize=15)
plt.tight_layout()
plt.savefig('5_most_accurate_compound.pdf', dpi=400)
print(type(cc_sorted_5_most_series_index))


# #### Descriptors usage for the 5 compounds with lowest accuracy

# In[52]:


cc_sorted_5_least = cc_sorted.iloc[-n_compound:, :]
cc_sorted_5_least_series = cc_sorted_5_least.drop(['Formula', 'Accuracy', 'Metal',
                                                   'Insulator', 'MIT', 'Label'], axis=1).sum(axis=0).sort_values(ascending=False)
print('{n_1} Least Accurate:\n{n_2}\n'.format(n_1=n_compound, n_2=cc_sorted_5_least_series))


# In[53]:


cc_sorted_5_least_series_index = cc_sorted_5_least_series.index.tolist()
cc_sorted_5_least_series_values = cc_sorted_5_least_series.values
cc_sorted_5_least_series_values_recalculated = [
    x/total_appearance_2*100 for x in cc_sorted_5_least_series_values]


# In[54]:


pf_6 = PlotlyFig(y_title='Appearance Frequency (%)',
                 title='Descriptors Usage Frequency of 5 Least Accurate Compounds',
                 fontsize=20,
                 mode='notebook',
                 ticksize=15)

pf_6.bar(x=cc_sorted_5_least_series_index,
         y=cc_sorted_5_least_series_values_recalculated)


# In[55]:


sns.reset_defaults()
x = range(11)
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(x, cc_sorted_5_least_series_values_recalculated)
plt.xticks(x, cc_sorted_5_least_series_index)
# Code modified from https://stackoverflow.com/questions/36220829/
# fine-control-over-the-font-size-in-seaborn-plots-for-academic-papers
for lable in ax.get_xticklabels():
    lable.set_ha("left")
    lable.set_rotation(-45)
ax.tick_params(labelsize=15)
plt.ylabel('Relative Frequency (%)', fontsize=15)
plt.title('Descriptor Usage of 5 Least Accurate Compounds', fontsize=15)
plt.tight_layout()
plt.savefig('5_least_accurate_compound.pdf', dpi=400)


# In[56]:


ir = pd.read_csv('/Users/jasonrpw/Downloads/Individual_responses.csv')
print(ir.head())


# ### Descriptor usage by individual
# #### Descriptors usage for the 5 people with highest accuracy

# In[57]:


ir_sorted = ir.sort_values(ascending=False, by=['Accuracy'])
ir_sorted_top_5 = ir_sorted.iloc[:18*5]


# In[58]:


top_5_correct = ir_sorted_top_5[ir_sorted_top_5['Correct'] == 1].drop(['Formula', 'Accuracy', 'Metal',
                                                                       'Insulator', 'MIT', 'Label',
                                                                       'Correct', 'Respondent'], axis=1).sum(axis=0).sort_values(ascending=False)


# In[59]:


top_5_wrong = ir_sorted_top_5[ir_sorted_top_5['Correct'] == 0].drop(['Formula', 'Accuracy', 'Metal',
                                                                     'Insulator', 'MIT', 'Label',
                                                                     'Correct', 'Respondent'], axis=1).sum(axis=0).sort_values(ascending=False)


# In[60]:


print('Top 5 Correct: \n{}\n'.format(top_5_correct))
print('Top 5 Wrong: \n{}\n'.format(top_5_wrong))


# In[61]:


def draw_frequency(series, title_name):
    X = series.index.tolist()
    Y = series.values
    Y_sum = sum(series.values)
    Y_recalculated = [x/Y_sum*100 for x in Y]
    pf = PlotlyFig(y_title='Appearance Frequency (%)',
                   title=title_name,
                   fontsize=20,
                   mode='notebook',
                   ticksize=15)
    pf.bar(x=X, y=Y_recalculated)


# In[62]:


draw_frequency(top_5_correct,
               'Descriptor usage appearance of the 5 most accurate people (Correct)')


# In[63]:


draw_frequency(top_5_wrong,
               'Descriptor usage appearance of the 5 most accurate people (Wrong)')


# #### Descriptors usage for the 5 people with lowest accuracy

# In[64]:


ir_sorted_bot_5 = ir_sorted.iloc[-18*5:]
bot_5_correct = ir_sorted_bot_5[ir_sorted_bot_5['Correct'] == 1].drop(['Formula', 'Accuracy', 'Metal',
                                                                       'Insulator', 'MIT', 'Label',
                                                                       'Correct', 'Respondent'], axis=1).sum(axis=0).sort_values(ascending=False)
bot_5_wrong = ir_sorted_bot_5[ir_sorted_bot_5['Correct'] == 0].drop(['Formula', 'Accuracy', 'Metal',
                                                                     'Insulator', 'MIT', 'Label',
                                                                     'Correct', 'Respondent'], axis=1).sum(axis=0).sort_values(ascending=False)


print('Bottom 5 Correct: \n{}\n'.format(bot_5_correct))
print('Bottom 5 Wrong: \n{}\n'.format(bot_5_wrong))


# In[65]:


draw_frequency(bot_5_correct,
               'Descriptor usage appearance of the 5 least accurate people (Correct)')


# In[66]:


draw_frequency(bot_5_wrong,
               'Descriptor usage appearance of the 5 least accurate people (Wrong)')


# ## Compounds ranked by prediction accuracy

# Obtain the accuracy values and the corresponding chemical formulas

# In[67]:


X_cc_sorted = cc_sorted.Formula.values
Y_cc_sorted = cc_sorted.Accuracy.values
Y_cc_sorted_recalculated = [x*100 for x in Y_cc_sorted]

print(X_cc_sorted)
print(Y_cc_sorted_recalculated)

X_cc_sorted_relabeled = ['Cr2O3_i','Sr2TiO4_i','MoO3_i','KVO3_i','LaFeO3_i','MnO_i',
                         'LaNiO3_m','Ca2RuO4_mit','BaVS3_mit','MoO2_m','Ti2O3_mit','LaRuO3_m',
                         'Ag2BiO3_mit','ReO3_m','NbO2_mit','TiO_m','CaFeO3_mit','SrCrO3_m']


# Plot a bar graph with the x-axis being the chemical formula and y-axis being the prediction accuracy

# In[68]:


pf_4 = PlotlyFig(y_title='Accuracy (%)',
                 title='Compounds',
                 fontsize=20,
                 mode='notebook',
                 ticksize=15)

pf_4.bar(x=X_cc_sorted_relabeled, y=Y_cc_sorted_recalculated)


# ## Per class accuracy distribution

# In[69]:


def class_accuracy_distribution(target_name):
    if target_name == 'MIT':
        target_number = 2
    elif target_name == 'Insulator':
        target_number = 0
    else:
        target_number = 1
        
    accuracies_list = []
    for i in np.arange(1, tn+1):
        df = ir[ir['Respondent'] == i]
        ir_class = df[df['Label'] == target_number]
        accuracy = ir_class[ir_class[target_name] == 1].shape[0] / ir_class.shape[0] * 100
        accuracies_list.append(round(accuracy,2))
        
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.distplot(accuracies_list, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], kde=False)
    plt.xlabel('Accuracy (%)', fontsize=24)
    plt.ylabel('Number of people', fontsize=24)
    plt.title(target_name + " accuracy distribution", fontsize = 32)


# ### MIT accuracy

# In[70]:


class_accuracy_distribution('MIT')


# ### Insulator accuracy

# In[71]:


class_accuracy_distribution('Insulator')


# ### Metal accuracy

# In[72]:


class_accuracy_distribution('Metal')


# # Comparison with Machine Learning Algorithm

# In[73]:


def print_table(table):
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))


# In[74]:


def print_report(target_name):
    if target_name == 'MIT':
        target_number = 2

    elif target_name == 'Insulator':
        target_number = 0

    elif target_name == 'Metal':
        target_number = 1

    tp = ir[(ir['Label'] == target_number) & (ir[target_name] == 1)]
    tn = ir[(ir['Label'] != target_number) & (ir[target_name] != 1)]
    fp = ir[(ir['Label'] != target_number) & (ir[target_name] == 1)]
    fn = ir[(ir['Label'] == target_number) & (ir[target_name] != 1)]

    n_tp = tp.shape[0]
    n_tn = tn.shape[0]
    n_fp = fp.shape[0]
    n_fn = fn.shape[0]

    p0 = round(n_tp / (n_tp + n_fp), 2)
    r0 = round(n_tp / (n_tp + n_fn), 2)
    f0 = round(2 * n_tp / (2 * n_tp + n_fp + n_fn), 2)
    s0 = n_tp + n_fn

    p1 = round(n_tn / (n_tn + n_fn), 2)
    r1 = round(n_tn / (n_tn + n_fp), 2)
    f1 = round(2 * n_tn / (2 * n_tn + n_fn + n_fp), 2)
    s1 = n_tn + n_fp

    w0 = s0 / (s0+s1)
    w1 = s1 / (s0+s1)

    a1 = round(p0*w0 + p1*w1, 2)
    a2 = round(r0*w0 + r1*w1, 2)
    a3 = round(f0*w0 + f1*w1, 2)
    a4 = s0 + s1
    
    title_list = ["","Actual_"+target_name, "Actual_non_"+target_name]
    predict_list = ["Predicted_"+target_name, n_tp, n_fp]
    predict_non_list = ["Predicted_non_"+target_name, n_fn, n_tn]
    
    classification_matrix = [title_list,
                             predict_list,
                             predict_non_list,]
    
    print_table(classification_matrix)

    #print("                   Actual_" + target_name + "   " + "Actual_non_" + target_name)
    #print("Predicted_{n3}:     {n1}          {n2}".format(n1=n_tp, n2=n_fp, n3=target_name))
    #print("Predicted_non_{n3}: {n1}          {n2}".format(n1=n_fn, n2=n_tn, n3=target_name))
    print()
    
    performance_category = ["","precison","recall","f1", "support"]
    if target_name == "MIT":
        non_target_list = ["Insulator+Metal", p1, r1, f1, s1]
    elif target_name == "Insulator":
        non_target_list = ["MIT+Metal", p1, r1, f1, s1]
    else:
        non_target_list = ["MIT+Insulator", p1, r1, f1, s1]
    
    target_list = [target_name, p0, r0, f0, s0]
    avg_total_list = ["avg/tol", a1, a2, a3, a4]
    
    classification_report = [performance_category,
                            non_target_list,
                            target_list,
                            avg_total_list,]
    
    print_table(classification_report)
    print()
    
    #print('\t', 'precision', '\t', 'recall', '\t', 'f1', '\t', 'support')
    #print('\t', p1, '\t', r1, '\t', f1, '\t', s1)
    #print('\t', p0, '\t', r0, '\t', f0, '\t', s0)
    #print('\t', a1, '\t', a2, '\t', a3, '\t', a4)
    print(target_name+"_vs_rest_accuracy: {}".format(round((n_tp+n_tn)/(n_tp+n_tn+n_fp+n_fn),4)))


# MIT vs. Rest

# In[75]:


print_report('MIT')


# Insulator vs. Rest

# In[76]:


print_report('Insulator')


# Metal vs. Rest

# In[77]:


print_report('Metal')


# In[78]:


print('Number of MIT: ' + str(ir[ir['Label'] == 2].shape[0]))
#print('Number of non-MIT: ' + str(ir[ir['Label'] != 2].shape[0]))
print('Number of Metal: ' + str(ir[ir['Label'] == 1].shape[0]))
#print('Number of non-Metal: ' + str(ir[ir['Label'] != 1].shape[0]))
print('Number of Insulator: ' + str(ir[ir['Label'] == 0].shape[0]))
#print('Number of non-Insulator: ' + str(ir[ir['Label'] != 0].shape[0]))


# In[79]:


def individual_report(target_number, target_name, ir):
    tp = ir[(ir['Label'] == target_number) & (ir[target_name] == 1)]
    tn = ir[(ir['Label'] != target_number) & (ir[target_name] != 1)]
    fp = ir[(ir['Label'] != target_number) & (ir[target_name] == 1)]
    fn = ir[(ir['Label'] == target_number) & (ir[target_name] != 1)]

    n_tp = tp.shape[0]
    n_tn = tn.shape[0]
    n_fp = fp.shape[0]
    n_fn = fn.shape[0]

    if (n_tp + n_fp != 0) & (n_tn + n_fn != 0):
        p0 = round(n_tp / (n_tp + n_fp), 2)
        r0 = round(n_tp / (n_tp + n_fn), 2)
        f0 = round(2 * n_tp / (2 * n_tp + n_fp + n_fn), 2)
        s0 = n_tp + n_fn
    elif n_tp + n_fp == 0:
        p0 = 0
        r0 = round(n_tp / (n_tp + n_fn), 2)
        f0 = round(2 * n_tp / (2 * n_tp + n_fp + n_fn), 2)
        s0 = n_tp + n_fn
    elif n_tn + n_fn == 0:
        p0 = round(n_tp / (n_tp + n_fp), 2)
        r0 = 0
        f0 = round(2 * n_tp / (2 * n_tp + n_fp + n_fn), 2)
        s0 = n_tp + n_fn

    return {'Precision': p0, 'Recall': r0, 'F1': f0}


# In[80]:


def print_class_report(target_name):
    precision_list = []
    recall_list = []
    f1_list = []

    if target_name == 'MIT':
        target_number = 2
    elif target_name == 'Insulator':
        target_number = 0
    elif target_name == 'Metal':
        target_number = 1

    for i in range(1, tn+1):
        df = ir[ir['Respondent'] == i]
        result = individual_report(target_number, target_name, df)
        precision_list.append(result['Precision'])
        recall_list.append(result['Recall'])
        f1_list.append(result['F1'])

    return {'Precision_list': precision_list, 'Recall_list': recall_list, 'F1_list': f1_list}


# In[81]:


def give_stat(lst, name, target_name):
    print('The median of {n1} is {n2}'.format(n1=target_name + ' ' + name, n2=round(np.median(lst),2)))
    print('The IQR of {n1} is {n2}'.format(n1=target_name + ' ' + name, n2=round(IQR(lst),2)))
    print()


# In[82]:


for i in ['MIT', 'Insulator', 'Metal']:
    temp = print_class_report(i)
    for j in ['Precision', 'Recall', 'F1']:
        give_stat(temp[j+'_list'], j, i)


# ## Human vs. ML Comparison Plot
# ### Setting up constants for the plot

# In[83]:


MIT_TITLE = "MIT vs. Rest"
METAL_TITLE = "Metal vs. Rest"
INSULATOR_TITLE = "Insulator vs. Rest"
TITLES = [MIT_TITLE, METAL_TITLE, INSULATOR_TITLE]

TICK_SIZE = 19
BARWIDTH = 0.2


# ### MIT vs. Rest

# In[84]:


# Index refer to the bar's x position on the graph
human_index = [0.1, 0.6, 1.1]
ml_index = [0.3, 0.8, 1.3]
# Precision, Recall, F1 for human scientists when classifying MIT vs. Rest
mit_human = [0.39, 0.67, 0.47]
# IQR for human scientists when classifying MIT vs. Rest
mit_human_IQR = [0.18, 0.5, 0.25]
# Divide the IQR into half to plot as error bars on the graph
mit_human_err = [i/2 for i in mit_human_IQR]


# In[85]:


# Precision, Recall, F1 for machine learning model when classifying MIT vs. Rest 
mit_ml = [0.90, 0.78, 0.82]
# IQR for machine learning model when classifying MIT vs. Rest
mit_ml_IQR = [0.20, 0.39, 0.23]
# Divide the IQR into half to plot as error bars on the graph
mit_ml_err = [i/2 for i in mit_ml_IQR]


# ### Metal vs. Rest

# In[86]:


metal_human = [0.44, 0.67, 0.5]
metal_human_IQR = [0.2, 0.34, 0.22]
metal_human_err = [i/2 for i in metal_human_IQR]

metal_ml = [0.60, 0.50, 0.50]
metal_ml_IQR = [0.38, 0.33, 0.27]
metal_ml_err = [i/2 for i in metal_ml_IQR]


# ### Insulator vs. Rest

# In[87]:


insulator_human = [0.44, 0.83, 0.57]
insulator_human_IQR = [0.25, 0.33, 0.25]
insulator_human_err = [i/2 for i in insulator_human_IQR]

insulator_ml = [0.79, 0.92, 0.84]
insulator_ml_IQR = [0.17, 0.14, 0.12]
insulator_ml_err = [i/2 for i in insulator_ml_IQR]


# In[88]:


human = [mit_human, metal_human, insulator_human]
ml = [mit_ml, metal_ml, insulator_ml]

human_err = [mit_human_err, metal_human_err, insulator_human_err]
ml_err = [mit_ml_err, metal_ml_err, insulator_ml_err]


# ### Plot three binary class comparison vertically

# In[89]:


f, axarr = plt.subplots(3, sharex=True, sharey=True, figsize=(7, 9))

# Plot the three subplots in the order of "MIT", "Metal", "Insulator"
for i in range(3):
    axarr[i].bar(human_index, human[i], width=BARWIDTH, yerr=human_err[i], color="g")
    axarr[i].bar(ml_index, ml[i], width=BARWIDTH, yerr=ml_err[i], color="b")
    if i == 0:
        axarr[i].set_ylim(0, 1.0)
        axarr[i].legend(('Human', 'Computer'), frameon=True, bbox_to_anchor=(-0.01, 1.02), loc=3,
           ncol=2)
    axarr[i].tick_params(labelsize=TICK_SIZE)
    ax = axarr[i].twinx()
    ax.set_ylabel(TITLES[i], rotation=270, labelpad=20, size=TICK_SIZE)
    ax.set_yticks([])
    
f.subplots_adjust(hspace=0.1)
# Hide x labels and tick labels for all but bottom plot.
plt.xticks((0.2, 0.7, 1.2), ('Precision', 'Recall', 'F1'))
plt.tight_layout()


# ## F1 Score for Best respondent and Median Respondent

# In[90]:


def print_individual_report(target_name, ir):
    if target_name == 'MIT':
        target_number = 2

    elif target_name == 'Insulator':
        target_number = 0

    elif target_name == 'Metal':
        target_number = 1

    tp = ir[(ir['Label'] == target_number) & (ir[target_name] == 1)]
    tn = ir[(ir['Label'] != target_number) & (ir[target_name] != 1)]
    fp = ir[(ir['Label'] != target_number) & (ir[target_name] == 1)]
    fn = ir[(ir['Label'] == target_number) & (ir[target_name] != 1)]

    n_tp = tp.shape[0]
    n_tn = tn.shape[0]
    n_fp = fp.shape[0]
    n_fn = fn.shape[0]
    
    title_list = ["","Actual_"+target_name, "Actual_non_"+target_name]
    predict_list = ["Predicted_"+target_name, n_tp, n_fp]
    predict_non_list = ["Predicted_non_"+target_name, n_fn, n_tn]
    
    classification_matrix = [title_list,
                             predict_list,
                             predict_non_list,]
    
    print_table(classification_matrix)
    print()
    
    if (n_tp + n_fp != 0) & (n_tn + n_fn != 0):
        p0 = round(n_tp / (n_tp + n_fp), 2)
        r0 = round(n_tp / (n_tp + n_fn), 2)
        f0 = round(2 * n_tp / (2 * n_tp + n_fp + n_fn), 2)
        s0 = n_tp + n_fn

        p1 = round(n_tn / (n_tn + n_fn), 2)
        r1 = round(n_tn / (n_tn + n_fp), 2)
        f1 = round(2 * n_tn / (2 * n_tn + n_fn + n_fp), 2)
        s1 = n_tn + n_fp

        w0 = s0 / (s0+s1)
        w1 = s1 / (s0+s1)

        a1 = round(p0*w0 + p1*w1, 2)
        a2 = round(r0*w0 + r1*w1, 2)
        a3 = round(f0*w0 + f1*w1, 2)
        a4 = s0 + s1
        
        performance_category = ["","precison","recall","f1", "support"]
        if target_name == "MIT":
            non_target_list = ["Insulator+Metal", p1, r1, f1, s1]
        elif target_name == "Insulator":
            non_target_list = ["MIT+Metal", p1, r1, f1, s1]
        else:
            non_target_list = ["MIT+Insulator", p1, r1, f1, s1]
    
        target_list = [target_name, p0, r0, f0, s0]
        avg_total_list = ["avg/tol", a1, a2, a3, a4]
    
        classification_report = [performance_category,
                                non_target_list,
                                target_list,
                                avg_total_list,]
    
        print_table(classification_report)
    
    #print('\t', 'precision', '\t', 'recall', '\t', 'f1', '\t', 'support')
    #print('\t', p1, '\t', r1, '\t', f1, '\t', s1)
    #print('\t', p0, '\t', r0, '\t', f0, '\t', s0)
    #print('\t', a1, '\t', a2, '\t', a3, '\t', a4)


# In[91]:


best = ir[ir['Respondent'] == 52]
print_individual_report('MIT', best)


# In[92]:


best = ir[ir['Respondent'] == 12]
print_individual_report('MIT', best)


# In[93]:


print_individual_report('Metal', best)


# In[94]:


print_individual_report('Insulator', best)


# In[95]:


median = ir[ir['Accuracy']==61]


# In[96]:


def give_f1(target_name, ir):
    if target_name == 'MIT':
        target_number = 2

    elif target_name == 'Insulator':
        target_number = 0

    elif target_name == 'Metal':
        target_number = 1
        
    tp = ir[(ir['Label'] == target_number) & (ir[target_name] == 1)]
    tn = ir[(ir['Label'] != target_number) & (ir[target_name] != 1)]
    fp = ir[(ir['Label'] != target_number) & (ir[target_name] == 1)]
    fn = ir[(ir['Label'] == target_number) & (ir[target_name] != 1)]

    n_tp = tp.shape[0]
    n_tn = tn.shape[0]
    n_fp = fp.shape[0]
    n_fn = fn.shape[0]

    if (n_tp + n_fp != 0) & (n_tn + n_fn != 0):
        p0 = round(n_tp / (n_tp + n_fp), 2)
        r0 = round(n_tp / (n_tp + n_fn), 2)
        f0 = round(2 * n_tp / (2 * n_tp + n_fp + n_fn), 2)
        s0 = n_tp + n_fn

        p1 = round(n_tn / (n_tn + n_fn), 2)
        r1 = round(n_tn / (n_tn + n_fp), 2)
        f1 = round(2 * n_tn / (2 * n_tn + n_fn + n_fp), 2)
        s1 = n_tn + n_fp

        w0 = s0 / (s0+s1)
        w1 = s1 / (s0+s1)

        a1 = round(p0*w0 + p1*w1, 2)
        a2 = round(r0*w0 + r1*w1, 2)
        a3 = round(f0*w0 + f1*w1, 2)
        a4 = s0 + s1
        
        return a3
    else:
        return 0


# In[97]:


def give_f1_list(target_name):
    f1_list = []
    
    for i in np.arange(1, tn + 1):
        df = ir[ir['Respondent'] == i]
        f1 = give_f1(target_name, df)
        f1_list.append(f1)
        
    return f1_list


# In[98]:


MIT_f1_list = give_f1_list('MIT')
Metal_f1_list = give_f1_list('Metal')
Insulator_f1_list = give_f1_list('Insulator')
print(MIT_f1_list)
plt.boxplot(MIT_f1_list, vert=False)
plt.title("MIT vs. Rest")
MIT_f1_bool = [x == 0.89 for x in MIT_f1_list]
print("Number of 0.89: "+ str(sum(MIT_f1_bool)))
print("MIT vs. Rest: Highest f1 score: {}".format(round(max(MIT_f1_list),2)))
print("MIT vs. Rest: Average f1 score: {n1} ± {n2}".format(n1 = round(np.mean(MIT_f1_list), 2), n2 = round(np.std(MIT_f1_list),2)))
print("MIT vs. Rest: Median f1 score: {n1} ± {n2}".format(n1=round(np.median(MIT_f1_list),2), 
                                                          n2=round(IQR(MIT_f1_list),2)))


# In[99]:


print(Metal_f1_list)
plt.boxplot(Metal_f1_list, vert=False)
plt.title("Metal vs. Rest")
Metal_f1_bool = [x == 0.89 for x in Metal_f1_list]
print("Number of 0.89: "+ str(sum(Metal_f1_bool)))
print("Metal vs. Rest: Highest f1 score: {}".format(round(max(Metal_f1_list),2)))
print("Metal vs. Rest: Average f1 score: {n1} ± {n2}".format(n1 = round(np.mean(Metal_f1_list), 2), n2 = round(np.std(Metal_f1_list),2)))
print("Metal vs. Rest: Median f1 score: {n1} ± {n2}".format(n1=round(np.median(Metal_f1_list),2), 
                                                          n2=round(IQR(Metal_f1_list),2)))


# In[100]:


print(Insulator_f1_list)
plt.boxplot(Insulator_f1_list, vert=False)
plt.title("Insulator vs. Rest")
Insulator_f1_bool = [x == 0.89 for x in Insulator_f1_list]
print("Number of 0.89: " + str(sum(Insulator_f1_bool)))
print("Insulator vs. Rest: Highest f1 score: {}".format(
    round(max(Insulator_f1_list), 2)))
print("Insulator vs. Rest: Average f1 score: {n1} ± {n2}".format(n1=round(
    np.mean(Insulator_f1_list), 2), n2=round(np.std(Insulator_f1_list), 2)))
print("Insulator vs. Rest: Median f1 score: {n1} ± {n2}".format(n1=round(np.median(Insulator_f1_list), 2),
                                                                n2=round(IQR(Insulator_f1_list), 2)))


# In[101]:


def plot_f1_distribution(f1_list, target_name):
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.distplot(
        f1_list, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], kde=False)
    plt.xlabel('f1 score', fontsize=24)
    plt.ylabel('Number of people', fontsize=24)
    plt.title(target_name + " vs. Rest: f1 score distribution", fontsize=32)
    if target_name == 'Metal':
        plt.yticks(np.arange(0, 20, 2))


# In[102]:


plot_f1_distribution(MIT_f1_list, 'MIT')


# In[103]:


plot_f1_distribution(Insulator_f1_list, 'Insulator')


# In[104]:


plot_f1_distribution(Metal_f1_list, 'Metal')

