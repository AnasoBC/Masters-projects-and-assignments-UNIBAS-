#!/usr/bin/env python
# coding: utf-8

# In[51]:


# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for making nice plots
import seaborn as sns
import matplotlib.patches as mpatches
import time
import plotly.express as px

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import collections

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,roc_curve, RocCurveDisplay

# We imported both files but for the mean-time, we only use the originator data, deepseek suggested the low_memory=false addition as the dataset did not load properly
service = pd.read_csv("C:\\Users\\DELL\\Downloads\\freddie maculmo\\servicing2017.csv",low_memory=False)
originator = pd.read_csv(r"C:\Users\DELL\Downloads\freddie maculmo\\origination2017.csv",low_memory=False)

# PERFORMANCE LEVEL DATA BASE WAS TOO LARGE AI WAS USED AND RECOMMENDED THE LOW_MEMORY=FALSE COMMAND


# In[52]:


# Here we preview dataset
#print(service.head())
#print(service.tail())
print(originator.head())


# There are 31 columns present in this dataset, hence it would be alot harder to go through them individually, especially since many of them are not neccessary and have missing values (hence need to be dropped) insteads below is a summary of the 5 most relevant columns:
# 
#     * FICO: Fair Isaac Corporation. It's the name of the company that created the FICO credit score. In this dataset, it is the credit score for each loan recipient.
#     
#     * dt_first_pi
#     * orig_loan_term
#     *

# In[89]:


originator.describe()


# In[ ]:





# Count: How many observations do we observe for the given variables? 1'000'000 for each
# 
# Mean: What is the mean value of the variable? For binary variables (0,1) this is also corresponds to the share of "1". e.g. 8.74% of the transactions are fraud tranactions.
# 
# Std.: The standard deviation of the variable:
# 
# Min: The lowest observed value for the variable.
# 
# 25% , 50% , 75%: The percentiles of the distribution of the variable. 50% = median. --> 50% of observations are above this value, 50% below.
# 
# max: The highest observed value for th variable.

# In[54]:


originator.columns


# In[55]:


# the following codes were adapted from the codes from class:
print(f"There are {originator.isnull().sum().max()} maximum missing values")

originator.isnull().sum()


# Unfortunately there are a lot of missing values in our dataset, the best strategy would be to remove the columns with the most missing values.
# 
# We also will have to remove values like 9, 99, 999, 999 and 9999, in the dataset as they signify missing values used in different columns.

# In[56]:


# dropping removing missing values from columns with the most missing values, I selected them by name
originator = originator.drop(columns=['flag_fthb', 'flag_sc', 'id_loan_preharp', 'ind_harp', 'ind_afdl', 'cd_msa'])


# In[57]:


# getting the number of columns
num_columns = len(originator.columns)
num_columns # could also use originator.info


# In[58]:


# fico is an especially important column, however the presence of 9999 skews the values
#print(originator.describe()['fico'])
originator.describe()


# Cleaning the data set, we can see that the dataset has the existence of spaceholder numbers such as 999 and 9999 and 99 used in some columns to stand in for unavailable, lets clean these in the relevant columns

# In[ ]:





# In[59]:


# Define filter conditions
filter_conditions = (
    (originator['fico'] != 9999) & 
    (originator['dti'] != 999) & 
    (originator['cltv'] != 999) & 
    (originator['ltv'] != 999) &
    (originator['cd_ppty_val_type'] != 9)
)

# Apply filters
fico_filtered = originator[filter_conditions].copy()  # Using .copy() to avoid SettingWithCopyWarning

# 1. FICO Score Analysis
print("\nFICO Score Distribution:")
print(fico_filtered['fico'].describe())

# 2. Full Dataset Statistics
print("\nFiltered Dataset Statistics:")
print(fico_filtered.describe())

# 3. Count of removed records
original_count = len(originator)
filtered_count = len(fico_filtered)
print(f"\nRecords removed: {original_count - filtered_count} ({((original_count - filtered_count)/original_count)*100:.1f}%)")

originator = fico_filtered


# In addition to stock values of 9999 in fico score, we also removed 999 in dti, cltv and ltv, and also the 9 in cd_ppty_val_type, which all represent NA in their representive columns, by so doing we now have a clearer representation of the maximum figures of these variables.

# In[60]:


print('Non_Fintech', round(originator['fintech'].value_counts()[0]/len(originator) * 100,2), '% of the dataset')
print('Fintech', round(originator['fintech'].value_counts()[1]/len(originator) * 100,2), '% of the dataset')


# In[95]:


import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score

originator.describe()


# In[61]:


print(originator['fintech'].count()) # this is the number of individual loans from the originator data set
# we can also use # service['id_loan'].nunique
print(service['id_loan'].nunique())


# In[62]:


originator['fintech'].isnull().sum()


# In[63]:


# Filtering Fintech and Non-Fintech lenders
fin_tech = originator[originator['fintech'] == "Fintech"]
non_fintech = originator[originator['fintech'] == "Non-fintech"]

# Share of loans by Fintech lenders
fintech_count = len(fin_tech)
non_fintech_count = len(non_fintech)
total_count = len(originator)

# Average credit scores
av_fintech = fin_tech['fico'].mean()
av_non_fin = non_fintech['fico'].mean()
print(f"Average FICO - Fintech: {av_fintech:.2f}, Non-Fintech: {av_non_fin:.2f}")

# Average interest rates
av_int_fint = fin_tech['int_rt'].mean()
av_int_nfnt = non_fintech['int_rt'].mean()

print(f"Average Interest Rate - Fintech: {av_int_fint:.2f}%, Non-Fintech: {av_int_nfnt:.2f}%")

# Credit score distribution plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=originator, x='fico', hue='fintech', fill=True, alpha=0.5)
plt.title('Credit Score Distribution by Lender Type')
plt.xlabel('Credit Score (FICO)')
plt.ylabel('Density')
plt.show()

# Merge origination and servicing data
#merged_data = pd.merge(originator, service, on='id_loan')

# Pie chart
plt.figure(figsize=(8, 6))
plt.pie([fintech_count, non_fintech_count], 
        labels=['Fintech', 'Non-Fintech'],
        colors=['green', 'blue'],
        autopct='%1.1f%%')
plt.title('Share of Loans by Lender Type')
plt.show()


# Calculate delinquency rates
#delinquency_rate = merged_data.groupby('fintech')['delq_sts'].mean() * 100
#print("\nDelinquency Rates:")
#print(delinquency_rate)



# In[ ]:





# In[76]:


# LETS HAVE A DEEP DIVE INTO LOAN DISTRIBUTION BY STATE(fintech vs non-fintech):

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create subplots
fig = make_subplots(rows=1, cols=2, 
                   column_widths=[0.6, 0.4],
                   specs=[[{"type": "choropleth"}, {"type": "bar"}]])

# Add map
fig.add_trace(go.Choropleth(
    locations=originator['st'],
    z=originator['fintech'],
    locationmode='USA-states',
    colorscale='Viridis',
    colorbar=dict(title='Loan Count')
), row=1, col=1)

# Add bar chart sorted by loan count
state_sorted = originator.sort_values('fintech', ascending=False)
fig.add_trace(go.Bar(
    x=state_sorted['st'],
    y=state_sorted['fintech'],
    marker_color='green'
), row=1, col=2)

# Update layout
fig.update_layout(
    title_text='Loan Distribution by State',
    geo_scope='usa',
    height=600,
    showlegend=False
)

fig.show()


# In[64]:


# top service providers among fintech and non-fintech


# Top Service Providers Analysis

# 1. Count of loans by servicer for each lender type
service_provider_counts = originator.groupby(['fintech', 'servicer_name']).size().reset_index(name='loan_count')

# 2. Top 5 service providers for Fintech lenders
top_fintech_servicers = (
    service_provider_counts[service_provider_counts['fintech'] == "Fintech"]
    .sort_values('loan_count', ascending=False)
    .head(5)
)

# 3. Top 5 service providers for Non-Fintech lenders
top_non_fintech_servicers = (
    service_provider_counts[service_provider_counts['fintech'] == "Non-fintech"]
    .sort_values('loan_count', ascending=False)
    .head(5)
)

# 4. Visualization
plt.figure(figsize=(14, 6))

# Fintech servicers plot
plt.subplot(1, 2, 1)
sns.barplot(data=top_fintech_servicers, x='loan_count', y='servicer_name', palette='Blues_d')
plt.title('Top 5 Service Providers - Fintech Lenders')
plt.xlabel('Number of Loans Serviced')
plt.ylabel('Service Provider')

# Non-Fintech servicers plot
plt.subplot(1, 2, 2)
sns.barplot(data=top_non_fintech_servicers, x='loan_count', y='servicer_name', palette='Reds_d')
plt.title('Top 5 Service Providers - Non-Fintech Lenders')
plt.xlabel('Number of Loans Serviced')
plt.ylabel('')

plt.tight_layout()
plt.show()


# In[ ]:





# In[65]:


# top loan tenure among fintech and non-fintech

# Calculate national average fintech share
national_fintech_share = (originator['fintech'] == 'Fintech').mean() * 100

plt.axhline(y=national_fintech_share, color='r', linestyle='--', label='National Average')

# Calculate loan counts by state and lender type
state_distribution = originator.groupby(['st', 'fintech']).size().unstack().fillna(0)
state_distribution['Total'] = state_distribution.sum(axis=1)
state_distribution['Fintech_Share'] = (state_distribution['Fintech'] / state_distribution['Total']) * 100
state_distribution = state_distribution.sort_values('Total', ascending=False)

# Calculate national average fintech share
national_fintech_share = (originator['fintech'] == 'Fintech').mean() * 100

# Percentage share
plt.subplot(2, 1, 2)
state_distribution['Fintech_Share'].head(15).plot(kind='bar', color='#2ca02c')
plt.title('Fintech Market Share in Top 15 States')
plt.ylabel('Fintech Share (%)')
plt.xlabel('State')
plt.axhline(y=national_fintech_share, color='r', linestyle='--', label='National Average')
plt.legend()

plt.tight_layout()
plt.show()

# AI WAS USED TO COMPLETE THIS CODE


# In[ ]:





# In[66]:


top_fintech_states = state_distribution.sort_values('Fintech_Share', ascending=False).head(5)
print("\nStates with Highest Fintech Market Share:")
print(top_fintech_states[['Fintech', 'Non-fintech', 'Fintech_Share']])

bottom_fintech_states = state_distribution[state_distribution['Total'] > 1000].sort_values('Fintech_Share').head(5)
print("\nStates with Lowest Fintech Market Share:")
print(bottom_fintech_states[['Fintech', 'Non-fintech', 'Fintech_Share']])


# In[24]:


originator.describe()


# In[17]:


# Adjust figure size
plt.figure(figsize=(4, 3))  # Smaller size for better visibility

# Use sns.relplot with scatter for better visualization
sns.boxplot(data=originator, y='int_rt', x='fintech')

# Set labels and title
plt.title('Interest Rate Distribution by Lender Type')
plt.xlabel('Interest Rate')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[18]:


# Credit score distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=originator, x='orig_upb', hue='fintech', fill=True, alpha=0.4)
plt.title('Loan Amount Distribution by Lender Type')
plt.xlabel('Lender Type')
plt.ylabel('Amount')
plt.show()


# In[47]:


print(service.describe())


# In[48]:


service.isnull().sum()


# In[67]:


# cleaning the service dataset and dropping elected columns
service = service.iloc[:, [0,1,2,3,4,5,10,11]]

service.isnull().sum() # keep eltv since there are over 843626 valid items, could interpolate but the columns isnt extremely necessary to our essay


# In[68]:


service['svcg_cycle'].tolist()


# In[69]:


# merging the datasets

merged_df = pd.merge(service, originator, on='id_loan', how='left')

merged_df


# In[70]:


merged_df = merged_df.dropna()

merged_df.isnull().sum().head(40)


# In[74]:


# create a delinquency flag

merged_df['delq_sts'] = pd.to_numeric(merged_df['delq_sts'], errors='coerce')  # ensure numeric
merged_df['delinquent'] = (merged_df['delq_sts'] != 0).astype(int)

merged_df['delinquent']


# In[75]:


print(merged_df['delinquent'])


# In[76]:


uniqlo = merged_df[merged_df['delq_sts'] != 0].nunique().copy()

uniqlo


# In[77]:


count = (merged_df['delq_sts'] != 0).sum()
print(f"This number appears {count} times in 'delq_sts',") # this is 


# In[ ]:





# In[59]:


#uniqlo = merged_df2.drop(merged_df2[merged_df2['delq_sts'] == 0].index).nunique()


#i created a new dataframe (merge_df2) that drops all non-delinquent loans, I will need the the main merged data set later 
#uniqlo = (merged_df.nunique())
#uniqlo = merged_df['id_loan'].nunique()
#print(f"{uniqlo[0]} loans went delinquent at some point:")


# In[78]:


merged_df['id_loan'].nunique()


# In[ ]:





# We create a copy of our main dataframe (merged_df2) to not make any major changes to is a copy of our dataframe, we do this to ascertain the actual number of loans that went delinquent, initially using

# In[79]:


# what percentage of loans become delinquent

percent_delinq_loan = uniqlo[0]/merged_df['id_loan'].nunique() * 100

print(f"{percent_delinq_loan:.1f}% of loans in the dataset went delinquent at some point")


# In[80]:


# Create filtered DataFrame (keeping all columns)
delinquent_loans_df = merged_df[merged_df['delq_sts'] != 0].copy()
# since we also created a flagged column we could also use:
#delinquent_loans_df = merged_df[merged_df['delinquent'] != 0]

# Get count of unique delinquent loans
delinquent_loan_count = delinquent_loans_df['id_loan'].nunique()
print(f"{delinquent_loan_count} loans went delinquent at some point:")

# Get list of unique delinquent loan IDs (while preserving all data)
unique_delinquent_ids = delinquent_loans_df['id_loan'].unique().tolist()

# Now you can analyze while keeping all associated data:
# Example 1: Count fintech vs non-fintech delinquencies
fintech_counts = delinquent_loans_df.groupby('fintech')['id_loan'].nunique()
print("\nDelinquencies by fintech status:")
print(fintech_counts)

# Example 2: Get full records for first 5 delinquent loans
print("\nSample delinquent loan records:")
print(delinquent_loans_df.head())


# In[60]:


merged_df


# In[ ]:





# In[81]:


# fintech vs non-fintech

colors = ["#6aa84f", "#DF0101"]
#old color: #8fce00

sns.countplot(data=merged_df,x='fintech',palette=colors)
plt.title('Class Distributions \n (0: Non Fintech || 1: Fintech)', fontsize=14)


# In[447]:





# In[36]:


plt.figure(figsize=(18,8))
corr = merged_df.corr()
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
heatmap.set_title("Imbalanced Correlation Matrix", fontsize=14)


# In[37]:


#merged_df = merged_df.dropna()

merged_df.isnull().sum()

# too many missing values, wont run if there are missing values


# In[123]:


# Log-transform continuous variables for modeling
merged_df['log_dti'] = np.log1p(merged_df['dti'])
merged_df['log_int_rt'] = np.log1p(merged_df['int_rt'])


# In[125]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  ('fico')
X = merged_df[['fico']]  
y = merged_df['delinquent']

# Train-test split (MISSING IN YOUR ORIGINAL CODE)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)

# Make predictions
y1_pred = model1.predict(X_test)
y1_pred_prob = model1.predict_proba(X_test)[:, 1]  # Fixed slicing (was [::,1])

# ROC Curve
fpr_1, tpr_1, threshold_1 = metrics.roc_curve(y_test, y1_pred_prob)
auc_score = roc_auc_score(y_test, y1_pred_prob)  # Calculate AUC

# Confusion Matrix
forest_cf = confusion_matrix(y_test, y1_pred)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(forest_cf, ax=ax1, annot=True, cmap='copper', fmt='d')
ax1.set_title("Random Forest Confusion Matrix", fontsize=14)
ax1.set_xlabel("Predicted Outcome", fontsize=12)
ax1.set_ylabel("Actual Outcome", fontsize=12)
ax1.xaxis.set_ticklabels(['Non-Delinquent', 'Delinquent'])
ax1.yaxis.set_ticklabels(['Non-Delinquent', 'Delinquent'])

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y1_pred))
print(f"AUC Score: {auc_score:.4f}")

# ROC Curve Plot (simplified)
plt.figure(figsize=(10, 6))
plt.plot(fpr_1, tpr_1, color="blue", label=f'Random Forest (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[126]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  ('dti')
X = merged_df[['dti']]  
y = merged_df['delinquent']

# Train-test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)

# Make predictions
y1_pred = model1.predict(X_test)
y1_pred_prob = model1.predict_proba(X_test)[:, 1]  # Fixed slicing (was [::,1])

# ROC Curve
fpr_1, tpr_1, threshold_1 = metrics.roc_curve(y_test, y1_pred_prob)
auc_score = roc_auc_score(y_test, y1_pred_prob)  # Calculate AUC

# Confusion Matrix
forest_cf = confusion_matrix(y_test, y1_pred)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(forest_cf, ax=ax1, annot=True, cmap='copper', fmt='d')
ax1.set_title("Random Forest Confusion Matrix", fontsize=14)
ax1.set_xlabel("Predicted Outcome", fontsize=12)
ax1.set_ylabel("Actual Outcome", fontsize=12)
ax1.xaxis.set_ticklabels(['Non-Delinquent', 'Delinquent'])
ax1.yaxis.set_ticklabels(['Non-Delinquent', 'Delinquent'])

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y1_pred))
print(f"AUC Score: {auc_score:.4f}")

# ROC Curve Plot (simplified)
plt.figure(figsize=(10, 6))
plt.plot(fpr_1, tpr_1, color="red", label=f'Random Forest (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[128]:


from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  ('Interest Rate')
X = merged_df[['log_int_rt']]  
y = merged_df['delinquent']

# Train-test split (MISSING IN YOUR ORIGINAL CODE)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest
model1 = RandomForestClassifier(random_state=42)
model1.fit(X_train, y_train)

# Make predictions
y1_pred = model1.predict(X_test)
y1_pred_prob = model1.predict_proba(X_test)[:, 1]  # Fixed slicing (was [::,1])

# ROC Curve
fpr_1, tpr_1, threshold_1 = metrics.roc_curve(y_test, y1_pred_prob)
auc_score = roc_auc_score(y_test, y1_pred_prob)  # Calculate AUC

# Confusion Matrix
forest_cf = confusion_matrix(y_test, y1_pred)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(forest_cf, ax=ax1, annot=True, cmap='copper', fmt='d')
ax1.set_title("Random Forest Confusion Matrix", fontsize=14)
ax1.set_xlabel("Predicted Outcome", fontsize=12)
ax1.set_ylabel("Actual Outcome", fontsize=12)
ax1.xaxis.set_ticklabels(['Non-Delinquent', 'Delinquent'])
ax1.yaxis.set_ticklabels(['Non-Delinquent', 'Delinquent'])

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y1_pred))
print(f"AUC Score: {auc_score:.4f}")

# ROC Curve Plot (simplified)
plt.figure(figsize=(10, 6))
plt.plot(fpr_1, tpr_1, color="green", label=f'Random Forest (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[46]:


f, axes = plt.subplots(ncols=2, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="fico", y="fintech", data=merged_df, palette=None, ax=axes[0])
axes[0].set_title('Delinquencies from Fintech vs. Non-fintech: Positive Correlation', fontsize=14)


sns.boxplot(x="delinquent", y="log_dti", data=merged_df, palette=None, ax=axes[1])
axes[1].set_title('correlation between delinquency and log dti', fontsize=14)

plt.show()


# In[138]:


# fintech vs non-fintech interest rate random forest.

# Assuming merged_df contains:
# 'log_int_rt' - log-transformed interest rate
# 'delinquent' - target variable (1 = delinquent, 0 = non-delinquent)
# 'lender_type' - categorical column indicating FinTech vs Non-Fintech

# Create lender subsets
fintech_df = merged_df[merged_df['fintech'] == 'finTech']
non_fintech_df = merged_df[merged_df['fintech'] == 'non-fintech']

def run_rf_analysis(df, lender_name):
    """Run Random Forest analysis for a specific lender type"""
    X = merged_df[['log_int_rt']]
    y = merged_df['delinquent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    auc_score = roc_auc_score(y_test, y_pred_prob)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='copper', 
                xticklabels=['Non-Delinquent', 'Delinquent'],
                yticklabels=['Non-Delinquent', 'Delinquent'])
    plt.title(f"{lender_name} Confusion Matrix\nAUC = {auc_score:.4f}", fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.show()
    
    # Generate ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='green', label=f'{lender_name} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{lender_name} ROC Curve')
    plt.legend()
    plt.show()
    
    # Print classification report
    print(f"\n{lender_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {auc_score:.4f}\n")
    
    return auc_score

# Run analysis for both lender types
print("="*50)
fintech_auc = run_rf_analysis(fintech_df, "FinTech Interest Rate Model")
print("="*50)
non_fintech_auc = run_rf_analysis(non_fintech_df, "Non-Fintech Interest Rate Model")
print("="*50)

# Compare AUC scores
print("\n" + "="*30 + " SUMMARY " + "="*30)
print(f"FinTech AUC: {fintech_auc:.4f}")
print(f"Non-fintech AUC: {non_fintech_auc:.4f}")
print(f"Performance Difference: {fintech_auc - non_fintech_auc:.4f}")
print("="*70)


# In[129]:


# Setting up the figure with a more appropriate size
plt.figure(figsize=(18, 6))

# Create subplots with 3 columns
fig, axes = plt.subplots(ncols=3, figsize=(18, 6))

# Custom palette for better visual distinction
palette = {0: "lightblue", 1: "salmon"}

# FICO Score vs Delinquency
sns.boxplot(x="delinquent", y="fico", data=merged_df, palette=palette, ax=axes[0], 
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[0].set_title('FICO Score vs Delinquency Status', fontsize=14, pad=20)
axes[0].set_xlabel('Delinquent Status (0=No, 1=Yes)', fontsize=12)
axes[0].set_ylabel('FICO Score', fontsize=12)
axes[0].set_xticklabels(['Non-Delinquent', 'Delinquent'])

# Interest Rate vs Delinquency
sns.boxplot(x="delinquent", y="int_rt", data=merged_df, palette=palette, ax=axes[1],
           showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[1].set_title('Interest Rate vs Delinquency Status', fontsize=14, pad=20)
axes[1].set_xlabel('Delinquent Status (0=No, 1=Yes)', fontsize=12)
axes[1].set_ylabel('Interest Rate (%)', fontsize=12)
axes[1].set_xticklabels(['Non-Delinquent', 'Delinquent'])

# Log DTI vs Delinquency
sns.boxplot(x="delinquent", y="log_dti", data=merged_df, palette=palette, ax=axes[2],
           showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
axes[2].set_title('Log(DTI) vs Delinquency Status', fontsize=14, pad=20)
axes[2].set_xlabel('Delinquent Status (0=No, 1=Yes)', fontsize=12)
axes[2].set_ylabel('Log(Debt-to-Income Ratio)', fontsize=12)
axes[2].set_xticklabels(['Non-Delinquent', 'Delinquent'])

# Adjust layout to prevent label overlap
plt.tight_layout()

# Add an overall title
plt.suptitle('Relationship Between Key Features and Loan Delinquency', y=1.05, fontsize=16, fontweight='bold')

plt.show()


# In[ ]:


# resampling 

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model = RandomForestClassifier(class_weight='balanced').fit(X_resampled, y_resampled)


# In[ ]:





# In[132]:


# Calculate delinquency rates
delinquency_rates = merged_df.groupby('fintech')['delinquent'].mean() * 100
print(f"\nOverall delinquency rate: {merged_df['delinquent'].mean()*100:.2f}%")
print(f"Fintech delinquency: {delinquency_rates['Fintech']:.2f}%")
print(f"Non-Fintech delinquency: {delinquency_rates['Non-fintech']:.2f}%")

# PREDICTIVE MODELING ==========
from sklearn.metrics import roc_curve, auc

# Model 1: FICO only
X_fico = merged_df[['fico']]
y = merged_df['delinquent']
X_train, X_test, y_train, y_test = train_test_split(X_fico, y, test_size=0.3, random_state=42)

rf_fico = RandomForestClassifier(n_estimators=100)
rf_fico.fit(X_train, y_train)
y_pred_proba_fico = rf_fico.predict_proba(X_test)[:,1]
fpr_fico, tpr_fico, _ = roc_curve(y_test, y_pred_proba_fico)
auc_fico = auc(fpr_fico, tpr_fico)

# Model 2: DTI only
X_dti = merged_df[['dti']]
X_train, X_test, y_train, y_test = train_test_split(X_dti, y, test_size=0.3, random_state=42)

rf_dti = RandomForestClassifier(n_estimators=100)
rf_dti.fit(X_train, y_train)
y_pred_proba_dti = rf_dti.predict_proba(X_test)[:,1]
fpr_dti, tpr_dti, _ = roc_curve(y_test, y_pred_proba_dti)
auc_dti = auc(fpr_dti, tpr_dti)

# INTEREST RATE MODELS BY LENDER TYPE ==========
# Fintech subset
fintech_df = merged_df[merged_df['fintech'] == 'Fintech']
X_int_fintech = fintech_df[['int_rt']]
y_fintech = fintech_df['delinquent']

X_train, X_test, y_train, y_test = train_test_split(X_int_fintech, y_fintech, test_size=0.3, random_state=42)
rf_fintech = RandomForestClassifier(n_estimators=100)
rf_fintech.fit(X_train, y_train)
y_pred_proba_fintech = rf_fintech.predict_proba(X_test)[:,1]
fpr_fintech, tpr_fintech, _ = roc_curve(y_test, y_pred_proba_fintech)
auc_fintech = auc(fpr_fintech, tpr_fintech)

# Non-fintech subset
non_fintech_df = merged_df[merged_df['fintech'] == 'Non-fintech']
X_int_non = non_fintech_df[['int_rt']]
y_non = non_fintech_df['delinquent']

X_train, X_test, y_train, y_test = train_test_split(X_int_non, y_non, test_size=0.3, random_state=42)
rf_non = RandomForestClassifier(n_estimators=100)
rf_non.fit(X_train, y_train)
y_pred_proba_non = rf_non.predict_proba(X_test)[:,1]
fpr_non, tpr_non, _ = roc_curve(y_test, y_pred_proba_non)
auc_non = auc(fpr_non, tpr_non)


plt.figure(figsize=(10, 8))
# FICO vs DTI
plt.plot(fpr_fico, tpr_fico, color='blue', lw=2, 
         label=f'FICO (AUC = {auc_fico:.2f})')
plt.plot(fpr_dti, tpr_dti, color='green', lw=2, 
         label=f'DTI (AUC = {auc_dti:.2f})')

# Fintech vs Non-fintech
plt.plot(fpr_fintech, tpr_fintech, color='red', linestyle='--', 
         label=f'Fintech Interest (AUC = {auc_fintech:.2f})')
plt.plot(fpr_non, tpr_non, color='purple', linestyle='--', 
         label=f'Non-Fintech Interest (AUC = {auc_non:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()

# ========== 5. CONCLUSION ANALYSIS ==========
print("\n" + "="*50)
print("Key Conclusions:")
print("="*50)
print(f"1. FICO is a better predictor (AUC={auc_fico:.2f}) than DTI (AUC={auc_dti:.2f})")
print(f"2. Interest rate predicts better for Fintech (AUC={auc_fintech:.2f}) than Non-Fintech (AUC={auc_non:.2f})")
print(f"3. Fintech delinquency rate: {delinquency_rates['Fintech']:.2f}% vs Non-Fintech: {delinquency_rates['Non-fintech']:.2f}%")

if auc_fintech > auc_non:
    print("4. Fintech lenders show better loan performance prediction")
    print("   suggesting more risk-based pricing in interest rates")
else:
    print("4. Non-Fintech lenders show better loan performance prediction")
    
if delinquency_rates['Fintech'] < delinquency_rates['Non-fintech']:
    print("5. Fintech lenders have lower delinquency rates")
    print("   indicating superior underwriting standards")
else:
    print("5. Non-Fintech lenders have lower delinquency rates")
    
# AI WAS USED TO COMPLETE THIS CODE


# In[131]:


from sklearn.metrics import classification_report


# Model 1: FICO only
rf_fico = RandomForestClassifier(n_estimators=100)
rf_fico.fit(X_train, y_train)
y_pred_fico = rf_fico.predict(X_test)  # Get class predictions (not probabilities)
print("\nFICO Model Classification Report:")
print(classification_report(y_test, y_pred_fico))

# Model 2: DTI only
rf_dti = RandomForestClassifier(n_estimators=100)
rf_dti.fit(X_train, y_train)
y_pred_dti = rf_dti.predict(X_test)
print("\nDTI Model Classification Report:")
print(classification_report(y_test, y_pred_dti))

# Fintech interest rate model
rf_fintech = RandomForestClassifier(n_estimators=100)
rf_fintech.fit(X_train, y_train)
y_pred_fintech = rf_fintech.predict(X_test)
print("\nFintech Interest Rate Model Classification Report:")
print(classification_report(y_test, y_pred_fintech))

# Non-fintech interest rate model
rf_non = RandomForestClassifier(n_estimators=100)
rf_non.fit(X_train, y_train)
y_pred_non = rf_non.predict(X_test)
print("\nNon-Fintech Interest Rate Model Classification Report:")
print(classification_report(y_test, y_pred_non))


# In[ ]:





# In[ ]:




