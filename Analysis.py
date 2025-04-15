#!/usr/bin/env python
# coding: utf-8

# In[54]:

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

dataurl= r"C:\Users\urvi9\marketing_campaign.xlsx"
descriptor= r"C:\Users\urvi9\Variable Descriptor.xlsx"

des = pd.read_excel(descriptor)
df = pd.read_excel(dataurl)
df

# In[55]:

des

# In[56]:


print("Duplicates found:",df.duplicated().sum())
df.dropna(inplace=True)


# In[57]:

df = df.drop(columns = "Z_CostContact")
df = df.drop(columns = "Z_Revenue")

#Combining columns 
df['MntTotalExpense'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df.drop(columns=['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], inplace=True)

df['TotalPurchases'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']
df.drop(columns=['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'], inplace=True)

df['AccepetedCmp1-5'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4']+ df['AcceptedCmp5']
df.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'], inplace=True)

df


# In[58]:

X = df[['NumWebVisitsMonth','MntTotalExpense','TotalPurchases','AccepetedCmp1-5']]
y = df['Response'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
accuracy_lr1 = accuracy_score(y_test, y_pred)
precision_lr1 = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_lr1}")
print(f"Precision: {precision_lr1}")
print(f"Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)


# In[59]:


svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

accuracy_svm1 = accuracy_score(y_test, y_pred_svm)
precision_svm1 = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM Model Performance:")
print(f"Accuracy: {accuracy_svm1}")
print(f"Precision: {precision_svm1}")
print(f"Recall: {recall_svm}")
print("Confusion Matrix:")
print(conf_matrix_svm)


# In[60]:


X2 = df[['Income']]
y = df['Response'] 
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X2_train, y_train)
y_pred = logistic_model.predict(X2_test)
accuracy_lr2 = accuracy_score(y_test, y_pred)
precision_lr2 = precision_score(y_test, y_pred, zero_division='warn')
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Logistic Regression Model Performance:")
print(f"Accuracy: {accuracy_lr2}")
print(f"Precision: {precision_lr2}")
print(f"Recall: {recall}")
print("Confusion Matrix:")
print(conf_matrix)


# In[61]:


svm_model = SVC()
svm_model.fit(X2_train, y_train)
y_pred_svm = svm_model.predict(X2_test)

accuracy_svm2 = accuracy_score(y_test, y_pred_svm)
precision_svm2 = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print("SVM Model Performance:")
print(f"Accuracy: {accuracy_svm2}")
print(f"Precision: {precision_svm2}")
print(f"Recall: {recall_svm}")
print("Confusion Matrix:")
print(conf_matrix_svm)

# In[63]:

df = df.drop(columns = "Education")
df = df.drop(columns = "Marital_Status")
df = df.drop(columns = "Dt_Customer")

# Heatmap for correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='flare')
plt.title('Correlation Heatmap')
plt.show()
