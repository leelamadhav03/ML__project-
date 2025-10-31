#!/usr/bin/env python
# coding: utf-8

# In[124]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

#Observation: Imported all required libraries and suppressed warnings for clean output.


# 

# In[125]:


df = pd.read_csv('IPO.csv')
print("First 5 rows of the dataset:")
df.head()
#Observation: Loaded IPO dataset and displayed initial rows for structural understanding.


# In[126]:


df.shape
#Observation: Checked the number of rows and columns to understand dataset size.


# In[127]:


df.info()
# Observation: Verified data types and presence of missing values.


# In[128]:


print("\n Descriptive Statistics:")
display(df.describe().T)
#Observation: Viewed statistical summary (mean, std, range) for numerical columns.


# In[129]:


df.isna().sum()
#Observation: Identified columns with null values for later imputation.


# In[130]:


df['Issue_Size(crores)'] = df['Issue_Size(crores)'].astype(str).str.replace(',', '', regex=False)
df['Issue_Size(crores)'] = pd.to_numeric(df['Issue_Size(crores)'], errors='coerce')
#Observation: Removed commas and converted ‘Issue_Size(crores)’ to numeric format.


# In[131]:


numeric_cols_for_outliers = [
    "Issue_Size(crores)", "QIB", "HNI", "RII",
    "Listing_Open", "Listing_Close", "Listing_Gains(%)",
    "CMP", "Current_gains"
]
#Observation: Ensured all numeric columns are in the correct numeric format.


# In[132]:


for col in numeric_cols_for_outliers:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    #Observation: Visualized distribution and detected potential outliers before treatment.


# In[133]:


plt.figure(figsize=(15,8))
sns.boxplot(data=df[numeric_cols_for_outliers], orient="h")
plt.title("Boxplot Before Outlier Treatment")
plt.show()


# In[134]:


def treat_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
    df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
    return df
for col in numeric_cols_for_outliers:
    df = treat_outliers_iqr(df, col)
  # Observation: Applied IQR-based capping to handle extreme outlier values.


# In[135]:


plt.figure(figsize=(15,8))
sns.boxplot(data=df[numeric_cols_for_outliers], orient="h")
plt.title("Boxplot After Outlier Treatment")
plt.show()


# In[136]:


for col in numeric_cols_for_outliers:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)

print("\nMissing Values After Median Substitution:")
display(df.isna().sum())
#Observation: Replaced remaining NaN values with the median of each column.
# And Verified all missing values are handled successfully.


# In[137]:


df_encoded = pd.get_dummies(df, columns=['IPO_Name'], drop_first=True)
df_encoded['Date'] = pd.to_datetime(df_encoded['Date'], format='%d-%m-%Y', errors='coerce')
df_encoded['Year'] = df_encoded['Date'].dt.year
df_encoded['Month'] = df_encoded['Date'].dt.month
#Observation: Converted categorical IPO names into dummies and extracted year & month from date.


# In[138]:


df_encoded.drop(columns=['Date'], inplace=True)
#Observation: Removed redundant date column after feature extraction.


# In[139]:


y = df_encoded['CMP']
X = df_encoded.drop(columns=['CMP'])
#Observation: Separated dependent variable (CMP) and independent features (X).


# In[140]:


scaler = StandardScaler()

numeric_cols_for_scaling = [col for col in numeric_cols_for_outliers if col in X.columns]
numeric_cols_for_scaling += ['Year', 'Month']

X[numeric_cols_for_scaling] = scaler.fit_transform(X[numeric_cols_for_scaling])
#Observation: Standardized numerical variables to bring all features on comparable scales.


# In[141]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
# Observation: Split dataset into 80–20 ratio for model training and evaluation.


# In[142]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[143]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
# Observation: Fitted baseline Linear Regression model on training data.


# In[144]:


y_pred = lr_model.predict(X_test)


# In[145]:


print("First 10 Predicted CMP Values:")
print(y_pred[:10])
# Observation: Generated predicted CMP values on test data.


# In[146]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
#Observation: Evaluated model performance using MAE, MSE, RMSE, and R² metrics.


# In[147]:


print("\n📈 Linear Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")


# In[148]:


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)  # perfect prediction line
plt.xlabel("Actual Listing Gains (%)")
plt.ylabel("Predicted Listing Gains (%)")
plt.title("Actual vs Predicted Listing Gains (%)")
plt.show()


# In[149]:


#  Random Forest Regression Model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1
)
#Observation: Trained an ensemble Random Forest Regressor for improved prediction accuracy.


# In[150]:


rf_model.fit(X_train, y_train)


# In[151]:


y_pred_rf = rf_model.predict(X_test)
#Observation: Assessed model performance; typically higher R² and lower errors than Linear Regression.


# In[152]:


mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\n Random Forest Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf:.2f}")
print(f"R-squared (R²): {r2_rf:.2f}")


# In[153]:


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
plt.xlabel("Actual CMP")
plt.ylabel("Predicted CMP")
plt.title("Actual vs Predicted CMP (Random Forest)")
plt.show()
#Observation: Visualization: Actual vs Predicted for Random Forest


# In[154]:


print("\n🔍 Model Comparison:")
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [mae, mae_rf],
    'RMSE': [rmse, rmse_rf],
    'R²': [r2, r2_rf]
})
display(comparison)
# Observation: Compared both models quantitatively based on key metrics.


# In[155]:


plt.figure(figsize=(6,5))

sns.barplot(
    x='Model',
    y='R²',
    data=comparison,
    palette='coolwarm'
)

plt.title("R² Score Comparison: Linear Regression vs Random Forest", fontsize=14)
plt.ylabel("R² Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
#Observation: Visualized which model has stronger explanatory power (R²).


# In[156]:


# Test data fo Aditya Infotech Ltd (CP PLUS):
test_data = pd.DataFrame({
    'Issue_Size_crores': [1300],
    'QIB': [75],#%
    'HNI': [15],
    'RII': [10],
    'Issue_price': [675],
    'Listing_Open': [1015],
    'Listing_Close': [np.nan],
    'Listing_Gains_percent': [50.37],
    'CMP': [1373.5],
    'Current_gains': [64.81]
})
# Observation: Created new IPO record with real features for out-of-sample predictio


# In[157]:


test_data['Year'] = [2025]
test_data['Month'] = [10]


# In[158]:


for col in X.columns:
    if col not in test_data.columns:
        test_data[col] = 0

test_data = test_data[X.columns]
#  Ensure same feature columns as training data


# In[159]:


test_data[numeric_cols_for_scaling] = scaler.transform(test_data[numeric_cols_for_scaling])

predicted_cmp_rf = rf_model.predict(test_data)
# Observation: Applied same scaling transformation to test data for normalization.


# In[160]:


actual_cmp = [1373.65]
predicted_cmp = [predicted_cmp_rf[0]]


# In[161]:


# ✅ Print predicted vs actual CMP
print(f"\n Predicted CMP for Aditya Infotech Ltd IPO (Random Forest): ₹{predicted_cmp_rf[0]:.2f}")
#Observation: Predicted CMP for Aditya Infotech IPO using the trained Random Forest model.


# In[162]:


plt.figure(figsize=(6,5))
plt.bar(['Actual CMP', 'Predicted CMP'], [actual_cmp[0], predicted_cmp[0]], color=['skyblue', 'lightgreen'])
plt.ylabel('CMP (₹)')
plt.title('Canara HSBC Life Insurance IPO: Actual vs Predicted CMP')
plt.show()
# Observation: Visualized and compared actual vs predicted CMP values for new IPO.

