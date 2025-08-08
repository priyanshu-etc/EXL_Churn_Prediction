# Submitted by: Priyanshu

# File: data_cleaning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('exl_credit_card_churn_data.csv')

print("\nBefore Cleaning - Missing Values Count:")
print(df.isnull().sum())

# Display Initial shape of the dataset
print("\nInitial shape of the dataset:", df.shape)

# Save missing value heatmap before cleaning
plt.figure(figsize=(20, 10))
sns.heatmap(df.isnull(), cbar=False, cmap="plasma")
plt.title("Missing Values - Before Cleaning")
plt.tight_layout()
plt.savefig("missing_before_cleaning.png")
plt.close()

# --- CLEANING STEPS ---

# Clean and standardize 'Gender' column (remove spaces, capitalize first letter)
df['Gender'] = df['Gender'].astype(str).str.strip().str.title()

# Clean 'HasCrCard' and 'IsActiveMember' columns (remove spaces, convert to lowercase)
df['HasCrCard'] = df['HasCrCard'].astype(str).str.strip().str.lower()
df['IsActiveMember'] = df['IsActiveMember'].astype(str).str.strip().str.lower()

# Mapping yes/no/1/0 to binary values (1 and 0)
yes_no_map = {'yes': 1, 'no': 0, '1': 1, '0': 0}
df['HasCrCard'] = df['HasCrCard'].map(yes_no_map) 
df['IsActiveMember'] = df['IsActiveMember'].map(yes_no_map)

# Keep only rows where gender is 'Male' or 'Female'
df = df[df['Gender'].isin(['Male', 'Female'])] 

# Convert 'Age' to numeric and filter for valid age range (18 to 100)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce') #coerce converts invalid parsing to NaN
df = df[df['Age'].between(18, 100)]

# Convert 'Balance' to numeric and remove negative balances
df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
df = df[df['Balance'] >= 0]

# Convert 'EstimatedSalary' to numeric and filter for positive salaries
df['EstimatedSalary'] = pd.to_numeric(df['EstimatedSalary'], errors='coerce')
df = df[df['EstimatedSalary'] > 0]

# Clean 'Churn' column and keep only rows with valid values ('0' or '1')
df['Churn'] = df['Churn'].astype(str).str.strip()
df = df[df['Churn'].isin(['0', '1'])]

# Convert 'Churn' to integer
df['Churn'] = df['Churn'].astype(int)

# Drop any rows with remaining missing values
df.dropna(inplace=True)

# Convert selected columns to integer type for consistency
for col in ['Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']:
    if col in df.columns:
        df[col] = df[col].astype(int)


# Save cleaned file
df.to_csv('cleaned_exl_credit_card_churn_data.csv', index=False)

print("\nAfter Cleaning - Missing Values Count:")
print(df.isnull().sum())

# Save missing value heatmap after cleaning
plt.figure(figsize=(20, 10))
sns.heatmap(df.isnull(), cbar=False, cmap="plasma")
plt.title("Missing Values - After Cleaning")
plt.tight_layout()
plt.savefig("missing_after_cleaning.png")
plt.close()

print("\nData cleaned and saved as 'cleaned_exl_credit_card_churn_data.csv'")

# Display final shape of the cleaned dataset
print("\nFinal shape of cleaned dataset:", df.shape)
