# File: eda_on_cleaned.py 

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('cleaned_exl_credit_card_churn_data.csv')
os.makedirs('cleaned_EDA', exist_ok=True)
sns.set(style="whitegrid")

print("\n[EDA] Loaded cleaned data with shape:", df.shape)

# ---------- Basic Distributions ----------

def plot_distribution(column, bins=30, kde=True, color='skyblue'):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=bins, kde=kde, color=color)
    plt.title(f'{column} Distribution')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'cleaned_EDA/{column.lower()}_distribution.png')
    plt.close()

plot_distribution('Age')

if 'Tenure' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Tenure', data=df, palette='Set2')
    plt.title('Tenure Distribution')
    plt.xlabel('Tenure (Years)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('cleaned_EDA/tenure_distribution.png')
    plt.close()

# ---------- Gender Distribution ----------
if 'Gender' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Gender', data=df, palette='icefire')
    plt.title('Gender Distribution')
    plt.tight_layout()
    plt.savefig('cleaned_EDA/gender_distribution.png')
    plt.close()

# ---------- Churn Count ----------
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='icefire')
plt.title('Churn vs Non-Churn Count')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.savefig('cleaned_EDA/churn_count.png')
plt.close()

# ---------- Correlation Matrix ----------
corr = df.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='icefire', fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('cleaned_EDA/correlation_matrix.png')
plt.close()

# ---------- Outlier Boxplots ----------
for col in ['Age', 'Balance', 'EstimatedSalary']:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f'{col} - Boxplot (Outlier Check)')
        plt.tight_layout()
        plt.savefig(f'cleaned_EDA/{col.lower()}_boxplot.png')
        plt.close()

# ---------- Age vs Churn ----------
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='Age', data=df, palette='Set3')
plt.title('Age vs Churn')
plt.xlabel('Churn')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig('cleaned_EDA/age_vs_churn.png')
plt.close()

# ---------- Pairplot (Light Version) ----------
selected_cols = ['Age', 'Balance', 'EstimatedSalary', 'Churn']
if all(col in df.columns for col in selected_cols):
    sns.pairplot(df[selected_cols], hue='Churn', palette='husl')
    plt.savefig('cleaned_EDA/pairplot.png')
    plt.close()

print("\n[EDA] Visualizations saved in the 'eda' folder")
