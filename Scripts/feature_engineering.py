# File: feature_engineering.py 

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('outlier_handled_exl_credit_card_churn_data.csv')

print("\nInitial shape of data before feature engineering:", df.shape)

# ----- Visualize distributions before feature engineering -----
plt.figure(figsize=(8, 4))
sns.histplot(df['Balance'], bins=30, kde=True, color='violet')
plt.title('Balance - Before Feature Engineering')
plt.tight_layout()
plt.savefig('balance_before_engineering.png')
plt.close()

# --------- Feature Engineering ---------

# Gender Encoding
if 'Gender' in df.columns:
    df['GenderEncoded'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Balance to Salary Ratio
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1) # +1 to avoid division by zero

# Has Balance (binary)
df['HasBalance'] = (df['Balance'] > 0).astype(int)

# Tenure to Age Ratio
df['TenureAgeRatio'] = df['Tenure'] / (df['Age'] + 1)

# Visualize a newly created feature
plt.figure(figsize=(8, 4))
sns.histplot(df['BalanceSalaryRatio'], bins=30, kde=True, color='cyan')
plt.title('BalanceSalaryRatio - New Feature')
plt.tight_layout()
plt.savefig('balance_salary_ratio_distribution.png')
plt.close()

# --------- MinMax Scaling ---------
scaler = MinMaxScaler()
scale_cols = ['Age', 'Tenure', 'Balance', 'EstimatedSalary']

# Save unscaled distribution plot
for col in scale_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'{col} - Before Scaling')
    plt.tight_layout()
    plt.savefig(f'{col.lower()}_before_scaling.png')
    plt.close()

# Apply scaling
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Save scaled distribution plot
for col in scale_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], bins=30, kde=True, color='purple')
    plt.title(f'{col} - After MinMax Scaling')
    plt.tight_layout()
    plt.savefig(f'{col.lower()}_after_scaling.png')
    plt.close()

# Save engineered dataset
output_file = 'engineered_exl_credit_card_churn_data.csv'
df.to_csv(output_file, index=False)

print("\nFeature engineering complete.")
print(f"Final shape of dataset: {df.shape}")
print(f"File saved as: {output_file}")
