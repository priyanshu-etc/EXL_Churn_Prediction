#Handle outliers
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('cleaned_exl_credit_card_churn_data.csv')

#describe the dataset before handling outliers
print("\n[Outlier Handling] Data shape before handling outliers:", df.shape)

# Define columns for outlier handling
columns_to_check = ['Age', 'Tenure', 'Balance', 'EstimatedSalary']

# Function to apply IQR (Interquartile Range) filtering to remove outliers
def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)  # 25th percentile
    Q3 = dataframe[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1                          # Interquartile range
    lower_bound = Q1 - 1.5 * IQR           # Lower limit for outlier detection
    upper_bound = Q3 + 1.5 * IQR           # Upper limit for outlier detection
    # Return rows within the IQR range (i.e., without outliers)
    return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

# Apply IQR filtering to each numeric column listed in `columns_to_check`
for col in columns_to_check:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)  # Remove outliers from each column
        # print(f"Removed outliers from {col}: {df[col].describe()}")

# Save the result to a new CSV file
output_file = 'outlier_handled_exl_credit_card_churn_data.csv'
df.to_csv(output_file, index=False)

print(f"Outliers removed using IQR method for columns {columns_to_check}")
#describe the dataset after handling outliers
print("\n[Outlier Handling] Data shape after handling outliers:", df.shape)
print(f"File saved as: {output_file}")
