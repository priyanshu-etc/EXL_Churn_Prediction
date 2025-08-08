# EXL Churn Prediction

## 📌 Project Overview
This project predicts **credit card customer churn** for EXL using historical customer data.  
It involves **data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning** using a **Random Forest Classifier**.

## 📂 Folder Structure

## 🚀 Workflow
1. **Data Cleaning** → Handle missing values, outliers, and inconsistent formatting.  
2. **EDA** → Understand data distribution and correlations through visualizations.  
3. **Feature Engineering** → Create new features like `balance_salary_ratio`, scaling numerical features.  
4. **Model Training** → Train a **Random Forest Classifier** to predict churn.  
5. **Evaluation** → Evaluate with accuracy, precision, recall, and confusion matrix.  

## 📊 Key Visualizations
- Age, balance, and salary distribution before & after scaling  
- Correlation heatmap  
- Feature importance ranking  
- Confusion matrix for model performance  

## 🛠 Tech Stack
- **Language**: Python 3.x  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  

## 📥 Installation & Usage
```bash
# Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd EXL_Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Run scripts
python Scripts/data_cleaning.py
python Scripts/feature_engineering.py
python Scripts/random_forest.py

