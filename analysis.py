"""
Heart Attack Risk Analysis & Prediction - Comprehensive PBL Project
Course: Fundamentals of Data Analytics Lab (25B16CS211)
Authors: Aditya Dev Sharma & Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             roc_auc_score, roc_curve, precision_recall_curve)
import os

# --- INITIALIZATION ---
sns.set(style="whitegrid", palette="deep")
if not os.path.exists('plots'):
    os.makedirs('plots')

# --- DATA LOADING ---
print("[1/5] Loading and inspecting dataset...")
try:
    df = pd.read_csv('DOC-20260206-WA0006 (1).csv')
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# --- DATA PREPROCESSING & FEATURE ENGINEERING ---
print("[2/5] Performing feature engineering...")

# 1. Parsing Blood Pressure
# Some entries might have issues, we use a robust split
df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df['Pulse Pressure'] = df['Systolic'] - df['Diastolic']

# 2. Advanced Feature: BMI Categorization
df['BMI_Category'] = pd.cut(df['BMI'], 
                             bins=[0, 18.5, 25, 30, 100], 
                             labels=[0, 1, 2, 3]) # Under, Normal, Over, Obese

# 3. Encoding Categorical Variables
df['Sex_Code'] = df['Sex'].map({'Male': 1, 'Female': 0})
df['Diet_Code'] = df['Diet'].map({'Unhealthy': 0, 'Average': 1, 'Healthy': 2})

# Selecting comprehensive feature set
features = [
    'Age', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History', 
    'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 
    'Stress Level', 'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 
    'Physical Activity Days Per Week', 'Sleep Hours Per Day', 
    'Systolic', 'Diastolic', 'Pulse Pressure', 'Sex_Code', 'Diet_Code'
]
X = df[features]
y = df['Heart Attack Risk']

# Handling missing values (if any)
X = X.fillna(X.median())

# Scaling for Logistic Regression and Gradient Boosting stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- ADVANCED EDA ---
print("[3/5] Generating analytical visualizations...")

# 1. Target Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Heart Attack Risk', data=df, palette='viridis')
plt.title('Distribution of Heart Attack Risk (0=Low, 1=High)')
plt.savefig('plots/risk_distribution.png')
plt.close()

# 2. Correlation Matrix
plt.figure(figsize=(15, 12))
mask = np.triu(np.ones_like(X.corr(), dtype=bool))
sns.heatmap(X.corr(), mask=mask, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Triangular Correlation Matrix of Features')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# 3. Age & Gender Risk Analysis
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='Age', hue='Heart Attack Risk', fill=True, common_norm=False, alpha=0.5)
plt.title('Age Density Estimate by Risk Category')
plt.savefig('plots/age_risk_density.png')
plt.close()

# --- MODELING & COMPARISON ---
print("[4/5] Training and cross-validating models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42)
}

model_comparison = []

for name, model in models.items():
    print(f"  Processing {name}...")
    # 5-Fold Cross-Validation
    cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    # Final Fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    model_comparison.append({
        "Model": name,
        "Accuracy": acc,
        "CV Accuracy": cv_acc,
        "ROC-AUC": auc
    })
    
    # Save Confusion Matrix for the best performing model on Accuracy
    if name == "Gradient Boosting":
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('plots/confusion_matrix.png')
        plt.close()

# Comparison Results
comp_df = pd.DataFrame(model_comparison)
print("\n--- Model Performance Comparison ---")
print(comp_df)

# ROC Curve Visualization
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves Comparison')
plt.legend()
plt.savefig('plots/model_comparison_roc.png')
plt.close()

# --- INSIGHTS ---
print("[5/5] Extracting final insights...")

# Feature Importance from Gradient Boosting
gb_model = models["Gradient Boosting"]
feat_imp = pd.Series(gb_model.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feat_imp.head(10).plot(kind='barh', color='teal')
plt.title('Top 10 Significant Risk Predictors (Gradient Boosting)')
plt.savefig('plots/feature_importance.png')
plt.close()

print("\nFinal Analysis Complete. All artifacts saved to 'plots/'.")
comp_df.to_csv('model_results_summary.csv', index=False)
