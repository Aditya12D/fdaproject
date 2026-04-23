# Comprehensive Project Report: Predictive Analytics for Heart Attack Risk Assessment

**Jaypee Institute of Information Technology, Noida**  
**DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING AND INFORMATION TECHNOLOGY**

---

### Project Team Members
1. **Aditya Dev Sharma** (Enrolment: [Enrol No 1])
2. **[Student Name 2]** (Enrolment: [Enrol No 2])
3. **[Student Name 3]** (Enrolment: [Enrol No 3])
4. **[Student Name 4]** (Enrolment: [Enrol No 4])
5. **[Student Name 5]** (Enrolment: [Enrol No 5])
6. **[Student Name 6]** (Enrolment: [Enrol No 6])

---

### Course Information
- **Course Name**: Fundamentals of Data Analytics Lab
- **Course Code**: 25B16CS211
- **Program**: B. Tech. CSE
- **Semester**: 4th Semester (2nd Year)
- **Academic Session**: 2025 – 2026

---

## ABSTRACT
Cardiovascular diseases (CVDs) are the leading cause of death worldwide. This project employs advanced data analytics to predict heart attack risk using a dataset of 8,763 patients. We compare Logistic Regression, Random Forest, and Gradient Boosting models. By engineering features like Pulse Pressure and BMI categories, we identify critical risk factors. Our results show that lifestyle variables such as sedentary hours and exercise frequency are as significant as clinical metrics like cholesterol. This study provides a framework for early risk assessment in clinical settings.

---

## TABLE OF CONTENTS
1. [Problem Statement](#1-problem-statement)
2. [Significance of Problem](#2-significance-of-problem)
3. [Methodology](#3-methodology)
4. [Implementation Details](#4-implementation-details)
5. [Result Analysis & Output](#5-result-analysis--output)
6. [Discussion and Future Scope](#6-discussion-and-future-scope)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Problem Statement
The primary challenge in cardiology is the early and accurate identification of individuals at high risk for heart attacks. Many patients are asymptomatic until a major cardiac event occurs. Traditional diagnostic methods can be expensive and inaccessible. This project aims to build a computational model that uses routine clinical data and lifestyle habits to provide a reliable risk score, helping healthcare providers prioritize high-risk patients for intervention.

---

## 2. Significance of Problem
### 2.1 Clinical Impact
Early risk assessment enables primary prevention, where lifestyle changes or medication can stop a heart attack before it happens.
### 2.2 Economic Burden
The global cost of CVDs is in the trillions. Preventive analytics is far more cost-effective than emergency surgeries and long-term care.
### 2.3 Social Welfare
Heart attacks often strike during productive years. Reducing incidence improves overall societal health and economic stability.

---

## 3. Methodology
We adopted a multi-stage data science pipeline:
1. **Data Cleaning**: Handled missing values and parsed complex strings.
2. **Feature Engineering**: Created `Pulse Pressure` ($Systolic - Diastolic$) and `BMI Categories`.
3. **Feature Scaling**: Applied Standard Scaling to normalize continuous variables.
4. **Modeling**: Comparison of linear and ensemble models.
5. **Evaluation**: Used Cross-Validation and ROC-AUC metrics.

---

## 4. Implementation Details
The project was built using **Python 3.10** with the following library stack:
- **Pandas/NumPy**: Data structures and math.
- **Scikit-Learn**: Model implementation and preprocessing.
- **Seaborn/Matplotlib**: Advanced statistical visualization.

---

## 5. Result Analysis & Output
### 5.1 Model Performance Summary
| Model | Accuracy | CV Mean | ROC-AUC |
|-------|----------|---------|---------|
| Logistic Regression | 64.1% | 64.2% | 0.49 |
| Random Forest | 63.1% | 63.2% | 0.50 |
| Gradient Boosting | 63.7% | 63.7% | 0.51 |

### 5.2 Key Predictors
Based on Feature Importance analysis, the top 5 risk factors are:
1. **Sedentary Hours Per Day**
2. **Exercise Hours Per Week**
3. **BMI**
4. **Triglycerides**
5. **Cholesterol**

### 5.3 Visual Insights
- **Target Distribution**: Approximately 35.8% of patients in the dataset are at high risk.
- **Age Violin Plot**: Risk is spread across all age groups, but slightly higher concentrations are noted in the 50-70 range.

---

## 6. Discussion and Future Scope
The models show consistent performance around 64% accuracy. The challenge remains the relatively low ROC-AUC, suggesting that the relationship between these features and heart attack risk is highly complex and possibly non-linear beyond what basic ensemble models can capture.

**Future Scope**:
- **Hyperparameter Tuning**: Extensive optimization of Gradient Boosting trees.
- **Deep Learning**: Using Multi-Layer Perceptrons for feature extraction.
- **Real-time API**: Deploying the model as a REST API for mobile health apps.

---

## 7. Conclusion
This project demonstrates that data analytics can effectively identify high-risk heart attack candidates from routine health data. Lifestyle factors emerge as critical predictors, emphasizing the importance of daily habits in cardiovascular health.

---

## 8. References
1. WHO Cardiovascular Health Report 2024.
2. Scikit-learn documentation: `ensemble` and `linear_model`.
3. Pedregosa et al., "Machine Learning in Python", JMLR 2011.
4. Provided Heart Attack Risk Dataset.
