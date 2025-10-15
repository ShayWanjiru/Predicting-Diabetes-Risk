# Predicting Diabetes Risk in Underserved Populations Using Demographic and Health Data

### Author
**Shalyne Wanjiru Murage**  
Bachelor of Science in Data Science and Analytics  
United States International University – Africa  
Fall Semester 2025  

---

## 1. Overview

This project develops a machine learning framework to predict the risk of diabetes using easily accessible demographic and health data. The goal is to create an interpretable, data-driven system that can support early diagnosis efforts in underserved populations where access to healthcare is limited.

The work integrates predictive modeling with explainable artificial intelligence (XAI) techniques such as **SHapley Additive Explanations (SHAP)** to ensure transparency and interpretability of results. By identifying key risk indicators such as **glucose levels, body mass index (BMI), and age**, the study aims to support healthcare providers and policymakers with actionable insights for targeted interventions.

---

## 2. Problem Statement

Diabetes remains one of the fastest-growing global health concerns, disproportionately affecting individuals in low- and middle-income regions. Early diagnosis is often hindered by lack of access to health facilities, high testing costs, and limited awareness.

This project addresses these challenges by leveraging demographic and health data to build a **non-invasive, affordable, and explainable prediction model** that identifies individuals at high risk of developing diabetes before the onset of severe symptoms.

---

## 3. Objectives

- **Primary Objective:**  
  Develop and validate a predictive model capable of estimating an individual’s likelihood of having or developing diabetes using demographic and physiological attributes.

- **Secondary Objectives:**  
  1. Compare multiple machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting).  
  2. Optimize and evaluate models through cross-validation and hyperparameter tuning.  
  3. Apply interpretability techniques (SHAP) to explain the influence of each feature on predictions.  
  4. Examine variations in risk across demographic groups and propose data-driven interventions.

---

## 4. Methodology

### Data Sources
- **Primary Dataset:** Pima Indians Diabetes Dataset (UCI Repository)  
- **Additional References:** NHANES and DHS datasets for potential model validation  
- **Format:** Structured tabular data (768 rows × 9 variables)  

### Modeling Process
1. **Data Preprocessing:**  
   - Handled missing values and standardized numeric attributes.  
   - Scaled features using StandardScaler.  
   - Encoded target variable (`Outcome`: 0 = No Diabetes, 1 = Diabetes).  
2. **Modeling:**  
   - Implemented Logistic Regression, Random Forest, and Gradient Boosting.  
   - Evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.  
3. **Interpretability:**  
   - Used SHAP (Explainable AI) to interpret model decisions and identify top predictors.  
4. **Optimization:**  
   - Applied GridSearchCV and cross-validation to tune hyperparameters.  

---

## 5. Key Results (Current Progress)

| Model | Accuracy | F1-Score | ROC-AUC |
|:------|:----------|:----------|:----------|
| Logistic Regression | 0.78 | 0.64 | 0.81 |
| Random Forest (Tuned) | **0.85** | **0.70** | **0.86** |
| Gradient Boosting | 0.84 | 0.71 | 0.87 |

- **Top Predictors:** Glucose, BMI, and Age.  
- **Best Performing Model:** Tuned Random Forest (selected for production pipeline).  
- **Explainability Tools:** SHAP plots and feature importance analyses (pending final correction due to visualization errors).

---

## 6. Technical Challenges and Current Fixes

| Issue | Cause | Proposed Solution |
|:------|:-------|:------------------|
| SHAP Comparison Plot Failure | Multidimensional SHAP arrays | Aggregate or flatten arrays before normalization |
| Production Pipeline Error | Custom preprocessor not compatible with sklearn pipeline | Convert class to TransformerMixin or use ColumnTransformer |
| Deployment Report Error | Missing metric keys (`roc_auc`) due to failed training | Ensure final model fits successfully before generating reports |

---

## 7. Tools and Technologies

- **Languages:** Python 3.13  
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, shap  
- **Development Environment:** Jupyter Notebook  
- **Version Control:** GitHub  
