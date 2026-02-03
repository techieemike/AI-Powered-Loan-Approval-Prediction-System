# Loan Approval Prediction System

![Project Banner](img/banner.jpeg)


## 🧠 Overview
This project is an advanced machine learning system designed to predict loan approval with a heavy focus on **interpretability**. Unlike "black-box" models, this system utilizes a **Stacked Ensemble Model** (Logistic Regression, Random Forest, and XGBoost) and leverages **SHAP** and **LIME** to provide transparent, case-by-case justifications for every decision.

---

## 📊 Business Relevance

In traditional banking, loan approval processes are often plagued by three main issues:
1. **Slow Turnaround:** Manual reviews can take days or weeks.
2. **Subjectivity:** Human loan officers may have unconscious biases.
3. **Information Gap:** Applicants often don't understand why they were rejected.

**This AI solution addresses these challenges by:**
* **Accelerating Decisions:** Providing near-instant predictions using a high-accuracy Stacked Ensemble.
* **Ensuring Fairness:** Using data-driven logic to minimize subjective bias.
* **Enhancing Transparency:** Utilizing **LIME** and **SHAP** to provide "Right to Explanation," showing applicants the exact financial factors (like Loan-to-Income ratio) that influenced their result.

---

## 🏗️ Project Structure

AI Loan Approval Prediction Systems/
├── 📁 dataset/                       # Raw, balanced, and processed CSV files
│   ├── train_data.csv
│   └── test_data.csv
│   └── final_loan_predictions_submission.csv
├── 📁 img/                            # SHAP plots, LIME outputs, banner and UI screenshots
├── 📁 src/                          
│   ├── fillmissingvalue.py            # Automated data cleaning
│   ├── visualization.py               # Charts & plots (EDA, feature importance)
│   ├── AI_Loan_Approval_Prediction.ipynb  # Full workflow: model, feature engineering
│   └── Data Results.html              # ydata-profiling report
│   └── featureengineering.py           # Log transforms & Ratio calculations
│   └── model.py                       # Stacked Ensemble & Threshold logic
├── 📁 models/
│   └── loan_model_stacked.pkl        # Final trained Stacked Classifier 
├── 📁 venv/
├── 📄 README.md                     # Documentation
├── 📄 requirements.txt              # Cloud deployment dependencies
└── 📄 app.py                        # Streamlit Dashboard (Deployed)
└── 📄 loan_model_stacked.pkl 





---

## 🚀 Tech Stack

### **Data Science & Machine Learning**
- **Core:** Python (Pandas, NumPy)
- **Modeling:** Scikit-Learn (Logistic Regression, Random Forest)
- **Ensemble Learning:** XGBoost & StackingClassifier
- **Class Balancing:** SMOTE (imbalanced-learn)
- **Model Persistence:** Joblib / Pickle

### **Explainable AI (XAI)**
- **SHAP:** Global feature importance and model "brain" visualization.
- **LIME:** Local, instance-level explanations for individual loan decisions.

### **Deployment & Visualization**
- **App Framework:** Streamlit
- **Web Hosting:** Streamlit Community Cloud
- **Profiling:** ydata-profiling (formerly Pandas Profiling)
- **Charts:** Matplotlib, Seaborn, and Plotly

### **DevOps & Tools**
- **Version Control:** Git & GitHub
- **Environment:** Virtual Environments (venv)
- **Notebooks:** Jupyter Notebook / VS Code

---

## 🔍 Key Features

### **1. Advanced Ensemble Modeling**
- **Stacked Classifier:** Combines **Logistic Regression**, **Random Forest**, and **XGBoost** into a single "meta-model" to capture both linear and non-linear relationships.
- **Optimized Probability Threshold:** Custom-tuned decision threshold of **0.31** to improve approval sensitivity and model fairness.

### **2. Explainable AI (XAI) Dashboard**
- **Global Transparency (SHAP):** Uses SHAP (SHapley Additive exPlanations) to rank features by their overall impact on the model's logic.
- **Local Justification (LIME):** Provides a "Reasoning Report" for every individual prediction, showing exactly which factors (e.g., Credit History vs. Income) led to a specific approval or rejection.

### **3. Automated Feature Engineering**
- **Debt-to-Income Analysis:** Automatically calculates the `Loan_to_Income_Ratio` to assess borrower stress.
- **Repayment Estimation:** Computes `Monthly_Installment` based on loan amount and term to simulate real-world affordability.
- **Distribution Normalization:** Applies Log-transformations to skewed financial data to improve model convergence.

### **4. Robust Data Pipeline**
- **Smart Imputation:** Handles missing data using specialized strategies (Mode for categorical, Median for numerical) to maintain data integrity.
- **Class Balancing:** Addresses the "Approval Bias" in historical data using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns from both successes and failures equally.

---

## 🚀 How to Run

### **Option 1: Live Web Demo (Streamlit Cloud)**
The easiest way to explore this project is through the live interactive dashboard:
👉 **[Insert Your Streamlit URL Here]**

---

### **Option 2: Local Installation**

If you want to run the system on your machine, follow these steps:

#### **1. Clone the Repository**
```bash
git clone https://github.com/techieemike/AI-Loan-Approval-Prediction-System.git
cd AI-Loan-Approval-Prediction-System

2. Create and activate a virtual environment:
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Launch the Streamlit app:
   streamlit run app2.py



#### **2. Set up Virtual Environment

# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate


#### **3. Install Dependencies
pip install -r requirements.txt





#### **4. Launch the Dashboard
streamlit run app.py






---

## 🗺️ Project Roadmap

- [x] **Phase 1: Data Exploration & Cleaning**
  - Exploratory Data Analysis (EDA) to identify trends and outliers.
  - Handling missing values using statistical imputation (Median/Mode).
  - Visualization of class imbalances (Approved vs. Rejected).

- [x] **Phase 2: Advanced Feature Engineering**
  - Calculation of financial ratios (`Loan_to_Income_Ratio`).
  - Creation of repayment metrics (`Monthly_Installment`).
  - Log-transformation of skewed numerical features for model stability.

- [x] **Phase 3: Model Development & Optimization**
  - Implementation of a **Stacked Ensemble Classifier** (Logistic Regression + Random Forest + XGBoost).
  - Class balancing using **SMOTE** to handle data disparity.
  - Hyperparameter tuning and custom **Probability Threshold (0.31)** optimization.

- [x] **Phase 4: Explainable AI (XAI) Integration**
  - Global model interpretation using **SHAP** summary plots.
  - Individual "local" prediction justifications using **LIME**.

- [x] **Phase 5: Deployment**
  - Developing a user-friendly UI with **Streamlit**.
  - Hosting the live application on **Streamlit Community Cloud**.

- [ ] **Phase 6: Future Enhancements (Next Steps)**
  - Integration of a real-time database (PostgreSQL) to store application history.
  - Implementation of automated "Retraining Loops" as new data arrives.




---

## 📈 Model Performance & Results

The final system utilizes a **Stacked Generalization Ensemble**, which combines the strengths of multiple base learners to achieve a more robust prediction than any single model.

### **1. Accuracy Comparison**
During testing, the Stacked Model consistently outperformed individual classifiers:

| Model                | Accuracy | Precision (Approved) | Recall (Approved) |
|:---------------------|:---------|:---------------------|:------------------|
| **Stacked Ensemble** | **0.86** | **0.84** | **0.91** |
| Logistic Regression  | 0.78     | 0.77                 | 0.82              |
| Random Forest        | 0.76     | 0.75                 | 0.79              |
| Decision Tree        | 0.69     | 0.68                 | 0.70              |

### **2. The 0.31 Decision Threshold**
Standard models use a default threshold of **0.50**. However, for this project, the threshold was optimized to **0.31**.

- **Why 0.31?** Financial institutions often prefer a model that is more "sensitive" to potential approvals while maintaining risk control. 
- **The Result:** By lowering the threshold to 0.31, we significantly increased the **Recall**, ensuring that fewer qualified applicants are wrongly rejected (False Negatives).

### **3. Feature Importance (SHAP)**
According to our SHAP global analysis, the top 3 drivers for loan approval in this model are:
1. **Credit History:** The single most dominant predictor.
2. **Log_Total_Income:** Higher stability leads to higher approval probability.
3. **Monthly_Installment:** Low monthly burdens strongly favor the applicant.

---







---

## 🔍 Feature Summary

| Feature | Description | Preprocessing / Engineering |
| :--- | :--- | :--- |
| **Gender** | Applicant's gender (Male/Female) | One-Hot Encoded |
| **Married** | Marital status (Yes/No) | One-Hot Encoded |
| **Dependents** | Number of people dependent on applicant | One-Hot Encoded (0, 1, 2, 3+) |
| **Education** | Academic qualification (Graduate/Not) | One-Hot Encoded |
| **Self_Employed** | Employment type status | One-Hot Encoded |
| **Credit_History** | 1.0 (Good) or 0.0 (Bad) | Cleaned & Kept As-is |
| **Property_Area** | Urban, Semiurban, or Rural | One-Hot Encoded |
| **Log_Total_Income** | Combined Applicant + Coapplicant income | Log Transformation ($np.log1p$) |
| **Log_LoanAmount** | Total loan amount requested | Log Transformation ($np.log1p$) |
| **Monthly_Installment** | Estimated monthly repayment | Engineered: $LoanAmount / Term$ |
| **Log_Loan_to_Income** | Ratio of loan to total income | Engineered & Log Transformed |
| **Log_Loan_Term** | Duration of the loan in months | Log Transformation ($np.log1p$) |

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork** the Project
2. Create your **Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** your Changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the Branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

---

## 🛡️ Badges

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)

---

## 📄 License

Distributed under the **MIT License**. See the `LICENSE` file for more information.

---

## 🙋‍♂️ Author

**Abikale Michael Raymond** *AI | AI Automation | Full Stack Developer* - **LinkedIn:** [Michael Raymond Abikale](https://www.linkedin.com/in/michael-raymond-abikale-27363949/)
- **GitHub:** [@techieemike](https://github.com/techieemike)
- **Email:** [abikalemichaelraymond@gmail.com](mailto:abikalemichaelraymond@gmail.com)