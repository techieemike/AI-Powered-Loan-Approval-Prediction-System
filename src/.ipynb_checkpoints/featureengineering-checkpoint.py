import numpy as np
import pandas as pd

def process_loan_data(df):
    #df = df.copy()

    # 1. Target Mapping
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

    # 2. Base Features & Ratios
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['Total_Income']
    df['Monthly_Installment'] = df['LoanAmount'] / df['Loan_Amount_Term'].replace({0: 360, np.nan: 360})
    
    # 3. Binning Loan Amount (New Signal)
    # Categorizing loans helps models handle outliers better
    df['Loan_Amount_Bin'] = pd.cut(df['LoanAmount'], bins=[0, 100, 200, 700], labels=['Low', 'Medium', 'High'])

    # 4. All Log Transformations (Restored)
    log_map = {
        'Log_ApplicantIncome': 'ApplicantIncome',
        'Log_LoanAmount': 'LoanAmount',
        'Log_Total_Income': 'Total_Income',
        'Log_Loan_to_Income': 'Loan_to_Income_Ratio'
    }
    for new_col, old_col in log_map.items():
        if old_col in df.columns:
            df[new_col] = np.log1p(df[old_col])

    # 5. Categorical Encoding (Including our new Bin)
    cat_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Amount_Bin']
    existing_cat = [c for c in cat_features if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat, drop_first=True, dtype=int)

    # 6. Drop Raw Columns
    cols_to_drop = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'Loan_ID']
    df_final = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df_final