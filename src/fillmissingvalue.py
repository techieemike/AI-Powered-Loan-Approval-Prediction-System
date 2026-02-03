import numpy as np
import pandas as pd

def clean_data(df):
    # Step 1: Handle numerical missing values
    num_col = ['LoanAmount']
    for feature in num_col:
        if feature in df.columns:
            df[feature].fillna(df[feature].median(), inplace=True)

    # Step 2: Handle categorical missing values
    cat_col = ['Credit_History', 'Self_Employed', 'Dependents',
               'Loan_Amount_Term', 'Gender', 'Married']
    for feature in cat_col:
        if feature in df.columns:
            df[feature].fillna(df[feature].mode()[0], inplace=True)
    
        