import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

def handle_imbalance(X_train, y_train):
    """Applies SMOTE to balance the target classes."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def run_model_tournament(X_train, X_test, y_train, y_test):
    """Runs a baseline tournament to track individual model performance."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    performance_data = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        performance_data.append({
            'Model': name, 
            'Accuracy (%)': round(acc * 100, 2), 
            'F1-Score': round(f1, 4)
        })
    
    summary_df = pd.DataFrame(performance_data).sort_values(by='Accuracy (%)', ascending=False)
    return summary_df

def run_stacked_model(X_train, X_test, y_train, y_test):
    """Combines top performers into a Stacking Classifier."""
    level_0_estimators = [
        ('xgb', XGBClassifier(learning_rate=0.05, n_estimators=100, max_depth=5, eval_metric='logloss', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ]
    
    stack = StackingClassifier(
        estimators=level_0_estimators, 
        final_estimator=LogisticRegression(),
        passthrough=True 
    )
    
    stack.fit(X_train, y_train)
    return stack

def optimize_threshold(model, X_test, y_test):
    """Finds the probability threshold that maximizes accuracy."""
    y_probs = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.3, 0.7, 0.01)
    accuracies = []
    
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        accuracies.append(accuracy_score(y_test, y_pred_t))
    
    best_t = thresholds[np.argmax(accuracies)]
    best_acc = max(accuracies)
    
    print(f"\n--- THRESHOLD OPTIMIZATION ---")
    print(f"Optimal Threshold: {best_t:.2f}")
    print(f"Max Accuracy at this Threshold: {best_acc:.2%}")
    
    y_final_pred = (y_probs >= best_t).astype(int)
    print("\nDetailed Report at Optimal Threshold:")
    print(classification_report(y_test, y_final_pred))
    
    return best_t

def get_feature_importance(model, X_train):
    """Visualizes feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=X_train.columns)
        feat_importances.nlargest(10).plot(kind='barh', color='skyblue')
        plt.title("Top 10 Features")
        plt.show()
    else:
        print("Model does not support direct feature importance plotting.")