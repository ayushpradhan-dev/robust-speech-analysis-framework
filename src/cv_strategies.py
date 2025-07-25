# src/cv_strategies.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

def run_standard_kfold_cv(X, y, n_splits=5, n_features_to_select=50):
    """
    Performs a standard k-fold cross-validation experiment.
    
    Returns:
        tuple: (pd.DataFrame of results, list of (y_test, y_prob) for each fold)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    fold_predictions = [] # To store raw predictions for ROC curves

    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Standard K-fold")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=n_features_to_select)),
            ('classifier', SVC(kernel='linear', probability=True, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fold_predictions.append({'y_true': y_test, 'y_prob': y_prob})
        
        selected_features_mask = pipeline.named_steps['feature_selection'].get_support()
        selected_features = X.columns[selected_features_mask].tolist()

        results.append({
            'fold': fold + 1, 'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'auc': roc_auc_score(y_test, y_prob),
            'selected_features': selected_features
        })

    return pd.DataFrame(results), fold_predictions


def run_nested_kfold_cv(X, y, n_splits_outer=5, n_splits_inner=3):
    """
    Performs a nested k-fold cross-validation experiment.

    Returns:
        tuple: (pd.DataFrame of results, list of (y_test, y_prob) for each fold)
    """
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    results = []
    fold_predictions = [] # To store raw predictions for ROC curves

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', SVC(kernel='linear', probability=True, random_state=42))
    ])

    k_options = [10, 20, 30, 40, 50]
    if X.shape[1] < 50: 
        k_options = [5, 10, 15, 20, min(25, X.shape[1])]
        
    param_grid = {'feature_selection__k': k_options}

    for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X, y), total=n_splits_outer, desc="Nested K-fold")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        best_k = grid_search.best_params_['feature_selection__k']
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        fold_predictions.append({'y_true': y_test, 'y_prob': y_prob})

        selected_features_mask = best_model.named_steps['feature_selection'].get_support()
        selected_features = X.columns[selected_features_mask].tolist()

        results.append({
            'fold': fold + 1, 'best_k_found': best_k,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'auc': roc_auc_score(y_test, y_prob),
            'selected_features': selected_features
        })
        
    return pd.DataFrame(results), fold_predictions