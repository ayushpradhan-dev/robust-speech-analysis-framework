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
    Perform a standard k-fold cross-validation experiment for an SVM.

    In each fold, fit a scikit-learn pipeline consisting of a scaler,
    a feature selector with a fixed number of features, and a classifier
    on the training data. Then, evaluate the pipeline on the test data.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The label vector.
        n_splits (int, optional): The number of folds for cross-validation.
                                  Defaults to 5.
        n_features_to_select (int, optional): The fixed number of features for
                                              SelectKBest to select. Defaults to 50.

    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): A DataFrame with performance metrics
                                         and selected features for each fold.
            - predictions (list): A list of dictionaries, each holding the true
                                  labels and predicted probabilities for a fold,
                                  for later ROC analysis.
    """
    # Initialize the stratified k-fold splitter.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    fold_predictions = [] 

    # Iterate through each fold.
    for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Standard K-fold")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Define the full modeling pipeline.
        # This ensures that scaling and feature selection are only performed on the training data.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=n_features_to_select)),
            ('classifier', SVC(kernel='linear', probability=True, random_state=42))
        ])

        # Fit the entire pipeline on the training data for this fold.
        pipeline.fit(X_train, y_train)

        # Make predictions on the unseen test data for this fold.
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Store the raw predictions for later ROC curve generation.
        fold_predictions.append({'y_true': y_test, 'y_prob': y_prob})
        
        # Identify and store the names of the features selected in this fold.
        selected_features_mask = pipeline.named_steps['feature_selection'].get_support()
        selected_features = X.columns[selected_features_mask].tolist()

        # Compile all performance metrics for this fold.
        results.append({
            'fold': fold + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'auc': roc_auc_score(y_test, y_prob),
            'selected_features': selected_features
        })

    return pd.DataFrame(results), fold_predictions


def run_nested_kfold_cv(X, y, n_splits_outer=5, n_splits_inner=3):
    """
    Perform a nested k-fold cross-validation experiment for an SVM.

    The outer loop provides an unbiased estimate of the final model performance.
    The inner loop, managed by GridSearchCV, finds the best hyperparameter
    (number of features to select) for each outer fold's training data.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The label vector.
        n_splits_outer (int, optional): The number of outer folds for evaluation.
                                        Defaults to 5.
        n_splits_inner (int, optional): The number of inner folds for tuning.
                                        Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): A DataFrame with performance metrics,
                                         the best 'k' found, and selected
                                         features for each outer fold.
            - predictions (list): A list of dictionaries with raw predictions
                                  for ROC analysis.
    """
    # Initialize the outer and inner cross-validation splitters.
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    results = []
    fold_predictions = []

    # Define the pipeline to be tuned.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', SVC(kernel='linear', probability=True, random_state=42))
    ])

    # Define the search space for the number of features 'k'.
    # Adapt the search space for smaller feature sets.
    k_options = [10, 20, 30, 40, 50]
    if X.shape[1] < 50: 
        k_options = [5, 10, 15, 20, min(25, X.shape[1])]
        
    param_grid = {'feature_selection__k': k_options}

    # Iterate through each outer fold.
    for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X, y), total=n_splits_outer, desc="Nested K-fold")):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner Loop: Hyperparameter Tuning
        # Use GridSearchCV to automatically perform the inner cross-validation.
        # It searches for the best 'k' using only the outer training data (X_train, y_train).
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='f1_macro')
        grid_search.fit(X_train, y_train)

        # Retrieve the best hyperparameter found by the inner loop.
        best_k = grid_search.best_params_['feature_selection__k']
        # The .best_estimator_ is a pipeline already refit on the entire X_train using the best 'k'.
        best_model = grid_search.best_estimator_

        # Outer Loop: Performance Evaluation
        # Evaluate the best model on the held-out outer test set.
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        fold_predictions.append({'y_true': y_test, 'y_prob': y_prob})

        # Identify the features selected by the best model for this fold.
        selected_features_mask = best_model.named_steps['feature_selection'].get_support()
        selected_features = X.columns[selected_features_mask].tolist()

        # Compile all performance metrics for this outer fold.
        results.append({
            'fold': fold + 1,
            'best_k_found': best_k,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'auc': roc_auc_score(y_test, y_prob),
            'selected_features': selected_features
        })
        
    return pd.DataFrame(results), fold_predictions