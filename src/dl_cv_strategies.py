# src/dl_cv_strategies.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm
import collections
import os
import copy
import optuna

# Import your CNNLSTM model
from .models import CNNLSTM

# Dataset and Collate Function 
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, transform=None, sample_rate=16000):
        self.sequences, self.labels, self.transform, self.sample_rate = sequences, labels, transform, sample_rate
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        sequence_np = self.sequences[idx]
        if self.transform:
            sequence_np = self.transform(samples=sequence_np, sample_rate=self.sample_rate)
        return torch.FloatTensor(sequence_np.copy()), torch.tensor(self.labels[idx])

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.stack(labels)

# Training and Evaluation Functions 
def _train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None
    for epoch in range(epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    return model

def _eval_model(model, data_loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy()); all_probs.extend(probs.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Optuna Objective Function for Inner Loop
def _objective(trial, X_train_val, y_train_val, device, n_splits_inner):
    """
    The objective function that Optuna tries to maximize, now with a more
    memory-efficient configuration.
    """
    # Define a more constrained hyperparameter search space to reduce model size
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'cnn_out_channels': trial.suggest_categorical('cnn_out_channels', [32, 64, 128]), 
        'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [64, 128]), 
        'activation_fn': trial.suggest_categorical('activation_fn', ['silu', 'gelu']),
    }
    
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    inner_f1_scores = []

    for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_val, y_train_val)):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        train_dataset = SequenceDataset(X_train, y_train)
        val_dataset = SequenceDataset(X_val, y_val)
        
        # Reduce the batch size for the inner tuning loop
        inner_batch_size = 4 
        train_loader = DataLoader(train_dataset, batch_size=inner_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=inner_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model = CNNLSTM(
            input_dim=X_train[0].shape[1],
            dropout_rate=params['dropout_rate'],
            cnn_out_channels=params['cnn_out_channels'],
            lstm_hidden_dim=params['lstm_hidden_dim'],
            activation_fn=params['activation_fn']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        
        # Train the model for a short period for this trial
        model.train()
        for epoch in range(15):
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                optimizer.zero_grad(); outputs = model(sequences); loss = loss_fn(outputs, labels); loss.backward(); optimizer.step()
        
        y_true_val, y_pred_val, _ = _eval_model(model, val_loader, device)
        inner_f1_scores.append(f1_score(y_true_val, y_pred_val, average='macro'))

    return np.mean(inner_f1_scores)

# Final Main Cross-Validation Orchestrator
def run_pytorch_nested_cv_with_optuna(
    sequences_dict, 
    metadata_df,
    n_splits_outer=5,
    n_splits_inner=3,
    n_trials=20, # Number of hyperparameter combinations to try
    epochs=50, 
    patience=10,
    batch_size=8
):
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device: {device} ---")
    
    results, fold_predictions = [], []
    label_map = metadata_df.set_index('unique_participant_id')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    common_participants = sorted(list(set(sequences_dict.keys()) & set(label_map.index)))
    X_data = np.array([sequences_dict[pid] for pid in common_participants], dtype=object)
    y_data = np.array(label_map.loc[common_participants].values)

    for fold, (train_val_idx, test_idx) in enumerate(tqdm(outer_cv.split(X_data, y_data), total=n_splits_outer, desc="Outer CV Fold")):
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]

        # Inner Loop: Hyperparameter Tuning with Optuna
        study = optuna.create_study(direction='maximize')
        objective_with_data = lambda trial: _objective(trial, X_train_val, y_train_val, device, n_splits_inner)
        study.optimize(objective_with_data, n_trials=n_trials, n_jobs=1) # n_jobs=1 is safer for GPU
        
        best_params = study.best_params
        
        # Final Training for this Fold
        # Create full training set (train + val)
        train_val_dataset = SequenceDataset(X_train_val, y_train_val)
        train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        
        # Build the final model with the best found hyperparameters
        final_model = CNNLSTM(
            input_dim=X_train_val[0].shape[1],
            dropout_rate=best_params['dropout_rate'],
            cnn_out_channels=best_params['cnn_out_channels'],
            lstm_hidden_dim=best_params['lstm_hidden_dim'],
            activation_fn=best_params['activation_fn']
        ).to(device)
        
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # Train the final model with early stopping (using a validation split is not strictly necessary here, but we can reuse the loop for simplicity)
        best_model = _train_eval_loop(final_model, train_val_loader, train_val_loader, loss_fn, optimizer, scheduler, device, epochs, patience) # Using train_val_loader for validation here is just for the early stopping mechanism
        
        # Final Evaluation for this Fold
        test_dataset = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        y_true, y_pred, y_prob = _eval_model(best_model, test_loader, device)
        
        fold_predictions.append({'y_true': y_true, 'y_prob': y_prob})
        results.append({
            'fold': fold + 1, 'best_params': best_params,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'auc': roc_auc_score(y_true, y_prob)
        })

    return pd.DataFrame(results), fold_predictions