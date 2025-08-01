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

from .models import CNNLSTM

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, transform=None, sample_rate=16000):
        self.sequences, self.labels, self.transform, self.sample_rate = sequences, labels, transform, sample_rate
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        sequence_np = self.sequences[idx]
        if self.transform:
            sequence_np = self.transform(samples=sequence_np, sample_rate=self.sample_rate)
        return torch.FloatTensor(sequence_np.copy()), torch.tensor(self.labels[idx])

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0), torch.stack(labels)

def _train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience):
    histories = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            optimizer.zero_grad(); out = model(seq); loss = loss_fn(out, lab); loss.backward(); optimizer.step()
            train_loss += loss.item()
        histories['train_loss'].append(train_loss / len(train_loader))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, lab in val_loader:
                seq, lab = seq.to(device), lab.to(device)
                out = model(seq); loss = loss_fn(out, lab); val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        histories['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss; best_model_weights = copy.deepcopy(model.state_dict()); epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"  > Early stopping triggered at epoch {epoch + 1}")
            break
    if best_model_weights: model.load_state_dict(best_model_weights)
    return model, histories['train_loss'], histories['val_loss']

def _eval_model(model, data_loader, device):
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for seq, lab in data_loader:
            seq, lab = seq.to(device), lab.to(device)
            out = model(seq); prob = torch.softmax(out, dim=1)[:, 1]; pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy()); probs.extend(prob.cpu().numpy()); labels.extend(lab.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)

def _objective(trial, X_train_val, y_train_val, device, n_splits_inner):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'cnn_out_channels': trial.suggest_categorical('cnn_out_channels', [32, 64, 128]),
        'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [64, 128]),
        'activation_fn': trial.suggest_categorical('activation_fn', ['silu', 'gelu']),
    }
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    inner_f1_scores = []
    for train_idx, val_idx in inner_cv.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        train_ds = SequenceDataset(X_train, y_train); val_ds = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
        model = CNNLSTM(input_dim=X_train[0].shape[1], **{k:v for k,v in params.items() if k != 'learning_rate'}).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(15):
            model.train()
            for seq, lab in train_loader:
                seq, lab = seq.to(device), lab.to(device)
                optimizer.zero_grad(); out = model(seq); loss = loss_fn(out, lab); loss.backward(); optimizer.step()
        y_true, y_pred, _ = _eval_model(model, val_loader, device)
        inner_f1_scores.append(f1_score(y_true, y_pred, average='macro'))
    return np.mean(inner_f1_scores)

def run_pytorch_nested_cv_with_optuna(
    sequences_dict, 
    metadata_df,
    n_splits_outer=5,
    n_splits_inner=3,
    n_trials=20,
    epochs=100,
    patience=25, 
    batch_size=8
):
    """
    Performs nested k-fold CV with Optuna tuning.
    Now returns model weights for stability analysis.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device: {device} ---")
    
    results, fold_predictions, all_weights = [], [], []
    
    label_map = metadata_df.set_index('unique_participant_id')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    common_participants = sorted(list(set(sequences_dict.keys()) & set(label_map.index)))
    X_data = np.array([sequences_dict[pid] for pid in common_participants], dtype=object)
    y_data = np.array(label_map.loc[common_participants].values)

    for fold, (train_val_idx, test_idx) in enumerate(tqdm(outer_cv.split(X_data, y_data), total=n_splits_outer, desc="Outer CV Fold")):
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]
        
        study = optuna.create_study(direction='maximize')
        objective_with_data = lambda trial: _objective(trial, X_train_val, y_train_val, device, n_splits_inner)
        study.optimize(objective_with_data, n_trials=n_trials, n_jobs=1)
        best_params = study.best_params
        
        # Use a consistent 80/20 train/val split for the final training of this fold
        val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(val_splitter.split(X_train_val, y_train_val))
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        train_ds = SequenceDataset(X_train, y_train); val_ds = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model_params = {k:v for k,v in best_params.items() if k != 'learning_rate'}
        final_model = CNNLSTM(input_dim=X_train_val[0].shape[1], **model_params).to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        best_model, _, _ = _train_eval_loop(final_model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience)
        
        # Capture model weights for stability analysis
        first_layer_weights = best_model.res_block1.conv1.weight.data.cpu().numpy()
        feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
        all_weights.append(feature_importance)
        
        test_ds = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        y_true, y_pred, y_prob = _eval_model(best_model, test_loader, device)
        
        fold_predictions.append({'y_true': y_true, 'y_prob': y_prob})
        results.append({'fold': fold + 1, 'best_params': best_params, 'accuracy': accuracy_score(y_true, y_pred), 'f1_score': f1_score(y_true, y_pred, average='macro'), 'precision': precision_score(y_true, y_pred, average='macro', zero_division=0), 'recall': recall_score(y_true, y_pred, average='macro', zero_division=0), 'auc': roc_auc_score(y_true, y_prob)})
        
    return pd.DataFrame(results), fold_predictions, np.array(all_weights)


# Function for Standard K-Fold CV
def run_pytorch_standard_kfold_cv(
    sequences_dict, 
    metadata_df,
    hyperparams, 
    n_splits=5,
    epochs=100,
    patience=25,
    batch_size=8
):
    """
    Performs a STANDARD k-fold CV for a PyTorch model using a fixed set of hyperparameters.
    Now returns training histories AND model weights for stability analysis.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results, fold_predictions, all_histories, all_weights = [], [], [], []
    
    # data alignment
    label_map = metadata_df.set_index('unique_participant_id')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    common_participants = sorted(list(set(sequences_dict.keys()) & set(label_map.index)))
    X_data = np.array([sequences_dict[pid] for pid in common_participants], dtype=object)
    y_data = np.array(label_map.loc[common_participants].values)

    for fold, (train_idx, test_idx) in enumerate(tqdm(cv.split(X_data, y_data), total=n_splits, desc=f"Standard {n_splits}-Fold CV")):
        # the train/validation split and dataloader creation remains the same
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_split_idx, val_split_idx = next(val_splitter.split(X_train, y_train))
        X_train_inner, X_val_inner = X_train[train_split_idx], X_train[val_split_idx]
        y_train_inner, y_val_inner = y_train[train_split_idx], y_train[val_split_idx]
        train_ds = SequenceDataset(X_train_inner, y_train_inner); val_ds = SequenceDataset(X_val_inner, y_val_inner); test_ds = SequenceDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        model_params = {k:v for k,v in hyperparams.items() if k != 'learning_rate'}
        model = CNNLSTM(input_dim=X_train[0].shape[1], **model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        best_model, train_hist, val_hist = _train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience)
        all_histories.append({'train': train_hist, 'val': val_hist})

        # Capture model weights for stability analysis
        # Get the weights from the first convolutional layer of the best model
        first_layer_weights = best_model.res_block1.conv1.weight.data.cpu().numpy()
        # Calculate the mean absolute weight for each input dimension (768) across all output channels
        feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
        all_weights.append(feature_importance)

        y_true, y_pred, y_prob = _eval_model(best_model, test_loader, device)
        fold_predictions.append({'y_true': y_true, 'y_prob': y_prob})
        results.append({
            'fold': fold + 1, 'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'auc': roc_auc_score(y_true, y_prob)
        })
        
    # Return the histories and the weights
    return pd.DataFrame(results), fold_predictions, all_histories, np.array(all_weights)


# Function to Get Weights from a completed nested CV Run
def get_weights_from_tuned_results(
    sequences_dict, 
    metadata_df, 
    results_df, # The DataFrame from a completed tuned run
    n_splits=5,
    epochs=100, 
    patience=25,
    batch_size=8
):
    """
    Re-trains the best model for each fold from a completed Optuna run
    to extract the final model weights for stability analysis.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_weights = []
    
    label_map = metadata_df.set_index('unique_participant_id')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    common_participants = sorted(list(set(sequences_dict.keys()) & set(label_map.index)))
    X_data = np.array([sequences_dict[pid] for pid in common_participants], dtype=object)
    y_data = np.array(label_map.loc[common_participants].values)

    fold_iterator = tqdm(
        enumerate(outer_cv.split(X_data, y_data)), 
        total=n_splits, 
        desc="Re-training Folds to Get Weights"
    )

    for fold, (train_val_idx, test_idx) in fold_iterator:
        # Get the best hyperparameters that were already found for this fold
        # The 'fold' in the results_df is 1-based, so use fold + 1
        best_params = results_df[results_df['fold'] == fold + 1]['best_params'].iloc[0]
        
        X_train_val, _ = X_data[train_val_idx], X_data[test_idx]
        y_train_val, _ = y_data[train_val_idx], y_data[test_idx]

        val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(val_splitter.split(X_train_val, y_train_val))
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        train_ds = SequenceDataset(X_train, y_train); val_ds = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        model_params = {k:v for k,v in best_params.items() if k != 'learning_rate'}
        model = CNNLSTM(input_dim=X_train[0].shape[1], **model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # Re-train this single best model
        best_model, _, _ = _train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience)
        
        # Extract and save its weights
        first_layer_weights = best_model.res_block1.conv1.weight.data.cpu().numpy()
        feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
        all_weights.append(feature_importance)
        
    return np.array(all_weights)