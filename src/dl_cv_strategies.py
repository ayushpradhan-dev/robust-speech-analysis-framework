# src/dl_cv_strategies.py

import os
import copy
import collections
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm

from .models import CNNLSTM

class SequenceDataset(Dataset):
    """
    Create a PyTorch Dataset for sequences of varying lengths.

    This class handles loading individual sequences and applying optional
    on-the-fly data augmentations during the training process.

    Args:
        sequences (list or np.ndarray): A list-like object where each element
                                        is a NumPy array representing a sequence
                                        of shape (sequence_length, feature_dim).
        labels (list or np.ndarray): A list-like object of corresponding integer labels.
        transform (callable, optional): An augmentation pipeline (e.g., from
                                        `audiomentations`) to be applied to the
                                        samples. Defaults to None.
        sample_rate (int, optional): The sample rate of the audio, required by
                                     certain augmentation libraries. Defaults to 16000.
    """
    def __init__(self, sequences, labels, transform=None, sample_rate=16000):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform
        self.sample_rate = sample_rate
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetch and process a single sample from the dataset.

        Applies augmentations if a transform is provided and converts the final
        NumPy array into a PyTorch tensor.
        """
        sequence_np = self.sequences[idx]
        
        # Apply augmentations only if a transform pipeline is passed (i.e., for the training set).
        if self.transform:
            sequence_np = self.transform(samples=sequence_np, sample_rate=self.sample_rate)
            
        # Ensure the final output is a float tensor and a tensor label.
        # Use .copy() to avoid negative stride issues with NumPy arrays after certain transforms.
        return torch.FloatTensor(sequence_np.copy()), torch.tensor(self.labels[idx])

def collate_fn(batch):
    """
    Pad sequences within a batch to a uniform length.

    This custom collate function is passed to the DataLoader. It takes a list of
    (sequence, label) tuples and pads each sequence to match the longest
    sequence in the batch, enabling efficient batch processing.

    Args:
        batch (list): A list of tuples, where each tuple contains a sequence
                      tensor and a label tensor.

    Returns:
        tuple: A tuple containing a padded sequence tensor of shape
               (batch_size, max_seq_len, feature_dim) and a label tensor of
               shape (batch_size,).
    """
    sequences, labels = zip(*batch)
    # Use pad_sequence to pad with zeros and stack into a single tensor.
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.stack(labels)


def _train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience):
    """
    Execute the main training and validation loop for a single model.

    This private helper function handles the epoch-based training, evaluates
    performance on a validation set after each epoch, implements early stopping,
    and saves the weights of the best performing model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_fn: The loss function (e.g., nn.CrossEntropyLoss).
        optimizer: The optimization algorithm (e.g., torch.optim.Adam).
        scheduler: The learning rate scheduler.
        device (torch.device): The device to train on ('cuda' or 'cpu').
        epochs (int): The maximum number of epochs to train for.
        patience (int): The number of epochs to wait for improvement before stopping.

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The trained model with the best weights loaded.
            - train_loss_history (list): A list of the average training loss per epoch.
            - val_loss_history (list): A list of the average validation loss per epoch.
    """
    histories = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None
    
    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()
        train_loss = 0
        for seq, lab in train_loader:
            seq, lab = seq.to(device), lab.to(device)
            optimizer.zero_grad()
            out = model(seq)
            loss = loss_fn(out, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        histories['train_loss'].append(train_loss / len(train_loader))
        
        # Set the model to evaluation mode to disable dropout for validation.
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, lab in val_loader:
                seq, lab = seq.to(device), lab.to(device)
                out = model(seq)
                loss = loss_fn(out, lab)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        histories['val_loss'].append(avg_val_loss)
        
        # Update the learning rate scheduler based on the validation loss.
        scheduler.step(avg_val_loss)
        
        # Implement the model checkpointing logic.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save a deep copy of the model's state dictionary.
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Check for the early stopping condition.
        if epochs_no_improve >= patience:
            print(f"  > Early stopping triggered at epoch {epoch + 1}")
            break
            
    # Load the best model weights found during training before returning.
    if best_model_weights:
        model.load_state_dict(best_model_weights)
        
    return model, histories['train_loss'], histories['val_loss']


def _eval_model(model, data_loader, device):
    """
    Evaluate a trained model on a given dataset.

    Args:
        model (nn.Module): The trained PyTorch model.
        data_loader (DataLoader): DataLoader for the dataset to be evaluated.
        device (torch.device): The device to perform evaluation on.

    Returns:
        tuple: A tuple of NumPy arrays containing:
            - The true labels.
            - The predicted labels.
            - The predicted probabilities for the positive class.
    """
    model.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for seq, lab in data_loader:
            seq, lab = seq.to(device), lab.to(device)
            out = model(seq)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())
            labels.extend(lab.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def _objective(trial, X_train_val, y_train_val, device, n_splits_inner):
    """
    Define the objective function for the Optuna hyperparameter search.

    This function defines the search space, creates a model with a suggested
    set of hyperparameters, trains it using an inner cross-validation loop,
    and returns a performance score for Optuna to maximize.

    Args:
        trial (optuna.Trial): An Optuna trial object.
        X_train_val (np.ndarray): The feature data for the outer training fold.
        y_train_val (np.ndarray): The labels for the outer training fold.
        device (torch.device): The device to train on.
        n_splits_inner (int): The number of folds for the inner CV.

    Returns:
        float: The mean F1-score across the inner cross-validation folds.
    """
    # Define the search space for Optuna to explore.
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'cnn_out_channels': trial.suggest_categorical('cnn_out_channels', [32, 64, 128]),
        'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [64, 128]),
        'activation_fn': trial.suggest_categorical('activation_fn', ['silu', 'gelu']),
    }
    
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)
    inner_f1_scores = []
    
    # Run a full cross-validation within the outer fold to evaluate this set of hyperparameters.
    for train_idx, val_idx in inner_cv.split(X_train_val, y_train_val):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        train_ds = SequenceDataset(X_train, y_train); val_ds = SequenceDataset(X_val, y_val)
        # Use a small batch size for the inner loop to manage memory.
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        model_params = {k:v for k,v in params.items() if k != 'learning_rate'}
        model = CNNLSTM(input_dim=X_train[0].shape[1], **model_params).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        
        # Train for a fixed, small number of epochs for speed during hyperparameter tuning.
        for epoch in range(15):
            model.train()
            for seq, lab in train_loader:
                seq, lab = seq.to(device), lab.to(device)
                optimizer.zero_grad(); out = model(seq); loss = loss_fn(out, lab); loss.backward(); optimizer.step()
        
        y_true_val, y_pred_val, _ = _eval_model(model, val_loader, device)
        inner_f1_scores.append(f1_score(y_true_val, y_pred_val, average='macro'))
        
    # Return the mean score, which Optuna will use to guide its search.
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
    Execute a full nested cross-validation with Optuna hyperparameter tuning.

    The outer loop provides an unbiased estimate of the final model performance.
    The inner loop, managed by Optuna, finds the best hyperparameters for each
    outer fold's training data.

    Args:
        sequences_dict (dict): Maps participant IDs to their sequence arrays.
        metadata_df (pd.DataFrame): Contains participant IDs and labels.
        n_splits_outer (int): Number of outer folds for evaluation.
        n_splits_inner (int): Number of inner folds for tuning.
        n_trials (int): Number of hyperparameter combinations for Optuna to test.
        epochs (int): Max epochs for training the final model in each outer fold.
        patience (int): Patience for early stopping in the final model training.
        batch_size (int): Batch size for the final model training.

    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): Performance metrics for each outer fold.
            - predictions (list): Raw predictions for ROC analysis.
            - weights (np.ndarray): Learned weights of the first CNN layer from
                                    each outer fold's best model, for stability analysis.
    """
    outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on device: {device} ---")
    
    results, fold_predictions, all_weights = [], [], []
    
    # Align the sequence data with the labels using the participant ID.
    label_map = metadata_df.set_index('unique_participant_id')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    common_participants = sorted(list(set(sequences_dict.keys()) & set(label_map.index)))
    X_data = np.array([sequences_dict[pid] for pid in common_participants], dtype=object)
    y_data = np.array(label_map.loc[common_participants].values)

    for fold, (train_val_idx, test_idx) in enumerate(tqdm(outer_cv.split(X_data, y_data), total=n_splits_outer, desc="Outer CV Fold")):
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]
        
        # Inner Loop: Run the Optuna hyperparameter search
        study = optuna.create_study(direction='maximize')
        objective_with_data = lambda trial: _objective(trial, X_train_val, y_train_val, device, n_splits_inner)
        study.optimize(objective_with_data, n_trials=n_trials, n_jobs=1) # Use n_jobs=1 for GPU safety.
        
        best_params = study.best_params
        
        # Final Training for this Fold
        # Create a train/validation split for robust early stopping.
        val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 80/20 split
        train_idx, val_idx = next(val_splitter.split(X_train_val, y_train_val))
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        train_ds = SequenceDataset(X_train, y_train); val_ds = SequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        model_params = {k:v for k,v in best_params.items() if k != 'learning_rate'}
        final_model = CNNLSTM(input_dim=X_train_val[0].shape[1], **model_params).to(device)
        
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # Train the final model for this fold using the best hyperparameters.
        best_model, _, _ = _train_eval_loop(final_model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience)
        
        # Extract weights from the trained model's first convolutional layer for stability analysis.
        first_layer_weights = best_model.res_block1.conv1.weight.data.cpu().numpy()
        all_weights.append(np.mean(np.abs(first_layer_weights), axis=0))
        
        # Evaluate the final model on the held-out outer test set.
        test_ds = SequenceDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
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
        
    return pd.DataFrame(results), fold_predictions, np.array(all_weights)


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
    Execute a standard k-fold cross-validation using a fixed set of hyperparameters.

    This function is used to create a direct comparison to the nested CV approach,
    demonstrating the performance of a pre-tuned model configuration.

    Args:
        sequences_dict (dict): Maps participant IDs to their sequence arrays.
        metadata_df (pd.DataFrame): Contains participant IDs and labels.
        hyperparams (dict): A dictionary of fixed hyperparameters for the model.
        n_splits (int): Number of folds for the cross-validation.
        epochs (int): Max epochs for training in each fold.
        patience (int): Patience for early stopping.
        batch_size (int): Batch size for training.

    Returns:
        tuple: A tuple containing:
            - results_df (pd.DataFrame): Performance metrics for each fold.
            - predictions (list): Raw predictions for ROC analysis.
            - histories (list): Training and validation loss histories for each fold.
            - weights (np.ndarray): Learned weights of the first CNN layer from
                                    each fold's model, for stability analysis.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results, fold_predictions, all_histories, all_weights = [], [], [], []
    
    label_map = metadata_df.set_index('unique_participant_id')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    common_participants = sorted(list(set(sequences_dict.keys()) & set(label_map.index)))
    X_data = np.array([sequences_dict[pid] for pid in common_participants], dtype=object)
    y_data = np.array(label_map.loc[common_participants].values)

    for fold, (train_idx, test_idx) in enumerate(tqdm(cv.split(X_data, y_data), total=n_splits, desc=f"Standard {n_splits}-Fold CV")):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # Create an inner train/validation split for early stopping.
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

        # Capture model weights for stability analysis.
        first_layer_weights = best_model.res_block1.conv1.weight.data.cpu().numpy()
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
        
    return pd.DataFrame(results), fold_predictions, all_histories, np.array(all_weights)