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

from .models import CNNLSTM

class SequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for sequences.
    """
    def __init__(self, sequences, labels, transform=None, sample_rate=16000):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform # Augmentation pipeline
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Start with the sequence as a NumPy array
        sequence_np = self.sequences[idx]
        
        # Apply augmentations (which operate on NumPy arrays)
        if self.transform:
            sequence_np = self.transform(samples=sequence_np, sample_rate=self.sample_rate)
            
        # Convert the (potentially augmented) NumPy array to a PyTorch tensor
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
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler with the validation loss 
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"  > Early stopping triggered at epoch {epoch + 1}")
            break
            
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model

# Model evaluation metrics
def _eval_model(model, data_loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Main Cross-Validation Orchestrator
def run_pytorch_cv_with_early_stopping(
    sequences_dict, 
    metadata_df, 
    n_splits=5,
    epochs=50, 
    patience=10,
    batch_size=8, 
    learning_rate=1e-4,
    augmentations=None # Add augmentation pipeline as an argument
):
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    fold_predictions = []
    
    labels = metadata_df.set_index('filename')['label'].apply(lambda x: 1 if x == 'Patient' else 0)
    filenames = list(sequences_dict.keys())
    
    aligned_sequences = [sequences_dict[fname] for fname in filenames]
    aligned_labels = labels.loc[filenames].values

    X_data = np.array(aligned_sequences, dtype=object)
    y_data = np.array(aligned_labels)

    outer_fold_iterator = tqdm(enumerate(outer_cv.split(X_data, y_data)), total=n_splits, desc=f"Running {n_splits}-Fold CV")

    for fold, (train_val_idx, test_idx) in outer_fold_iterator:
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]

        val_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(val_splitter.split(X_train_val, y_train_val))
        
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # Pass augmentations to the training dataset only
        train_dataset = SequenceDataset(X_train, y_train, transform=augmentations)
        val_dataset = SequenceDataset(X_val, y_val, transform=None) # No augmentation on validation set
        test_dataset = SequenceDataset(X_test, y_test, transform=None) # No augmentation on test set
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, persistent_workers=True)
        
        model = CNNLSTM(input_dim=X_train[0].shape[1]).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initialize the Learning Rate Scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        
        # Pass the scheduler to the training loop
        best_model = _train_eval_loop(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, patience)
            
        y_true, y_pred, y_prob = _eval_model(best_model, test_loader, device)
        
        fold_predictions.append({'y_true': y_true, 'y_prob': y_prob})
        
        results.append({
            'fold': fold + 1,
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='macro'),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'auc': roc_auc_score(y_true, y_prob)
        })

    return pd.DataFrame(results), fold_predictions