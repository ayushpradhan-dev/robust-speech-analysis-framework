# src/foundation_model_extractor.py

import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm.auto import tqdm
import pickle
import numpy as np

def extract_wav2vec2_embeddings(input_df, model_name="facebook/wav2vec2-base-960h", audio_file_column='filepath', verbose=True):
    """
    Extract mean-pooled Wav2Vec2 embeddings for each audio file.

    Process each file individually for maximum stability, especially with audio
    clips of highly variable length. This function is used to generate summary
    feature vectors for classifiers like SVMs.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        model_name (str): The name of the Hugging Face model to use.
        audio_file_column (str): The name of the column in input_df that holds the filepaths.
        verbose (bool): If True, print progress and warning messages.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an audio file,
                      containing a 'filename' column and feature columns ('dim_0', 'dim_1', ...).
    """
    # Set the computation device to GPU if available, otherwise fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Using device: {device}")
        
    try:
        # Load the pre-trained model and its corresponding processor from Hugging Face.
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.eval() # Set the model to evaluation mode (disables dropout, etc.).
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}"); return pd.DataFrame()

    all_embeddings_list = []
    
    # Create an iterator with a progress bar if verbose mode is on.
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Extracting {os.path.basename(model_name)} Embeddings") if verbose else input_df.iterrows()

    # Process each audio file specified in the input DataFrame.
    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)
        
        try:
            # Load the audio waveform.
            waveform, sr = torchaudio.load(filepath)
            
            # Define a minimum duration threshold to filter out invalid/empty clips.
            min_duration_samples = int(16000 * 0.5) # 0.5 seconds
            if waveform.shape[1] < min_duration_samples:
                if verbose: print(f"INFO: Skipping very short file '{filename}'.")
                continue

            # --- Pre-processing Steps ---
            # 1. Convert to mono by averaging channels if necessary.
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            # 2. Resample to 16kHz, which is required by the Wav2Vec2 model.
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)

            # Use the processor to convert the raw waveform into the model's expected input format.
            input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
            
            # Perform a forward pass through the model without calculating gradients.
            with torch.no_grad():
                outputs = model(input_values)
            
            # --- Feature Aggregation ---
            # Take the output of the last hidden layer and apply mean pooling across the time dimension.
            # This creates a single summary vector for the entire audio clip.
            mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Convert the embedding vector into a dictionary of features.
            embedding_dict = {f'dim_{k}': val for k, val in enumerate(mean_embedding[0])}
            embedding_dict['filename'] = filename
            all_embeddings_list.append(embedding_dict)

            # Explicitly free up GPU memory after each file to prevent crashes with large datasets.
            del input_values, outputs, mean_embedding, waveform
            if device.type == 'cuda': torch.cuda.empty_cache()

        except Exception as e:
            if verbose: print(f"ERROR processing file '{filename}': {e}. Skipping.")
            continue

    if not all_embeddings_list and verbose:
        print("Warning: No embeddings were successfully extracted.")
        
    return pd.DataFrame(all_embeddings_list)


def extract_wav2vec2_sequences(input_df, model_name="facebook/wav2vec2-base-960h", audio_file_column='filepath', verbose=True):
    """
    Extracts the full sequence of Wav2Vec2 embeddings for each audio file.

    This function does not aggregate the embeddings. It returns a dictionary
    mapping each filename to its sequence of feature vectors, which is the
    required format for sequence models like LSTMs or Transformers.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        model_name (str): The name of the Hugging Face model to use.
        audio_file_column (str): The name of the column in input_df that holds the filepaths.
        verbose (bool): If True, print progress and warning messages.

    Returns:
        dict: A dictionary where keys are filenames (str) and values are the
              corresponding embedding sequences as NumPy arrays (shape: [time_steps, 768]).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Using device: {device}")
        
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}"); return {}

    sequences_dict = {}
    
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Extracting {os.path.basename(model_name)} Sequences") if verbose else input_df.iterrows()

    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)

        try:
            waveform, sr = torchaudio.load(filepath)

            # Apply the same strict duration filter for consistency.
            min_duration_samples = int(16000 * 0.5)
            if waveform.shape[1] < min_duration_samples:
                if verbose: print(f"INFO: Skipping very short file '{filename}'.")
                continue
            
            # Apply the same pre-processing (mono, 16kHz resample).
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
            
            with torch.no_grad():
                outputs = model(input_values)
            
            # Does not pool, instead gets the full sequence from the last hidden state.
            # Squeeze to remove the batch dimension, move to CPU, and convert to NumPy.
            sequence_embedding = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
            # Store the entire sequence array in the output dictionary.
            sequences_dict[filename] = sequence_embedding

            # Clean up memory after each file.
            del input_values, outputs, sequence_embedding, waveform
            if device.type == 'cuda': torch.cuda.empty_cache()
                
        except Exception as e:
            if verbose: print(f"ERROR processing file '{filename}': {e}. Skipping.")
            continue
            
    return sequences_dict