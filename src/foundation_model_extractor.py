# src/foundation_model_extractor.py

import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm.auto import tqdm
import pickle
import numpy as np

def _safe_cuda_cleanup(*tensors):
    """
    Safely delete tensors and attempt to clear the CUDA cache.

    Iterate through a list of tensors, deleting each one. This function then
    attempts to empty the CUDA cache, ignoring any errors if the CUDA context
    has been compromised (e.g., by an out-of-memory error), which prevents
    the script from crashing during cleanup.

    Args:
        *tensors: A variable number of tensor objects to delete.
    """
    for tensor in tensors:
        if tensor is not None:
            del tensor
    # Check if a CUDA-enabled GPU is available.
    if torch.cuda.is_available():
        try:
            # Empty the PyTorch CUDA cache to free up unused memory.
            torch.cuda.empty_cache()
        except RuntimeError:
            # If the CUDA context is in an error state (e.g., from a previous OOM),
            # this call can fail. Ignore the error and proceed.
            pass

def extract_wav2vec2_sequences(
    input_df, 
    model_name="facebook/wav2vec2-base-960h", 
    audio_file_column='filepath', 
    chunk_seconds=5,
    overlap_seconds=1,
    verbose=True
):
    """
    Extract the full sequence of Wav2Vec2 embeddings for each audio file.

    This function uses a chunking strategy to process long audio files,
    preventing GPU out-of-memory errors. It slides a window across the
    audio, extracts embeddings for each chunk, and concatenates them to
    recreate a single sequence for the entire file.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        model_name (str): The name of the Hugging Face model to use.
        audio_file_column (str): The name of the column that holds the filepaths.
        chunk_seconds (int): The duration of each audio chunk in seconds.
        overlap_seconds (int): The duration of the overlap between chunks in seconds.
        verbose (bool): If True, print progress and warning messages.

    Returns:
        dict: A dictionary where keys are filenames (str) and values are the
              corresponding embedding sequences as NumPy arrays.
    """
    # Set the computation device and load the model from Hugging Face.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose: print(f"Using device: {device}")
        
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}"); return {}

    sequences_dict = {}
    sample_rate = 16000 # Define the model's required sample rate.
    
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Extracting Sequences")

    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)
        
        try:
            # Load and pre-process the audio waveform (mono, 16kHz).
            waveform, sr = torchaudio.load(filepath)
            if waveform.shape[1] < int(sample_rate * 0.5):
                if verbose: print(f"INFO: Skipping very short file '{filename}'.")
                continue
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)

            # Define the chunk and step sizes in samples.
            chunk_size = int(sample_rate * chunk_seconds)
            step_size = int(sample_rate * (chunk_seconds - overlap_seconds))
            
            all_chunk_embeddings = []

            # Iterate through the waveform using a sliding window.
            for i in range(0, waveform.shape[1], step_size):
                chunk = waveform[:, i:i+chunk_size]
                
                # Discard the final chunk if it's too short to be meaningful.
                if chunk.shape[1] < int(sample_rate * 0.5):
                    continue

                input_values, outputs, sequence_chunk = None, None, None
                try:
                    # Process the chunk and get the embeddings.
                    input_values = processor(chunk.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        outputs = model(input_values)
                    sequence_chunk = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                    all_chunk_embeddings.append(sequence_chunk)
                finally:
                    # Ensure memory is cleared after each chunk, even if an error occurs.
                    _safe_cuda_cleanup(input_values, outputs, sequence_chunk)

            # If any chunks were successfully processed, stack them into one long sequence.
            if all_chunk_embeddings:
                final_sequence = np.vstack(all_chunk_embeddings)
                sequences_dict[filename] = final_sequence
                
        except Exception as e:
            if verbose: print(f"FATAL ERROR processing file '{filename}': {e}. Skipping.")
            continue
            
    return sequences_dict

def extract_wav2vec2_embeddings(input_df, **kwargs):
    """
    Extract mean-pooled Wav2Vec2 embeddings for each audio file.

    This function serves as a wrapper around the `extract_wav2vec2_sequences`
    function. It first extracts the full temporal sequences and then calculates
    the mean across the time dimension to produce a single summary vector for
    each file. This is used to generate features for non-temporal models like SVMs.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        **kwargs: Arbitrary keyword arguments passed directly to the
                  `extract_wav2vec2_sequences` function.

    Returns:
        pd.DataFrame: A DataFrame containing the mean-pooled embeddings for
                      each successfully processed audio file.
    """
    # Call the sequence extractor to get the full embeddings.
    sequences_dict = extract_wav2vec2_sequences(input_df, **kwargs)
    if not sequences_dict:
        return pd.DataFrame()
        
    all_embeddings_list = []
    # Iterate through the returned sequences.
    for filename, sequence in sequences_dict.items():
        # Calculate the mean across the time dimension (axis 0).
        mean_embedding = np.mean(sequence, axis=0)
        # Create a dictionary for this file's features.
        embedding_dict = {f'dim_{k}': val for k, val in enumerate(mean_embedding)}
        embedding_dict['filename'] = filename
        all_embeddings_list.append(embedding_dict)
        
    return pd.DataFrame(all_embeddings_list)