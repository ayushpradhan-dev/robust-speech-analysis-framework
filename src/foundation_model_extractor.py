import os
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm.auto import tqdm

def extract_wav2vec2_embeddings(input_df, model_name="facebook/wav2vec2-base-960h", audio_file_column='filepath', verbose=True):
    """
    Extracts mean-pooled wav2vec2 embeddings for all audio files in a DataFrame.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        model_name (str): The name of the Hugging Face model to use.
        audio_file_column (str): The name of the column in input_df that holds the filepaths.
        verbose (bool): If True, prints progress and warnings.

    Returns:
        pd.DataFrame: A new DataFrame containing the extracted embeddings for each file.
                      Includes a 'filename' column for merging.
    """
    # Setup device, model, and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_name}'. Please check the model name and your internet connection.")
        print(e)
        return pd.DataFrame()

    all_embeddings_list = []
    
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Extracting {os.path.basename(model_name)} Embeddings") if verbose else input_df.iterrows()

    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)
        embedding_dict = {'filename': filename}

        try:
            # Load and resample audio
            waveform, sr = torchaudio.load(filepath)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            # Process audio and move to device
            input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

            # Get embeddings
            with torch.no_grad():
                outputs = model(input_values)
            
            # Mean-pool the embeddings from the last hidden state
            mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Create feature dictionary
            for i, val in enumerate(mean_embedding):
                embedding_dict[f'dim_{i}'] = val
            
            all_embeddings_list.append(embedding_dict)

        except Exception as e:
            if verbose:
                print(f"ERROR processing file '{filename}': {e}. Skipping.")
            continue # Skip to the next file if an error occurs
            
    # Clear GPU memory after the loop
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if not all_embeddings_list:
        print("Warning: No embeddings were successfully extracted.")
        return pd.DataFrame()
        
    return pd.DataFrame(all_embeddings_list)



def extract_wav2vec2_sequences(input_df, model_name="facebook/wav2vec2-base-960h", audio_file_column='filepath', verbose=True):
    """
    Extracts the full sequence of wav2vec2 embeddings for each audio file.

    Instead of mean-pooling, this function saves the entire sequence of hidden
    states, which is required for sequence models like LSTMs or CNNs. The
    results are saved as a dictionary mapping filenames to numpy arrays.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        model_name (str): The name of the Hugging Face model to use.
        audio_file_column (str): The name of the column in input_df that holds the filepaths.
        verbose (bool): If True, prints progress and warnings.

    Returns:
        dict: A dictionary where keys are filenames and values are the
              corresponding embedding sequences as NumPy arrays (shape: [time_steps, 768]).
    """
    # Setup device, model, and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{model_name}'. Please check the model name and your internet connection.")
        print(e)
        return {}

    sequences_dict = {}
    
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Extracting {os.path.basename(model_name)} Sequences") if verbose else input_df.iterrows()

    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)

        try:
            # Load and resample audio
            waveform, sr = torchaudio.load(filepath)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            # Process audio and move to device
            input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

            # Get embeddings
            with torch.no_grad():
                outputs = model(input_values)
            
            # --- THIS IS THE KEY CHANGE ---
            # Do NOT mean-pool. Get the full sequence.
            # Squeeze to remove the batch dimension, move to CPU, and convert to NumPy.
            sequence_embedding = outputs.last_hidden_state.squeeze().cpu().numpy()
            
            # Store the sequence array in the dictionary with its filename as the key
            sequences_dict[filename] = sequence_embedding

        except Exception as e:
            if verbose:
                print(f"ERROR processing file '{filename}': {e}. Skipping.")
            continue
            
    # Clear GPU memory after the loop
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if not sequences_dict:
        print("Warning: No sequences were successfully extracted.")
        
    return sequences_dict