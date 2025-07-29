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
    """Safely delete tensors and attempt to clear CUDA cache."""
    for tensor in tensors:
        if tensor is not None:
            del tensor
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            pass

def extract_wav2vec2_sequences(
    input_df, 
    model_name="facebook/wav2vec2-base-960h", 
    audio_file_column='filepath', 
    chunk_seconds=5,  # Process audio in 5-second chunks
    overlap_seconds=1, # Overlap chunks by 1 second
    verbose=True
):
    """
    Extracts the full sequence of wav2vec2 embeddings for each audio file,
    using a chunking strategy to handle long files and avoid GPU memory errors.
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
    sample_rate = 16000 # The model's required sample rate
    
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Extracting Sequences")

    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)
        
        try:
            waveform, sr = torchaudio.load(filepath)

            if waveform.shape[1] < int(sample_rate * 0.5):
                if verbose: print(f"INFO: Skipping very short file '{filename}'.")
                continue

            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)

            # Chunking Logic
            chunk_size = int(sample_rate * chunk_seconds)
            step_size = int(sample_rate * (chunk_seconds - overlap_seconds))
            
            all_chunk_embeddings = []

            # Slide a window across the waveform
            for i in range(0, waveform.shape[1], step_size):
                chunk = waveform[:, i:i+chunk_size]
                
                # If the last chunk is too small then skip it
                if chunk.shape[1] < int(sample_rate * 0.5):
                    continue

                input_values, outputs, sequence_chunk = None, None, None
                try:
                    input_values = processor(chunk.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
                    with torch.no_grad():
                        outputs = model(input_values)
                    sequence_chunk = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                    all_chunk_embeddings.append(sequence_chunk)
                finally:
                    _safe_cuda_cleanup(input_values, outputs, sequence_chunk)

            # If any chunks were processed, stack them into one long sequence
            if all_chunk_embeddings:
                final_sequence = np.vstack(all_chunk_embeddings)
                sequences_dict[filename] = final_sequence
                
        except Exception as e:
            if verbose: print(f"FATAL ERROR processing file '{filename}': {e}. Skipping.")
            continue
            
    return sequences_dict

# The summary embeddings function is a simple wrapper around the sequence extractor, 
# used to obtain the feature set for SVM.
def extract_wav2vec2_embeddings(input_df, **kwargs):
    """Extracts mean-pooled embeddings by first getting sequences."""
    
    sequences_dict = extract_wav2vec2_sequences(input_df, **kwargs)
    if not sequences_dict:
        return pd.DataFrame()
        
    all_embeddings_list = []
    for filename, sequence in sequences_dict.items():
        mean_embedding = np.mean(sequence, axis=0)
        embedding_dict = {f'dim_{k}': val for k, val in enumerate(mean_embedding)}
        embedding_dict['filename'] = filename
        all_embeddings_list.append(embedding_dict)
        
    return pd.DataFrame(all_embeddings_list)