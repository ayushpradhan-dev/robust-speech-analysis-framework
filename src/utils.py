# src/utils.py

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def aggregate_clip_features(clip_features_df, metadata_df):
    """
    Aggregate clip-level summary features to the session level.

    Take a DataFrame of features extracted from individual audio clips and
    summarize them for each unique participant session. Calculate the mean
    and standard deviation of each feature across all of a participant's clips.
    This prepares the data for classifiers that require a single feature vector
    per instance, such as SVMs.

    Args:
        clip_features_df (pd.DataFrame): DataFrame containing features for each
                                         individual clip. Must contain a 'filename'
                                         column to link with metadata.
        metadata_df (pd.DataFrame): The full interview_df from the data loader,
                                    containing the 'filename' and the
                                    'unique_participant_id' for each clip.

    Returns:
        pd.DataFrame: A DataFrame with one row per participant session, containing
                      the aggregated features. Columns are renamed to reflect the
                      aggregation (e.g., 'feature_mean', 'feature_std').
    """
    # Return immediately if the input DataFrame is empty to prevent errors.
    if clip_features_df.empty:
        print("Warning: Input clip_features_df is empty. Return an empty aggregated DataFrame.")
        return pd.DataFrame()
    
    # Select only the necessary columns from metadata to make the merge more memory-efficient.
    metadata_subset = metadata_df[['filename', 'unique_participant_id']]
    
    # Join the clip features with their corresponding participant ID.
    merged_df = pd.merge(metadata_subset, clip_features_df, on='filename')
    
    # Drop the clip-level 'filename' as it's no longer needed for aggregation.
    merged_df = merged_df.drop(columns=['filename'])
    
    # Core Aggregation Step
    # Group the DataFrame by the unique participant ID. For each feature column,
    # calculate both the mean and the standard deviation across all of a
    # participant's clips. This results in a DataFrame with a multi-level
    # column index, e.g., ('mean_F0', 'mean') and ('mean_F0', 'std').
    agg_df_multi_level = merged_df.groupby('unique_participant_id').agg(['mean', 'std'])
    
    # Flatten the multi-level column index into a single level for easier access.
    # e.g., ('mean_F0', 'mean') becomes 'mean_F0_mean'.
    agg_df_multi_level.columns = ['_'.join(col).strip() for col in agg_df_multi_level.columns.values]
    
    # Convert the 'unique_participant_id' from the DataFrame index back into a regular column.
    final_df = agg_df_multi_level.reset_index()
    
    return final_df


def aggregate_interview_sequences(clip_sequences, interview_metadata_df):
    """
    Aggregate clip-level sequences into a single sequence per session.

    Concatenate the individual feature sequences from each of a participant's
    audio clips into one long sequence. This prepares the data for sequential
    deep learning models like LSTMs or Transformers.

    Args:
        clip_sequences (dict): Dictionary mapping clip filenames to their
                               corresponding feature sequence (NumPy array).
        interview_metadata_df (pd.DataFrame): The full interview_df from the
                                              data loader, used to map clip
                                              filenames to participant IDs.

    Returns:
        dict: A dictionary where keys are participant IDs and values are the
              single, concatenated feature sequences as NumPy arrays.
    """
    # Group all clip filenames by their unique participant ID.
    participant_clips = interview_metadata_df.groupby('unique_participant_id')['filename'].apply(list)
    
    session_sequences = {}
    print("\nAggregating interview clips into single sequences per participant...")
    for participant_id, clip_filenames in tqdm(participant_clips.items(), desc="Aggregating Sequences"):
        
        # Collect all sequence arrays for the current participant.
        # Include a check to ensure the clip exists in the dictionary, which handles
        # cases where a clip failed feature extraction.
        participant_sequences = [clip_sequences[fname] for fname in clip_filenames if fname in clip_sequences]
        
        # Proceed only if there are valid sequences to process for the participant.
        if participant_sequences:
            # Vertically stack the sequence arrays to form one long sequence.
            # e.g., [[1,2],[3,4]] and [[5,6]] becomes [[1,2],[3,4],[5,6]].
            session_sequences[participant_id] = np.vstack(participant_sequences)
            
    return session_sequences