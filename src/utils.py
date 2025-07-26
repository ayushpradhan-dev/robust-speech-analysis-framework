# src/utils.py

import pandas as pd

def aggregate_clip_features(clip_features_df, metadata_df):
    """
    Aggregates clip-level features to the session level.

    Take a DataFrame of features extracted from individual audio clips and
    summarize them for each unique participant session by calculating the mean
    and standard deviation of each feature across all of a participant's clips.

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
    # Handle the edge case where no clip features were extracted.
    if clip_features_df.empty:
        print("Warning: Input clip_features_df is empty. Returning an empty aggregated DataFrame.")
        return pd.DataFrame()
    
    # To make the merge more memory-efficient, select only the necessary
    # columns from the metadata DataFrame.
    metadata_subset = metadata_df[['filename', 'unique_participant_id']]
    
    # Join the clip features with their corresponding participant ID.
    merged_df = pd.merge(metadata_subset, clip_features_df, on='filename')
    
    # The clip-level 'filename' is no longer needed after the merge, as all
    # subsequent operations are at the participant level.
    merged_df = merged_df.drop(columns=['filename'])
    
    # --- Core Aggregation Step ---
    # Group the DataFrame by the unique participant ID. For each feature column,
    # calculate both the mean and the standard deviation across all of a
    # participant's clips.
    # This results in a DataFrame with a multi-level column index, e.g.,
    # ('mean_F0', 'mean') and ('mean_F0', 'std').
    agg_df_multi_level = merged_df.groupby('unique_participant_id').agg(['mean', 'std'])
    
    # Flatten the multi-level column index into a single level for easier access.
    # e.g., ('mean_F0', 'mean') becomes 'mean_F0_mean'.
    agg_df_multi_level.columns = ['_'.join(col).strip() for col in agg_df_multi_level.columns.values]
    
    # The 'unique_participant_id' is currently the DataFrame index.
    # Convert it back into a regular column.
    final_df = agg_df_multi_level.reset_index()
    
    return final_df