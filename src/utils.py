# src/utils.py
import pandas as pd

def aggregate_clip_features(clip_features_df, metadata_df):
    """
    Aggregates clip-level features to the session level.

    Groups features by participant and calculates summary statistics (mean, std).

    Args:
        clip_features_df (pd.DataFrame): DataFrame containing features for each clip.
                                         Must have a 'filename' column for clip names.
        metadata_df (pd.DataFrame): The interview_df from the data loader, containing
                                    metadata for each clip.

    Returns:
        pd.DataFrame: A DataFrame with one row per participant session, containing
                      aggregated features.
    """
    # Merge with metadata to get the unique participant ID for each clip
    merged_df = pd.merge(metadata_df, clip_features_df, on='filename')
    
    # Define the columns to aggregate (all columns that are not metadata)
    feature_cols = [col for col in clip_features_df.columns if col != 'filename']
    
    # Group by the unique participant ID and aggregate
    agg_df = merged_df.groupby('unique_participant_id')[feature_cols].agg(['mean', 'std'])
    
    # Flatten the multi-level column names (e.g., 'mean_F0'/'mean' -> 'mean_F0_mean')
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    
    return agg_df