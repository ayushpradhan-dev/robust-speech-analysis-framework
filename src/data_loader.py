# src/data_loader.py

import os
import pandas as pd
import re

# Define a regular expression to capture metadata from the structured filenames.
# This pattern expects: ID_ConditionGenderAge_Education.wav
FILENAME_PATTERN = re.compile(r"(\d{1,2})_([PCX])([MF])(\d{2})_(\d)\.wav")

def _load_fold_maps(fold_list_csv_path):
    """
    Load and process the fold-lists.csv file to create lookup maps.

    Parse the provided CSV file which specifies which audio files belong to
    each of the 5 cross-validation folds for both the Reading and Interview tasks.

    Args:
        fold_list_csv_path (str): The full path to the fold-lists.csv file.

    Returns:
        tuple: A tuple containing two dictionaries:
               (read_task_file_to_fold_map, interview_task_file_to_fold_map).
               Return two empty dictionaries if the file is not found or an error occurs.
    """
    read_map = {}
    interview_map = {}
    
    try:
        # Load the CSV, treating the second row as the header.
        fold_df = pd.read_csv(fold_list_csv_path, header=1)
        
        # Define the expected column names for each task.
        read_cols = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
        interview_cols = ['fold1.1', 'fold2.1', 'fold3.1', 'fold4.1', 'fold5.1']

        # Iterate through the reading task columns.
        for col_name in read_cols:
            if col_name in fold_df:
                # Extract the fold number from the column name (e.g., 'fold1' -> 1).
                fold_num = int(re.search(r'(\d+)', col_name).group(1))
                # Create a mapping from the base filename to its assigned fold number.
                for fname in fold_df[col_name].dropna().astype(str):
                    # Clean the filename by removing extensions and quotes.
                    key = os.path.splitext(fname)[0].strip().strip("'")
                    read_map[key] = fold_num

        # Repeat the process for the interview task columns.
        for col_name in interview_cols:
            if col_name in fold_df:
                fold_num = int(re.search(r'(\d+)', col_name.split('.')[0]).group(1))
                for fname in fold_df[col_name].dropna().astype(str):
                    key = os.path.splitext(fname)[0].strip().strip("'")
                    interview_map[key] = fold_num
                        
        print(f"Successfully loaded {len(read_map)} Read task and {len(interview_map)} Interview task fold assignments.")
        
    except FileNotFoundError:
        print(f"ERROR: Fold list file not found at {fold_list_csv_path}")
    except Exception as e:
        print(f"An error occurred while processing {fold_list_csv_path}: {e}")
        
    return read_map, interview_map

def _parse_filename(filename_with_ext):
    """
    Parse a single Androids Corpus filename to extract metadata.

    Args:
        filename_with_ext (str): The filename (e.g., '01_CF56_1.wav').

    Returns:
        dict: A dictionary of metadata if the filename matches the pattern,
              otherwise return None.
    """
    match = FILENAME_PATTERN.match(filename_with_ext)
    if match:
        # Extract metadata components from the regex match groups.
        nn, cond_char, gen_char, age_s, edu_s = match.groups()
        metadata = {
            'unique_participant_id': f"{nn}_{cond_char}",
            'original_id_nn': nn,
            'label': "Patient" if cond_char == 'P' else "Control" if cond_char == 'C' else "Unknown",
            'gender': "Male" if gen_char == 'M' else "Female",
            'age': int(age_s),
            'education': int(edu_s)
        }
        return metadata
    return None

def load_androids_corpus(base_corpus_path, verbose=True):
    """
    Load the Androids Corpus data, process metadata, and assign fold numbers.

    Read the Reading Task audio files and the segmented Interview Task clips,
    creating two pandas DataFrames with structured information for each file.
    This serves as the main entry point for accessing the corpus data.

    Args:
        base_corpus_path (str): The relative or absolute path to the root 
                                of the Androids-Corpus directory.
        verbose (bool): If True, print status messages and warnings during processing.

    Returns:
        tuple: A tuple containing two pandas DataFrames: (reading_df, interview_df).
    """
    # Define paths to the data and metadata files.
    reading_task_root = os.path.join(base_corpus_path, 'Reading-Task', 'audio')
    interview_clips_root = os.path.join(base_corpus_path, 'Interview-Task', 'audio_clip')
    fold_list_csv_path = os.path.join(base_corpus_path, 'fold-lists.csv')

    # Create the filename-to-fold lookup dictionaries.
    read_fold_map, interview_fold_map = _load_fold_maps(fold_list_csv_path)

    # Process Reading Task
    reading_data = []
    if verbose:
        print(f"\nProcessing Reading Task from: {os.path.abspath(reading_task_root)}")
    
    # Iterate through the Healthy Control (HC) and Patient (PT) subfolders.
    for condition_folder in ['HC', 'PT']:
        condition_path = os.path.join(reading_task_root, condition_folder)
        if not os.path.isdir(condition_path):
            if verbose: print(f"Warning: Directory not found {condition_path}")
            continue
        
        for filename in os.listdir(condition_path):
            if filename.endswith('.wav'):
                metadata = _parse_filename(filename)
                if metadata:
                    filepath = os.path.join(condition_path, filename)
                    # Use the base filename (without extension) as the key for the fold map.
                    file_key = os.path.splitext(filename)[0]
                    fold_num = read_fold_map.get(file_key, -1)
                    
                    # Consolidate all information into the metadata dictionary.
                    metadata.update({
                        'filepath': filepath,
                        'filename': filename,
                        'task_type': 'Reading',
                        'fold': fold_num
                    })
                    reading_data.append(metadata)
                elif verbose and not filename.startswith('.'):
                    print(f"Warning: Could not parse filename '{filename}' in Reading-Task")
    
    reading_df = pd.DataFrame(reading_data)
    if verbose and not reading_df.empty:
        print(f"Processed {len(reading_df)} files from Reading-Task.")

    # Process Interview Task Clips
    interview_data = []
    if verbose:
        print(f"\nProcessing Interview Task clips from: {os.path.abspath(interview_clips_root)}")

    if not os.path.isdir(interview_clips_root):
        if verbose: print(f"Warning: Directory not found {interview_clips_root}")
    else:
        # Each subfolder in 'audio_clip' represents a full interview session.
        for session_folder in os.listdir(interview_clips_root):
            session_path = os.path.join(interview_clips_root, session_folder)
            if os.path.isdir(session_path):
                # Parse the session folder name to get the participant metadata.
                metadata = _parse_filename(session_folder + ".wav")
                if metadata:
                    # Get the fold number for the entire session.
                    fold_num = interview_fold_map.get(session_folder, -1)
                    # Iterate through each individual clip within the session folder.
                    for clip_filename in os.listdir(session_path):
                        if clip_filename.endswith('.wav'):
                            clip_filepath = os.path.join(session_path, clip_filename)
                            # Inherit the metadata from the parent session folder.
                            clip_metadata = metadata.copy()
                            clip_metadata.update({
                                'filepath': clip_filepath,
                                'filename': clip_filename,
                                'original_session_filename': session_folder,
                                'task_type': 'Interview_Clip',
                                'fold': fold_num
                            })
                            interview_data.append(clip_metadata)
                elif verbose and not session_folder.startswith('.'):
                    print(f"Warning: Could not parse interview session folder name: '{session_folder}'")
    
    interview_df = pd.DataFrame(interview_data)
    if verbose and not interview_df.empty:
        print(f"Processed {len(interview_df)} clip files from Interview-Task (audio_clip).")

    if verbose:
        print("\n--- Data Loading Complete ---")
        
    return reading_df, interview_df

# Define a main block to allow this script to be run directly for testing.
if __name__ == '__main__':
    # Determine the project root directory to construct an absolute path to the data.
    # This makes the test runnable from any location.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path = os.path.join(project_root, 'data', 'Androids_Corpus')

    print(f"Running data loader with base path: {base_path}")
    
    reading_df, interview_df = load_androids_corpus(base_path, verbose=True)

    # Display sample outputs and diagnostics if the script is run directly.
    if not reading_df.empty:
        print("\n--- Reading Task DataFrame Head ---")
        print(reading_df.head())
        print(f"\nFold distribution:\n{reading_df['fold'].value_counts().sort_index()}")
        
    if not interview_df.empty:
        print("\n--- Interview Task DataFrame Head ---")
        print(interview_df.head())
        print(f"\nFold distribution (by clips):\n{interview_df['fold'].value_counts().sort_index()}")