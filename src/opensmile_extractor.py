import os
import subprocess
import tempfile
import pandas as pd
from tqdm.auto import tqdm

def extract_opensmile_features(
    input_df, 
    opensmile_exe_path, 
    config_file_path, 
    audio_file_column='filepath', 
    verbose=True
):
    """
    Extracts OpenSMILE features using a corrected configuration file that
    produces a standard CSV output with a header.
    """
    all_features_list = []
    
    if not os.path.exists(opensmile_exe_path):
        print(f"FATAL ERROR: Could not find SMILExtract at '{opensmile_exe_path}'.")
        return pd.DataFrame()
        
    with tempfile.TemporaryDirectory() as temp_dir:
        if verbose:
            print(f"Using temporary directory for OpenSMILE outputs: {temp_dir}")
        
        iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Extracting OpenSMILE Features") if verbose else input_df.iterrows()

        for index, row in iterator:
            input_audio_path = row[audio_file_column]
            filename = os.path.basename(input_audio_path)
            
            output_csv_path = os.path.join(temp_dir, f"{os.path.splitext(filename)[0]}.csv")
            
            command = [
                opensmile_exe_path,
                '-C', config_file_path,
                '-I', input_audio_path,
                '-O', output_csv_path,
                # The -instname argument helps give the first column a consistent name
                '-instname', filename 
            ]
            
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                
                # Read the standard CSV file. Use the default separator ','
                # because the corrected config file should produce a standard CSV.
                features_df = pd.read_csv(output_csv_path, sep=',')
                
                # Instead of dropping the first column by a hardcoded name, select all
                # columns from the second column onwards.
                feature_dict = features_df.iloc[:, 1:].iloc[0].to_dict()
                
                feature_dict['filename'] = filename 
                all_features_list.append(feature_dict)
                
            except subprocess.CalledProcessError as e:
                if verbose:
                    print(f"ERROR: OpenSMILE failed for file '{filename}'. Stderr: {e.stderr}")
            except Exception as e:
                if verbose:
                    print(f"An unexpected error occurred while processing output for '{filename}': {e}")
    
    if not all_features_list:
        print("Warning: No features were successfully extracted. The returned DataFrame is empty.")
        return pd.DataFrame()
        
    return pd.DataFrame(all_features_list)