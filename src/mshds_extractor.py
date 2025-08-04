# src/mshds_extractor.py

import os
import math
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
from tqdm.auto import tqdm

def _speechrate(snd):
    """
    Calculate speech rate and pausing-related features from a sound object.

    This function implements a complex algorithm based on de Jong & Wempe (2009)
    to identify syllable nuclei from intensity peaks and distinguish voiced
    syllables using pitch information. It provides a detailed analysis of the
    temporal dynamics of the utterance.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object to be analyzed.

    Returns:
        tuple: A tuple containing the following features:
               (Speaking_Rate, Articulation_Rate, Phonation_Ratio,
                Pause_Rate, Mean_Pause_Dur). Returns NaNs on failure.
    """
    try:
        # Define key hyperparameters for pause and syllable detection.
        silencedb = -25
        mindip = 2
        minpause = 0.3
        
        # Adapt syllable detection sensitivity based on voice quality.
        # Use a less sensitive dip threshold for noisier signals (lower HNR).
        hnr = call(snd.to_harmonicity_cc(), "Get mean", 0, 0)
        if hnr < 60:
            mindip = 2

        # Create an intensity object for analysis.
        intensity = snd.to_intensity(minimum_pitch=50, time_step=0.016, subtract_mean=True)
        min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

        # Estimate the silence threshold relative to the 99th quantile of intensity.
        # This makes the threshold robust to short, loud, non-speech bursts.
        max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)
        silencedb_1 = max_99_intensity + silencedb
        if silencedb_1 < min_intensity:
            silencedb_1 = min_intensity
        db_adjustment = max_intensity - max_99_intensity
        silencedb_2 = silencedb - db_adjustment

        # Generate a TextGrid of silent and sounding intervals based on the threshold.
        textgrid = call(intensity, "To TextGrid (silences)", silencedb_2, minpause, 0.1, "silent", "sounding")
        
        # Extract the sounding intervals to calculate total phonation time.
        silencetier = call(textgrid, "Extract tier", 1)
        silencetable = call(silencetier, "Down to TableOfReal", "sounding")
        npauses = call(silencetable, "Get number of rows")
        
        Phonation_Time = 0
        if npauses == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
            
        for ipause in range(1, npauses + 1):
            beginsound = call(silencetable, "Get value", ipause, 1)
            endsound = call(silencetable, "Get value", ipause, 2)
            Phonation_Time += (endsound - beginsound)
            if ipause == 1:
                begin_speak = beginsound
            if ipause == npauses:
                end_speak = endsound

        # Identify syllable nuclei by finding intensity peaks.
        intensity_matrix = call(intensity, "Down to Matrix")
        sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
        point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
        
        numpeaks = call(point_process, "Get number of points")
        t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]
        
        timepeaks, intensities = [], []
        for i in range(numpeaks):
            value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
            if value > silencedb_1:
                intensities.append(value)
                timepeaks.append(t[i])

        # Filter peaks to count only those preceded by a significant intensity dip.
        validtime = []
        if len(timepeaks) > 1:
            currenttime = timepeaks[0]
            currentint = intensities[0]
            for p in range(len(timepeaks) - 1):
                following = p + 1
                dip = call(intensity, "Get minimum", currenttime, timepeaks[following], "None")
                if abs(currentint - dip) > mindip:
                    validtime.append(timepeaks[p])
                currenttime = timepeaks[following]
                currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")
        
        # Count only the valid syllable peaks that are voiced.
        pitch = snd.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
        Number_Syllables = 0
        for time in validtime:
            whichInterval = call(textgrid, "Get interval at time", 1, time)
            whichlabel = call(textgrid, "Get label of interval", 1, whichInterval)
            value = pitch.get_value_at_time(time)
            if not np.isnan(value) and whichlabel == "sounding":
                Number_Syllables += 1
        
        Original_Dur = end_speak - begin_speak
        Speaking_Rate = Number_Syllables / Original_Dur if Original_Dur > 0 else 0
        Articulation_Rate = Number_Syllables / Phonation_Time if Phonation_Time > 0 else 0
        Phonation_Ratio = Phonation_Time / Original_Dur if Original_Dur > 0 else 0
        Number_Pauses = npauses - 1
        Pause_Time = Original_Dur - Phonation_Time
        Pause_Rate = Number_Pauses / Original_Dur if Original_Dur > 0 else 0
        Mean_Pause_Dur = Pause_Time / Number_Pauses if Number_Pauses > 0 else 0
        
        return Speaking_Rate, Articulation_Rate, Phonation_Ratio, Pause_Rate, Mean_Pause_Dur

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def _pitch_values(snd):
    """
    Estimate a speaker-specific pitch floor and ceiling for robust analysis.

    Perform a wide pitch search first, then analyze the distribution to
    determine if the speaker is likely in a male or female pitch range. This
    prevents errors in subsequent, more focused pitch analyses.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.

    Returns:
        tuple: A tuple containing (pitch_floor, pitch_ceiling) in Hertz.
    """
    try:
        # Perform a wide, exploratory pitch analysis.
        pitch_wide = snd.to_pitch_ac(time_step=0.005, pitch_floor=50, pitch_ceiling=600)
        pitch_values_arr = pitch_wide.selected_array['frequency']
        pitch_values_arr = pitch_values_arr[pitch_values_arr != 0]
        if len(pitch_values_arr) == 0: return 75, 500 # Return default if no pitch is found.
        
        # Filter out outliers (beyond 2 standard deviations) to get a stable mean.
        pitch_values_z = (pitch_values_arr - np.mean(pitch_values_arr)) / np.std(pitch_values_arr)
        pitch_values_filtered = pitch_values_arr[abs(pitch_values_z) <= 2]
        if len(pitch_values_filtered) == 0: return 75, 500
        
        mean_pitch = np.mean(pitch_values_filtered)
        
        # Set a tighter, more appropriate search range based on the estimated mean pitch.
        if mean_pitch < 170:
            pitch_floor, pitch_ceiling = 60, 250 # Male range
        else:
            pitch_floor, pitch_ceiling = 100, 500 # Female range
        return pitch_floor, pitch_ceiling
    except Exception:
        return 75, 500 # Fallback to a generic default range on error.

def _extract_pitch(snd, floor, ceiling, frame_shift):
    """
    Extract fundamental frequency (F0) features.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The lower bound of the pitch search range (Hz).
        ceiling (int): The upper bound of the pitch search range (Hz).
        frame_shift (float): The time step for analysis (s).

    Returns:
        tuple: A tuple containing (mean_F0_Hz, stdev_F0_Semitones).
    """
    try:
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)
        mean_F0 = call(pitch, "Get mean", 0, 0, "Hertz")
        stdev_F0_Semitone = call(pitch, "Get standard deviation", 0, 0, "semitones")
        return mean_F0, stdev_F0_Semitone
    except Exception:
        return np.nan, np.nan

def _extract_intensity(snd, floor, frame_shift):
    """
    Extract intensity (energy) features.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The pitch floor, used by Praat to focus the intensity analysis.
        frame_shift (float): The time step for analysis (s).

    Returns:
        tuple: A tuple containing (mean_intensity_dB, intensity_range_ratio).
    """
    try:
        intensity = snd.to_intensity(minimum_pitch=floor, time_step=frame_shift, subtract_mean=True)
        mean_dB = call(intensity, "Get mean", 0, 0, "energy")
        min_dB = call(intensity, "Get minimum", 0, 0, "parabolic")
        max_dB = call(intensity, "Get maximum", 0, 0, "parabolic")
        range_dB_Ratio = max_dB / min_dB if min_dB != 0 else np.nan
        return mean_dB, range_dB_Ratio
    except Exception:
        return np.nan, np.nan

def _extract_harmonicity(snd, floor, ceiling, frame_shift):
    """
    Extract the mean Harmonics-to-Noise Ratio (HNR).

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The lower bound of the pitch search range (Hz).
        ceiling (int): The upper bound of the pitch search range (Hz).
        frame_shift (float): The time step for analysis (s).

    Returns:
        float: The mean HNR in decibels (dB).
    """
    try:
        harmonicity = snd.to_harmonicity_cc(time_step=frame_shift, minimum_pitch=floor, silence_threshold=0.1, periods_per_window=4.5)
        HNR_DB = call(harmonicity, "Get mean", 0, 0)
        return HNR_DB
    except Exception:
        return np.nan

def _extract_Slope_Tilt(snd, floor, ceiling):
    """
    Extract features from the Long-Term Average Spectrum (LTAS).

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The lower bound of the pitch search range (Hz).
        ceiling (int): The upper bound of the pitch search range (Hz).

    Returns:
        tuple: A tuple containing (spectral_slope, spectral_tilt).
    """
    try:
        # Create a pitch-corrected LTAS to remove F0 influence.
        ltsas_rep = call(snd, "To Ltas (pitch-corrected)...", floor, ceiling, 5000, 100, 0.0001, 0.02, 1.3)
        spc_Slope = call(ltsas_rep, "Get slope", 50, 1000, 1000, 4000, "dB")
        
        # Parse the text report to extract the spectral tilt value.
        spc_Tilt_Report = call(ltsas_rep, "Report spectral tilt", 100, 5000, "Linear", "Robust")
        srt_ST = spc_Tilt_Report.index("Slope: ") + len("Slope: ")
        end_ST = spc_Tilt_Report.index("d", srt_ST)
        spc_Tilt = float(spc_Tilt_Report[srt_ST:end_ST])
        return spc_Slope, spc_Tilt
    except Exception:
        return np.nan, np.nan

def _extract_CPP(snd, floor, ceiling, frame_shift):
    """
    Extract the mean Cepstral Peak Prominence (CPP).

    Analyzes only the voiced segments of the audio to get a robust measure of
    voice periodicity.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The lower bound of the pitch search range (Hz).
        ceiling (int): The upper bound of the pitch search range (Hz).
        frame_shift (float): The time step for analysis (s).

    Returns:
        float: The mean CPP value.
    """
    try:
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling, voicing_threshold=0.3)
        pulses = call([snd, pitch], "To PointProcess (cc)")
        textgrid = call(pulses, "To TextGrid (vuv)", 0.02, 0.1)
        vuv_table = call(textgrid, "Down to Table", 'no', 6, 'yes', 'no')
        
        n_intervals = call(vuv_table, "Get number of rows")
        CPP_list = []
        for i in range(1, n_intervals + 1):
            label = call(vuv_table, "Get value", i, 'text')
            if label == 'V':
                tmin = float(call(vuv_table, "Get value", i, 'tmin'))
                tmax = float(call(vuv_table, "Get value", i, 'tmax'))
                
                # Guard against zero-duration segments.
                if tmin >= tmax: continue
                
                snd_segment = snd.extract_part(from_time=tmin, to_time=tmax, preserve_times=False)
                
                if snd_segment.get_total_duration() > 0:
                    PowerCepstrogram = call(snd_segment, 'To PowerCepstrogram', 60, 0.002, 5000, 50)
                    try:
                        CPP_Value = call(PowerCepstrogram, 'Get CPPS...', "no", 0.01, 0.001, 60, 330, 0.05, "parabolic", 0.001, 0, "Straight", "Robust")
                        # Filter out low or invalid CPP values.
                        if not np.isnan(CPP_Value) and CPP_Value > 4:
                            CPP_list.append(CPP_Value)
                    except Exception:
                        continue # Skip segment if CPPS calculation fails.
        
        return np.mean(CPP_list) if CPP_list else np.nan

    except Exception:
        return np.nan

def _measureFormants(snd, floor, ceiling, frame_shift):
    """
    Extract summary statistics for the first two formants.

    Analyzes formants only at glottal pulse locations for higher accuracy.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The lower bound of the pitch search range (Hz).
        ceiling (int): The upper bound of the pitch search range (Hz).
        frame_shift (float): The time step for analysis (s).

    Returns:
        tuple: A tuple of 8 formant features (F1/B1/F2/B2 mean/std).
    """
    try:
        formants = call(snd, "To Formant (burg)", frame_shift, 5, 5000, 0.025, 50)
        pitch = snd.to_pitch_cc(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)
        pulses = call([snd, pitch], "To PointProcess (cc)")
        
        numPoints = call(pulses, "Get number of points")
        F1_list, B1_list, F2_list, B2_list = [], [], [], []

        for point in range(1, numPoints + 1):
            t = call(pulses, "Get time from index", point)
            if not np.isnan(v := call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')): F1_list.append(v)
            if not np.isnan(v := call(formants, "Get bandwidth at time", 1, t, 'Hertz', 'Linear')): B1_list.append(v)
            if not np.isnan(v := call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')): F2_list.append(v)
            if not np.isnan(v := call(formants, "Get bandwidth at time", 2, t, 'Hertz', 'Linear')): B2_list.append(v)

        return (np.mean(F1_list) if F1_list else np.nan, np.std(F1_list, ddof=1) if len(F1_list) > 1 else np.nan,
                np.mean(B1_list) if B1_list else np.nan, np.std(B1_list, ddof=1) if len(B1_list) > 1 else np.nan,
                np.mean(F2_list) if F2_list else np.nan, np.std(F2_list, ddof=1) if len(F2_list) > 1 else np.nan,
                np.mean(B2_list) if B2_list else np.nan, np.std(B2_list, ddof=1) if len(B2_list) > 1 else np.nan)
    except Exception:
        return (np.nan,) * 8

def _extract_Spectral_Moments(snd, floor, ceiling, window_size, frame_shift):
    """
    Extract the first four spectral moments from voiced segments.

    Args:
        snd (parselmouth.Sound): A Parselmouth Sound object.
        floor (int): The lower bound of the pitch search range (Hz).
        ceiling (int): The upper bound of the pitch search range (Hz).
        window_size (float): The length of the spectrogram window (s).
        frame_shift (float): The time step for analysis (s).

    Returns:
        tuple: A tuple containing the four spectral moments (Gravity, StdDev, Skewness, Kurtosis).
    """
    try:
        pitch = snd.to_pitch_ac(time_step=frame_shift, pitch_floor=floor, pitch_ceiling=ceiling)
        spectrogram = snd.to_spectrogram(window_length=window_size, time_step=frame_shift)
        
        Gravity_list, STD_list, Skew_list, Kurt_list = [], [], [], []
        
        num_steps = call(spectrogram, 'Get number of frames')
        for i in range(1, num_steps + 1):
            t = call(spectrogram, 'Get time from frame number', i)
            # Ensure analysis is only on voiced frames.
            if not np.isnan(pitch.get_value_at_time(t)):
                spectrum_slice = spectrogram.to_spectrum_slice(t)
                if not np.isnan(v := spectrum_slice.get_centre_of_gravity(power=2)): Gravity_list.append(v)
                if not np.isnan(v := spectrum_slice.get_standard_deviation(power=2)): STD_list.append(v)
                if not np.isnan(v := spectrum_slice.get_skewness(power=2)): Skew_list.append(v)
                if not np.isnan(v := spectrum_slice.get_kurtosis(power=2)): Kurt_list.append(v)
        
        return (np.mean(Gravity_list) if Gravity_list else np.nan,
                np.mean(STD_list) if STD_list else np.nan,
                np.mean(Skew_list) if Skew_list else np.nan,
                np.mean(Kurt_list) if Kurt_list else np.nan)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def extract_mshds_features(input_df, audio_file_column='filepath', verbose=True):
    """
    Extract a curated set of MSHDS features for all audio files in a DataFrame.

    This function serves as the main orchestrator, processing each audio file
    through a series of robust, speaker-adapted acoustic analyses.

    Args:
        input_df (pd.DataFrame): DataFrame containing filepaths to audio files.
        audio_file_column (str): The name of the column that holds the filepaths.
        verbose (bool): If True, print progress and warning messages.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an audio file,
                      containing a 'filename' column and all 25 MSHDS features.
    """
    all_features_list = []
    # Define the canonical list of feature names for consistent output.
    feature_names = [
        'Speaking_Rate', 'Articulation_Rate', 'Phonation_Ratio', 'Pause_Rate', 'Mean_Pause_Duration',
        'mean_F0', 'stdev_F0_Semitone', 'mean_dB', 'range_ratio_dB', 'HNR_dB',
        'Spectral_Slope', 'Spectral_Tilt', 'Cepstral_Peak_Prominence',
        'mean_F1_Loc', 'std_F1_Loc', 'mean_B1_Loc', 'std_B1_Loc',
        'mean_F2_Loc', 'std_F2_Loc', 'mean_B2_Loc', 'std_B2_Loc',
        'Spectral_Gravity', 'Spectral_Std_Dev', 'Spectral_Skewness', 'Spectral_Kurtosis'
    ]
    
    iterator = tqdm(input_df.iterrows(), total=input_df.shape[0], desc="Extracting MSHDS Features", disable=not verbose)

    for index, row in iterator:
        filepath = row[audio_file_column]
        filename = os.path.basename(filepath)
        feature_dict = {'filename': filename}

        try:
            # Pre-process the audio: load, convert to mono, and resample to 16kHz.
            snd = parselmouth.Sound(filepath)
            if snd.get_number_of_channels() > 1:
                snd = snd.convert_to_mono()
            if snd.get_sampling_frequency() != 16000.0:
                snd = snd.resample(16000, 50)
            
            # Extract feature groups by calling the internal helper functions.
            (
                feature_dict['Speaking_Rate'], feature_dict['Articulation_Rate'], 
                feature_dict['Phonation_Ratio'], feature_dict['Pause_Rate'], 
                feature_dict['Mean_Pause_Duration']
            ) = _speechrate(snd)

            pitch_floor, pitch_ceiling = _pitch_values(snd)
            
            feature_dict['mean_F0'], feature_dict['stdev_F0_Semitone'] = _extract_pitch(snd, pitch_floor, pitch_ceiling, 0.005)
            feature_dict['mean_dB'], feature_dict['range_ratio_dB'] = _extract_intensity(snd, pitch_floor, 0.005)
            feature_dict['HNR_dB'] = _extract_harmonicity(snd, pitch_floor, pitch_ceiling, 0.005)
            feature_dict['Spectral_Slope'], feature_dict['Spectral_Tilt'] = _extract_Slope_Tilt(snd, pitch_floor, pitch_ceiling)
            feature_dict['Cepstral_Peak_Prominence'] = _extract_CPP(snd, pitch_floor, pitch_ceiling, 0.005)
            
            (
                feature_dict['mean_F1_Loc'], feature_dict['std_F1_Loc'], 
                feature_dict['mean_B1_Loc'], feature_dict['std_B1_Loc'],
                feature_dict['mean_F2_Loc'], feature_dict['std_F2_Loc'], 
                feature_dict['mean_B2_Loc'], feature_dict['std_B2_Loc']
            ) = _measureFormants(snd, pitch_floor, pitch_ceiling, 0.005)
            
            (
                feature_dict['Spectral_Gravity'], feature_dict['Spectral_Std_Dev'], 
                feature_dict['Spectral_Skewness'], feature_dict['Spectral_Kurtosis']
            ) = _extract_Spectral_Moments(snd, pitch_floor, pitch_ceiling, 0.025, 0.005)
            
            all_features_list.append(feature_dict)

        except Exception as e:
            # On failure, append a row of NaNs to maintain DataFrame integrity.
            if verbose:
                print(f"ERROR processing file '{filename}': {e}. Appending NaNs.")
            error_dict = {'filename': filename}
            for feat_name in feature_names:
                error_dict[feat_name] = np.nan
            all_features_list.append(error_dict)
            
    return pd.DataFrame(all_features_list)