## MSc Dissertation: A Methodological Framework for Evaluating Speech-Based Machine Learning Models for Remote Health Assessment
> A full description of the project can be found in the research paper:
> **[A Framework for Evaluating Speech-Based ML Models for Remote Health Assessment — Ayush Pradhan](https://ayushpradhan-dev.github.io/robust-speech-analysis-framework/Speech-ML-Health-Assessment-Ayush-Pradhan.pdf)**

This repository contains the code for my MSc Data Science dissertation project at King's College London. The project develops and applies a comprehensive evaluation framework to critically investigate the impact of different cross-validation strategies and acoustic feature representations on the performance and stability of machine learning models for depression detection from speech.

The framework is implemented in Python and leverages libraries such as scikit-learn, PyTorch, and Optuna. It provides a full pipeline from data processing to model evaluation and visualization.

---

### Key Methodological Comparisons
This project systematically compares:
*   **Three Feature Approaches:**
    1.  **MSHDS:** A curated, handcrafted set of 25 acoustic features.
    2.  **OpenSMILE:** A large-scale, comprehensive handcrafted set of 911 features.
    3.  **Wav2Vec2:** Learned, sequential embeddings from a state-of-the-art foundation model.
*   **Two Classifier Architectures:**
    1.  **SVM:** A linear Support Vector Machine for baseline evaluation on summary-statistic features.
    2.  **CNN-LSTM:** A deep learning model with residual connections and attention pooling for evaluating sequential features.
*   **Two Cross-Validation Strategies:**
    1.  **Standard K-Fold CV:** A baseline evaluation method.
    2.  **Nested K-Fold CV:** A more robust method incorporating hyperparameter tuning to provide an unbiased performance estimate.

### Project Focus: A Unimodal Acoustic Approach
This project focuses exclusively on the **acoustic modality** of speech. Textual features from transcripts (NLP) and visual features from video were deliberately excluded. This unimodal approach was chosen for a critical methodological reason: to isolate and rigorously evaluate the predictive power and stability of vocal biomarkers alone.

The resulting feature extraction pipelines and trained models from this work can serve as a powerful acoustic component in future, more complex **multimodal systems** that also incorporate textual and visual analysis.

---

### Repository Structure

The project is organized into the following directories:

*   **/data/Processed\_Features/:** Contains generated feature files (`.csv`) and experiment results (`.pkl`). The smaller result files and summary-statistic feature files are tracked using **Git LFS** for convenience. The very large sequence embedding files (`*_sequences_*.pkl`) are excluded by `.gitignore` and must be generated or downloaded separately (see Usage section).
*   **/models/:** Contains the final, trained PyTorch model artefacts (`.pt` files).
*   **/notebooks/:** Contains the Jupyter notebooks used to orchestrate the experiments and visualize the results. They should be run in numerical order.
    *   `01_feature_extraction_setup.ipynb`: Loads raw data and runs all three feature extraction pipelines.
    *   `02_model_evaluation.ipynb`: Runs all SVM-based experiments and generates analysis.
    *   `03_cnn_lstm_experiment.ipynb`: Runs all CNN-LSTM experiments and generates the final comparative analysis.
*   **/src/:** Contains all modular Python source code for data loading, feature extraction, model architectures, and evaluation strategies.
*   `.gitignore`: Specifies which files and directories to exclude from version control.
*   `.gitattributes`: Configures Git LFS to track specific file types.
*   `environment.yml` & `conda-lock.yml`: Files for reproducing the Conda environment.

---

### Setup and Installation

This project uses `conda-lock` to ensure a fully reproducible environment. The provided lock file is for a **Windows x64 system with an NVIDIA GPU**.

#### For Users on Windows with an NVIDIA GPU (Recommended)
1.  Install `conda-lock` in your base Conda environment: `conda install -c conda-forge conda-lock`
2.  Navigate to the project's root directory.
3.  Create and activate the environment from the lock file:
    ```bash
    conda-lock install -n msc_final_env
    ```

#### For Users on Other Platforms (e.g., macOS, Linux, or Windows without GPU)
The provided `conda-lock.yml` will not work. You must create the environment from the `environment.yml` specification file. The CUDA dependency will be automatically ignored, and a CPU-only version of PyTorch will be installed.

1.  Navigate to the project's root directory.
2.  Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate msc_final_env
    ```

---

### Usage / Workflow to Reproduce Results

1.  **Setup Environment:** Create and activate the Conda environment using the instructions above.

2.  **Download Raw Data:** Download the Androids Corpus from the official repository: **[https://github.com/androidscorpus/data](https://github.com/androidscorpus/data)**. Place it in a directory outside of this project folder (e.g., `E:/Dissertation_Data/Androids-Corpus`). You must update the `BASE_DATA_PATH` variable in the notebooks to point to this location.

3.  **Replace Configuration File:** This project uses a corrected OpenSMILE configuration file (`Androids.conf`) to ensure the summary features are generated correctly. You must use the version provided in this repository.
    *   Take the `Androids.conf` file from the root of this project folder.
    *   **Copy and paste** it into your downloaded `Androids-Corpus` directory, **overwriting and replacing** the original `Androids.conf` file that came with the dataset.

4.  **Download Pre-computed Sequence Files:** The most time-consuming step is extracting the sequential features for the deep learning models. A pre-computed `.zip` file containing these is available for download.
    *   **Download Link:** **[Download Pre-computed Sequence Files (Google Drive)](https://drive.google.com/file/d/12uFFngU0cxEZbRRbyUfJ2Nf4VQeHhXO-/view?usp=drive_link)**
    *   Download the `precomputed_sequences_for_download.zip` file from the link above.
    *   **Unzip** the file.
    *   Place the two resulting `.pkl` files (`features_wav2vec2_sequences_reading_task.pkl` and `features_wav2vec2_sequences_interview_clips.pkl`) inside the `data/Processed_Features/` directory of this project.

5.  **Run Notebooks in Order:**
    *   **`notebooks/01_feature_extraction_setup.ipynb`**: Run this notebook to generate the summary-statistic feature files (`.csv`) for the SVM models. If you downloaded the pre-computed files in the previous step, this notebook will skip the slow sequence extraction.
    *   **`notebooks/02_model_evaluation.ipynb`**: Run this to perform all 18 SVM experiments and visualize the results.
    *   **`notebooks/03_cnn_lstm_experiment.ipynb`**: Run this to perform all 6 CNN-LSTM experiments and generate the final comparative analysis for the entire project.


