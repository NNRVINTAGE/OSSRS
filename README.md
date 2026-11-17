# One-Shot Signature Recognition System

An offline signature identification (`1:N`) and verification (`1:1`) system built around a strict one-shot enrollment constraint—requiring only a single reference signature per registered user. Developed as an engineering and research prototype for Digital Image Processing (*Pengolahan Citra Digital*).

Full architecture details, feature extraction specifications, experiment protocols, and troubleshooting guides are available in [documentation.md](documentation.md).

## Quick Start

This repository ships with a pre-configured virtual environment (`.venv`), a seeded SQLite database (`data/signature_recognition.db`), cached CEDAR features (`data/processed/cedar_feature_cache.npy`), and a calibrated Linear SVM pipeline (`data/models/linear_svm_pipeline.joblib`).

Run the application from PowerShell in the project root:

```powershell
cd c:\laragon\www\Signature
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Or launch Streamlit directly without activating the virtual environment:

```powershell
cd c:\laragon\www\Signature
.\.venv\Scripts\streamlit.exe run app.py
```

Open `http://localhost:8501` in your browser.

> **PowerShell Execution Policy**: If `.venv` activation is blocked, allow local scripts for the current session:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

## Fresh Installation

To set up the environment from scratch on a new machine (requires **Python 3.10+**):

```powershell
# 1. Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Seed the SQLite database with 5 demo users from the CEDAR dataset
python seed_demo_users.py

# 4. Run the unit test suite
python run_tests.py

# 5. Start the Streamlit web application
streamlit run app.py
```

## Web Interface (`pages/`)

| View | Source File | Description |
| :--- | :--- | :--- |
| **Dashboard** | `app.py` | Displays live SQLite metrics (enrolled users, verification attempts, match breakdown), recent activity logs, and E1–E6 benchmark previews. |
| **Register** | `pages/1_Register.py` | Enrolls a single reference template per user from `PNG`, `JPG`, `JPEG`, or `PDF` files (includes page selection and interactive ROI cropping for PDFs). |
| **1:N Identify** | `pages/2_Identify.py` | Matches an unlabelled probe signature against all enrolled templates using Cosine Similarity or Pairwise Linear SVM, returning ranked candidates and confidence margins. |
| **1:1 Verify** | `pages/3_Verify.py` | Verifies a probe signature against a claimed identity (`Match`, `Uncertain`, or `Not Match`) with side-by-side visual inspection. |
| **Evaluation** | `pages/4_Evaluation.py` | Presents empirical E1–E6 benchmark metrics on the CEDAR dataset (55 writers, writer-disjoint split), ROC curves, and confusion matrices. |

## Testing, Training & Experiments

```powershell
# Run unit tests (preprocessing, feature extraction, matching)
.\.venv\Scripts\python.exe run_tests.py

# Execute E1–E6 benchmark experiments and export metrics to experiments/results/
.\.venv\Scripts\python.exe experiments/run_experiments.py

# Regenerate evaluation charts (comparison bar chart, ROC curves, confusion matrix)
.\.venv\Scripts\python.exe experiments/generate_charts.py

# Re-extract enrolled user templates, sync CEDAR cache, and retrain SVM pipelines
.\.venv\Scripts\python.exe scripts/sync_all_features_and_models.py
```

See [documentation.md](documentation.md) for the complete `128 × 128` preprocessing pipeline, the `8,119`-dimensional hybrid descriptor (`HOG + Hu Moments + Geometric Features`), threshold parameters in `config/settings.py`, and directory layout.
