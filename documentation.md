# Technical Documentation — One-Shot Offline Signature Recognition System

Technical reference and operational guide for the **One-Shot Offline Signature Recognition & Verification System**, developed for the Digital Image Processing (*Pengolahan Citra Digital*) course.

## System Architecture & Processing Pipeline

The system performs offline signature **identification (`1:N`)** and **verification (`1:1`)** under a **One-Shot Registration** constraint: each enrolled user is represented by exactly **one** reference signature template.

### 1. Multi-Stage Image Preprocessing (`src/preprocessing.py`)

Every input signature (enrolled template or probe query) passes through a deterministic 9-stage normalization pipeline before feature extraction:

1. **Image Ingestion (`load_image`)**: Loads images from file paths, raw byte streams, or NumPy arrays.
2. **Grayscale Conversion & Border Trimming (`to_grayscale`, `trim_black_borders`)**: Converts inputs to single-channel grayscale (compositing RGBA alpha channels over white) and strips dark scanner/frame borders.
3. **Illumination Flattening (`flatten_illumination`)**: Estimates background lighting via morphological closing and normalizes uneven camera shadows.
4. **Noise Suppression (`reduce_noise`)**: Applies a `3 × 3` Gaussian blur kernel to smooth high-frequency sensor noise.
5. **Otsu Binarization (`binarize_signature`)**: Segments foreground ink strokes (`255`) from the paper background (`0`).
6. **Artifact & Speck Removal (`remove_ruled_lines`, `filter_small_noise`)**: Detects and removes horizontal ruled notebook/form lines and filters isolated connected components below the minimum area threshold.
7. **Tight Bounding-Box Cropping (`get_signature_bounding_box`)**: Locates the active ink envelope and crops surrounding whitespace.
8. **Deskewing (`correct_skew`)**: Corrects minor rotational tilt along the principal stroke axis.
9. **Aspect-Preserving Resize & Centroid Alignment (`resize_with_aspect_ratio`, `align_by_centroid`)**: Scales the cropped signature to fit a standardized **`128 × 128`** pixel canvas with symmetric padding, then shifts the ink center of mass to the canvas center.

### 2. Hybrid Feature Extraction — 8,119 Dimensions (`src/feature_extraction.py`)

Each sub-descriptor is independently **L2-normalized** prior to concatenation (`extract_features`) so high-dimensional gradient bins do not dominate global shape and geometric features:

- **HOG (Histogram of Oriented Gradients) — 8,100 dimensions**: Configured with `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, and `block_norm="L2-Hys"`. Encodes local stroke directionality, edge gradients, and line intersections across the `128 × 128` grid.
- **Hu Invariant Moments — 7 dimensions**: Global shape moments invariant to translation, scale, and rotation, stabilized via logarithmic transformation (`-sign(h) * log10(|h| + eps)` using absolute values for numerical stability).
- **Geometric Features — 12 dimensions**: Captures aspect ratio, ink occupancy ratio, normalized bounding-box width and height, contour area ratio, overall stroke density, normalized centroid coordinates (`cx`, `cy`), horizontal and vertical spatial variance, and left-right / top-bottom ink balance ratios.

### 3. Matching & Decision Models (`src/similarity.py`, `src/identification.py`, `src/verification.py`, `config/settings.py`)

- **Model A — Cosine Similarity (Non-Parametric Baseline)**
  Computes the direct cosine similarity between L2-normalized `8,119`-dimensional feature vectors:
  - Default Match Threshold (`DEFAULT_COSINE_MATCH_THRESHOLD`): `0.80`
  - Default Uncertain Lower Bound (`DEFAULT_COSINE_UNCERTAIN_LOWER`): `0.76`
  - Default Top-1 vs Top-2 Margin (`DEFAULT_TOP_MARGIN_THRESHOLD`): `0.025`

- **Model B — Pairwise Calibrated Linear SVM**
  Uses the pre-trained pipeline stored at `data/models/linear_svm_pipeline.joblib`, operating on pairwise comparison vectors constructed as `[|F_A - F_B|, F_A * F_B, cosine_similarity]`:
  - Default Match Threshold (`DEFAULT_SVM_MATCH_THRESHOLD`): `0.40`
  - Default Uncertain Lower Bound (`DEFAULT_SVM_UNCERTAIN_LOWER`): `0.40`

## Quick Start

The project workspace includes a ready-to-use virtual environment (`.venv`), an initialized SQLite database (`data/signature_recognition.db`) pre-loaded with demo identities (`Alice Vance`, `Bob Chen`, `Catherine Ross`, `David Miller`, `Emma Watson`, and `abi`), cached CEDAR features (`data/processed/cedar_feature_cache.npy`), and trained SVM models under `data/models/`.

Run the application from PowerShell:

```powershell
# 1. Navigate to the project root
cd c:\laragon\www\Signature

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Launch the Streamlit server
streamlit run app.py
```

**Direct execution without activating `.venv`:**

```powershell
cd c:\laragon\www\Signature
.\.venv\Scripts\streamlit.exe run app.py
```

Once running, open `http://localhost:8501` in a browser.

> **PowerShell Execution Policy**: If script execution is disabled on your system, run the following command in your active PowerShell session before activating `.venv`:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

## Fresh Environment Setup

Follow these steps when deploying the repository on a new machine or rebuilding the environment from scratch.

### Prerequisites

- **Operating System**: Windows (PowerShell), Linux, or macOS
- **Python**: **3.10+** (developed and tested on Python `3.10.0`)

### Step-by-Step Installation

1. **Create and activate a virtual environment (`.venv`)**:
   ```powershell
   cd c:\laragon\www\Signature
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Python dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
   Key packages installed: `streamlit==1.65.0`, `opencv-python==5.0.0.93`, `scikit-image==0.25.2`, `scikit-learn==1.7.2`, `pymupdf==1.28.2`, `numpy==2.2.6`, `pandas==2.3.3`, `matplotlib==3.10.9`, `joblib==1.6.0`, and `pytest==9.2.0`.

3. **Seed demo users into SQLite**:
   ```powershell
   python seed_demo_users.py
   ```
   Creates `data/signature_recognition.db` (if missing) and enrolls 5 baseline users from the CEDAR dataset (`Dataset/full_org/original_1_1.png` through `original_5_1.png`) whenever fewer than 3 users exist in the database.

4. **Verify module integrity via unit tests**:
   ```powershell
   python run_tests.py
   ```

5. **Start the Streamlit web application**:
   ```powershell
   streamlit run app.py
   ```

## Web Application Guide

The Streamlit interface is organized into five views accessible from the sidebar or the main dashboard navigation cards:

### 1. Main Dashboard (`app.py`)
- Displays **5 real-time SQLite summary counters**: *Registered Users*, *Total Attempts*, *Matches*, *Rejections / Unknown*, and *Uncertain*.
- Shows the **Recent Verification Activity** log table across both `1:N` identification and `1:1` verification runs.
- Renders the **Benchmark Preview (E1 - E6)** chart when `experiments/results/experiment_comparison_chart.png` is present.

### 2. Register (`pages/1_Register.py`)
Enrolls new users under the **One-Shot Enrollment** rule (1 reference template per identity):
1. Enter the **Full Name / User Identity** (enable *Overwrite existing template* to replace an existing user's reference signature).
2. Upload a signature file in **`PNG`**, **`JPG`**, **`JPEG`**, or **`PDF`** format.
   - **`PDF` inputs** are rendered via `PyMuPDF` (`fitz`). Select the target page number and isolate the signature region (*Region of Interest / ROI*) using the `X Position (%)`, `Y Position (%)`, `Width (%)`, and `Height (%)` sliders.
3. Inspect the live preprocessing preview in the right panel (**Original Input**, **Otsu Binary (Ink=255)**, and **128x128 Normalized Template**) along with the extracted `8,119`-dimensional vector summary.
4. Click **Complete Registration** to persist the normalized template image (`.png`), feature vector (`.npy`) in `data/templates/`, and user metadata in SQLite.
5. Manage existing enrollments in the **Currently Registered Users** section at the bottom of the page, including one-click user and template file deletion via **Delete**.

### 3. 1:N Identify (`pages/2_Identify.py`)
Determines the author of an unlabelled probe signature by comparing it against all enrolled templates:
1. Select a matching backend: **Cosine Similarity (HOG + Hu + Geometry)** or **Learned SVM (Pairwise Classifier)**.
2. *(Optional)* Expand **Advanced Threshold Settings** to customize the *Match Threshold*, *Uncertain Lower Bound*, and *Top-1 vs Top-2 Confidence Margin*.
3. Upload an image or PDF signature probe (with interactive ROI cropping for PDFs).
4. Review the resulting decision badge (**`MATCH`**, **`UNCERTAIN`**, or **`UNKNOWN`**), top similarity score, Top-1 vs Top-2 margin, full **Ranked Candidate Enrolled Templates** table, and side-by-side visual comparison between the *Query* and *Top Template*. Every identification attempt is recorded in the `verification_logs` table.

### 4. 1:1 Verify (`pages/3_Verify.py`)
Evaluates whether a submitted signature belongs to a specific claimed identity:
1. Select the **Claimed User** from the enrolled user dropdown (their reference template is previewed automatically).
2. Choose the matching model (**Cosine Similarity** or **Learned SVM**) and adjust **Decision Thresholds** if needed.
3. Upload the probe signature (`PNG`, `JPG`, `JPEG`, or `PDF` with ROI cropping).
4. Inspect the verification outcome (**`Match`**, **`Uncertain`**, or **`Not Match`**), *Similarity Score*, decision rule explanation, and side-by-side comparison of the *Submitted Probe* against the *Registered Template*. All verification events are logged to SQLite.

### 5. Evaluation (`pages/4_Evaluation.py`)
Visualizes empirical benchmark results loaded from `experiments/results/experiment_metrics.json` on the **CEDAR** dataset (55 writers, 2,640 genuine and skilled-forgery signatures evaluated under a strict **Writer-Disjoint Split**: 35 training writers, 10 validation writers, and 10 unseen test writers):
- Displays headline KPIs for the best-performing configuration (*Test Accuracy*, *EER*, *ROC AUC*, *Top-1 ID Accuracy*, *FAR*, and *Mean Latency*).
- Renders the full **E1 – E6** tabular breakdown alongside `experiment_comparison_chart.png`, `roc_curves.png`, and `best_model_confusion_matrix.png`.

## CLI Operations (Testing, Retraining & Experiments)

| Task | Command | Description |
| :--- | :--- | :--- |
| **Run Unit Tests** | `.\.venv\Scripts\python.exe run_tests.py` | Executes the unit test suite covering preprocessing, feature extraction, and similarity/matching (`tests/`). |
| **Run E1–E6 Benchmarks** | `.\.venv\Scripts\python.exe experiments/run_experiments.py` | Loads or extracts the CEDAR feature cache, runs experimental protocols E1–E6, and saves `experiment_metrics.json` and trained models. |
| **Regenerate Evaluation Plots** | `.\.venv\Scripts\python.exe experiments/generate_charts.py` | Rebuilds `experiment_comparison_chart.png`, `best_model_confusion_matrix.png`, and `roc_curves.png` from `experiment_metrics.json`. |
| **Sync Templates & Retrain SVM** | `.\.venv\Scripts\python.exe scripts/sync_all_features_and_models.py` | Re-extracts feature vectors for all registered SQLite users, synchronizes `cedar_feature_cache.npy`, and retrains `linear_svm_pipeline.joblib` and `best_model.joblib`. |
| **Full CEDAR Extraction & SVM Validation** | `.\.venv\Scripts\python.exe scripts/retrain_and_validate_svm.py` | Recomputes features across all 2,640 CEDAR images from scratch, then trains and validates both the Linear SVM and RBF SVM pipelines. |
| **Quick SVM Inference Check** | `.\.venv\Scripts\python.exe scripts/test_svm_verification.py` | Runs a sanity check on `linear_svm_pipeline.joblib` inference against currently enrolled database templates. |
| **Rebuild Training Notebook** | `.\.venv\Scripts\python.exe scripts/generate_training_notebook.py` | Regenerates `notebooks/Signature_Recognition_Training.ipynb`. |

To explore the interactive notebook (`notebooks/Signature_Recognition_Training.ipynb`), open it in VS Code or Jupyter and select the `.venv` Python kernel.

### Experimental Protocol Summary (E1 – E6)

| Experiment | Feature Representation | Classifier / Matching Strategy |
| :--- | :--- | :--- |
| **E1** | HOG (`8,100` dims) | Cosine Similarity |
| **E2** | HOG + Hu Moments (`8,107` dims) | Cosine Similarity |
| **E3** | HOG + Hu + Geometry (`8,119` dims) | Pairwise Linear SVM |
| **E4** | HOG + Hu + Geometry (`8,119` dims) | Pairwise RBF SVM |
| **E5** | Best Model without Augmentation | Evaluated on 10 Unseen Test Writers |
| **E6** | Best Model with Controlled Geometric Augmentation | Evaluated on 10 Unseen Test Writers |

## Repository Structure

```text
Signature/
├── app.py                              # Main Streamlit entrypoint (Dashboard & navigation)
├── requirements.txt                    # Python package dependencies
├── run_tests.py                        # Unit test runner (preprocessing, features, matching)
├── seed_demo_users.py                  # Database seeder for 5 CEDAR demo identities
├── README.md                           # Repository overview and quick start
├── documentation.md                    # Complete technical reference and operational guide
│
├── config/
│   └── settings.py                     # Directory paths, preprocessing/HOG configs, & decision thresholds
│
├── data/
│   ├── raw/                            # Raw uploaded assets (if retained)
│   ├── processed/
│   │   └── cedar_feature_cache.npy     # Cached feature vectors for the CEDAR dataset
│   ├── templates/                      # Enrolled user normalized templates (.png) & vectors (.npy)
│   ├── models/
│   │   ├── linear_svm_pipeline.joblib  # Calibrated Linear SVM pipeline (primary UI model)
│   │   ├── best_model.joblib           # Exported best-performing model artifact
│   │   └── rbf_svm_pipeline.joblib     # Calibrated RBF SVM pipeline
│   └── signature_recognition.db        # SQLite database (users, signature_templates, verification_logs)
│
├── Dataset/                            # CEDAR benchmark dataset (55 writers)
│   ├── full_org/                       # 1,320 genuine signatures (55 writers × 24 samples)
│   └── full_forg/                      # 1,320 skilled forgeries (55 writers × 24 samples)
│
├── pages/
│   ├── 1_Register.py                   # One-shot user enrollment view (Image/PDF + ROI crop)
│   ├── 2_Identify.py                   # 1:N signature identification view
│   ├── 3_Verify.py                     # 1:1 signature verification view
│   └── 4_Evaluation.py                 # E1–E6 benchmark metrics & visual plots view
│
├── src/
│   ├── preprocessing.py                # 9-stage image cleaning & 128x128 centroid normalization
│   ├── feature_extraction.py           # HOG, Hu Moments, and Geometric descriptor extraction
│   ├── similarity.py                   # Cosine similarity, Euclidean distance, & pairwise vector builder
│   ├── dataset_loader.py               # CEDAR dataset parser & writer-disjoint split generator
│   ├── augmentation.py                 # Controlled geometric augmentations for training experiments
│   ├── model_training.py               # SVM training pipelines, EER/AUC computation, & serialization
│   ├── identification.py               # 1:N candidate ranking & top-margin decision logic
│   ├── verification.py                 # 1:1 verification scoring & threshold classification
│   ├── pdf_handler.py                  # PyMuPDF page rendering & percentage-based ROI cropping
│   └── database.py                     # SQLite schema management, user CRUD, & audit logging
│
├── experiments/
│   ├── run_experiments.py              # End-to-end runner for experiments E1–E6
│   ├── generate_charts.py              # Plot generator for benchmark bars, confusion matrix, & ROC
│   └── results/                        # Output directory for experiment_metrics.json & PNG figures
│
├── scripts/
│   ├── sync_all_features_and_models.py # Syncs enrolled template vectors & retrains Linear SVM
│   ├── retrain_and_validate_svm.py     # Full CEDAR re-extraction & Linear/RBF SVM validation
│   ├── test_svm_verification.py        # CLI sanity check for SVM verification inference
│   └── generate_training_notebook.py   # Programmatic builder for the Jupyter training notebook
│
├── notebooks/
│   └── Signature_Recognition_Training.ipynb # Interactive model training & evaluation notebook
│
└── tests/
    ├── test_preprocessing.py           # Unit tests for image normalization steps
    ├── test_features.py                # Unit tests for feature dimensionality & normalization
    └── test_matching.py                # Unit tests for similarity metrics & pairwise construction
```

## Troubleshooting

1. **PowerShell blocks `.venv` activation (`cannot be loaded because running scripts is disabled`)**
   - Temporarily relax the execution policy for the active terminal process, or invoke the binary directly via `.\.venv\Scripts\streamlit.exe run app.py`:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
     ```

2. **Port `8501` is already in use**
   - Bind Streamlit to an alternate port such as `8502`:
     ```powershell
     .\.venv\Scripts\streamlit.exe run app.py --server.port 8502
     ```

3. **Warning: *"No users are currently registered"* on the Identify or Verify page**
   - Seed the default CEDAR identities by running `.\.venv\Scripts\python.exe seed_demo_users.py`, or enroll a user manually under **Register** (`pages/1_Register.py`).

4. **Warning: *"Trained SVM model not found. Using Cosine Similarity"* when selecting Learned SVM**
   - Verify that `data/models/linear_svm_pipeline.joblib` exists. To rebuild the model and synchronize enrolled user templates, run:
     ```powershell
     .\.venv\Scripts\python.exe scripts/sync_all_features_and_models.py
     ```

5. **Evaluation page displays *"metrics file not found yet"***
   - Run the benchmark suite and chart generator to populate `experiments/results/experiment_metrics.json` and its accompanying figures:
     ```powershell
     .\.venv\Scripts\python.exe experiments/run_experiments.py
     .\.venv\Scripts\python.exe experiments/generate_charts.py
     ```

## Academic Scope & Disclaimer

This software is an academic prototype built for instruction and empirical evaluation in Digital Image Processing (*Pengolahan Citra Digital*). High similarity scores reflect geometric and topological correspondence between 2D offline ink strokes and do not constitute forensic or legal certification of signature authenticity.
