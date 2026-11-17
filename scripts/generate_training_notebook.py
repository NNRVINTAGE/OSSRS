import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_markdown(source_lines):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines]
        })

    def add_code(source_lines):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines]
        })

    # -------------------------------------------------------------
    # Title & Metadata
    # -------------------------------------------------------------
    add_markdown([
        "# ✍️ One-Shot Offline Signature Recognition & Verification System",
        "### Project Tugas Besar: Pengolahan Citra Digital (Digital Image Processing)",
        "**Pendekatan:** Classical Computer Vision & Machine Learning (OpenCV, scikit-image, scikit-learn)",
        "",
        "---",
        "## 📌 Ringkasan Proyek",
        "Sistem ini dirancang untuk melakukan:",
        "1. **Signature Verification (1:1):** Memverifikasi apakah tanda tangan uji cocok dengan 1 template terdaftar milik pengguna yang diklaim.",
        "2. **Signature Identification (1:N):** Mencari dan mencocokkan tanda tangan yang tidak diketahui ke seluruh database template terdaftar.",
        "",
        "### 💡 Karakteristik Utama:",
        "- **One-Shot Registration:** Setiap pengguna hanya mendaftarkan tepat **1 citra tanda tangan**.",
        "- **Writer-Disjoint Splitting:** Pembagian dataset CEDAR menjamin bahwa penulis (*writers*) pada data latih tidak pernah muncul di data validasi dan pengujian (mencegah *data leakage*).",
        "- **Explainable Feature Extraction:** Kombinasi HOG (lokal), Hu Moments (invarian bentuk global), dan Fitur Geometri (distribusi stroke & rasio).",
        "- **Pairwise SVM Classifier:** Model klasifikasi berbasis vektor selisih mutlak, perkalian elemen, dan kesamaan kosinus.",
        "- **Model Export:** Model yang dilatih langsung dapat diekspor ke format `.joblib` untuk digunakan oleh aplikasi web Streamlit."
    ])

    # -------------------------------------------------------------
    # Setup & Imports
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 1. Setup Lingkungan (Google Colab / Local VS Code)",
        "Notebook ini dapat dijalankan langsung baik di **Local (VS Code / Jupyter Notebook)** maupun di **Google Colab**.",
        "",
        "> 💡 **Panduan Jika Menggunakan Google Colab:**",
        "> - Jika di Google Colab, Anda perlu meng-upload folder `Dataset/` dan `src/` (bisa melalui Google Drive atau upload file `.zip`).",
        "> - Sel di bawah ini secara otomatis mendeteksi lingkungan eksekusi dan mengonfigurasi `BASE_DIR` serta pustaka pendukung."
    ])

    add_code([
        "# Deteksi otomatis lingkungan eksekusi (Google Colab vs Local)",
        "try:",
        "    import google.colab",
        "    IN_COLAB = True",
        "    print('🚀 Berjalan di Google Colab!')",
        "except ImportError:",
        "    IN_COLAB = False",
        "    print('💻 Berjalan di Lingkungan Lokal (VS Code / Jupyter)')",
        "",
        "from pathlib import Path",
        "",
        "if IN_COLAB:",
        "    # 1. Jika menyimpan project di Google Drive (opsional):",
        "    # from google.colab import drive",
        "    # drive.mount('/content/drive')",
        "    # BASE_DIR = Path('/content/drive/MyDrive/Signature')",
        "    ",
        "    # 2. Atau jika di-upload langsung ke Colab:",
        "    BASE_DIR = Path('/content/Signature') if Path('/content/Signature').exists() else Path('/content')",
        "    !pip install -q opencv-python scikit-image scikit-learn joblib matplotlib pandas",
        "else:",
        "    BASE_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()",
        "",
        "import os",
        "import sys",
        "import time",
        "import random",
        "import joblib",
        "import cv2",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from typing import Dict, List, Tuple, Optional",
        "",
        "# Menambahkan path root proyek ke sys.path",
        "sys.path.insert(0, str(BASE_DIR))",
        "",
        "print(f'Root direktori aktif: {BASE_DIR}')",
        "print(f'OpenCV version: {cv2.__version__}')",
        "print(f'NumPy version: {np.__version__}')"
    ])

    # -------------------------------------------------------------
    # Dataset Loading & Splits
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 2. Dataset Inspection & Writer-Disjoint Splitting",
        "Dataset yang digunakan adalah **CEDAR Signature Dataset**:",
        "- 55 Penulis unik (*writers*)",
        "- Setiap penulis memiliki 24 tanda tangan asli (*genuine*) dan 24 pemalsuan ahli (*skilled forgeries*)",
        "- Total: $55 \\times 48 = 2.640$ citra tanda tangan.",
        "",
        "> ⚠️ **Prinsip Bebas Kebocoran Data (Zero Data Leakage):**",
        "> Pembagian data dilakukan per individu/penulis (*writer-disjoint*):",
        "> - **Train Writers (35 orang):** Penulis ID 1 s.d. 35",
        "> - **Validation Writers (10 orang):** Penulis ID 36 s.d. 45",
        "> - **Test Writers (10 orang):** Penulis ID 46 s.d. 55"
    ])

    add_code([
        "# Verifikasi keberadaan folder Dataset",
        "dataset_dir = BASE_DIR / 'Dataset'",
        "if not (dataset_dir / 'full_org').exists():",
        "    print(f'⚠️ PERINGATAN: Folder {dataset_dir / \"full_org\"} belum ditemukan!')",
        "    print('Jika di Google Colab, silakan upload Dataset.zip lalu jalankan:')",
        "    print('!unzip -q Dataset.zip -d /content/Signature/Dataset')",
        "else:",
        "    print(f'✅ Dataset CEDAR ditemukan di: {dataset_dir}')",
        "",
        "from src.dataset_loader import get_cedar_metadata, get_writer_splits",
        "",
        "# Membaca metadata seluruh citra CEDAR",
        "df_metadata = get_cedar_metadata(dataset_dir=dataset_dir)",
        "print(f'Total citra ditemukan: {len(df_metadata)}')",
        "print(df_metadata.head())",
        "",
        "# Writer-Disjoint Split",
        "train_wids, val_wids, test_wids = get_writer_splits(",
        "    df_metadata, train_count=35, val_count=10, test_count=10, random_seed=42",
        ")",
        "",
        "print(f'Jumlah Penulis Data Latih: {len(train_wids)} -> {train_wids[:10]} ...')",
        "print(f'Jumlah Penulis Data Validasi: {len(val_wids)} -> {val_wids}')",
        "print(f'Jumlah Penulis Data Uji: {len(test_wids)} -> {test_wids}')"
    ])

    # -------------------------------------------------------------
    # Image Preprocessing Pipeline
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 3. Pipeline Pra-pemrosesan Citra (Image Preprocessing)",
        "Tahapan pra-pemrosesan yang diimplementasikan:",
        "1. **Grayscale Conversion**: Mengubah citra RGB menjadi skala abu-abu.",
        "2. **Border Trimming**: Menghapus margin gelap atau bingkai viewport ponsel.",
        "3. **Illumination Flattening**: Mengoreksi bayangan tidak merata melalui operasi *morphological closing* dan *division* ($I_{flat} = I / I_{bg} \\times 255$).",
        "4. **Ruled Line Removal**: Menghilangkan garis buku bergaris/folio dengan kernel horizontal opening dan reconnecting strokes.",
        "5. **Otsu Binarization**: Segmentasi foreground tanda tangan (putih) dan latar belakang (hitam).",
        "6. **Bounding Box Cropping**: Memotong area tanda tangan aktif.",
        "7. **Aspect Ratio Padding**: Melakukan padding proporsional ke dimensi kanvas $128 \\times 128$ piksel.",
        "8. **Centroid (Center of Mass) Alignment**: Menggeser pusat massa tanda tangan ke $(64, 64)$ untuk menjamin *translation invariance*."
    ])

    add_code([
        "from src.preprocessing import preprocess_signature_image, load_image",
        "",
        "# Mengambil salah satu contoh citra dari dataset",
        "sample_path = df_metadata.iloc[0]['filepath']",
        "orig_img = load_image(sample_path)",
        "processed_img, debug_info = preprocess_signature_image(orig_img, return_debug=True)",
        "",
        "# Visualisasi Langkah-langkah Pra-pemrosesan",
        "fig, axes = plt.subplots(1, 4, figsize=(16, 4))",
        "axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))",
        "axes[0].set_title('1. Original Input')",
        "axes[0].axis('off')",
        "",
        "axes[1].imshow(debug_info.get('gray', orig_img), cmap='gray')",
        "axes[1].set_title('2. Grayscale & Flat Illum')",
        "axes[1].axis('off')",
        "",
        "axes[2].imshow(debug_info.get('binary', processed_img), cmap='gray')",
        "axes[2].set_title('3. Clean Binary Mask')",
        "axes[2].axis('off')",
        "",
        "axes[3].imshow(processed_img, cmap='gray')",
        "axes[3].set_title('4. Centroid Aligned (128x128)')",
        "axes[3].axis('off')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # -------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 4. Ekstraksi Fitur (Feature Extraction)",
        "Fitur diekstraksi dari citra biner ternormalisasi $128 \\times 128$:",
        "1. **HOG (Histogram of Oriented Gradients)**: 8.100 dimensi",
        "   - Menangkap orientasi stroke lokal dan pola goresan pena.",
        "   - Konfigurasi: 9 orientasi, sel $8 \\times 8$, blok $2 \\times 2$, normalisasi L2-Hys.",
        "2. **Hu Invariant Moments**: 7 dimensi",
        "   - Menangkap bentuk global yang invarian terhadap translasi, skala, dan rotasi.",
        "   - Diberikan log-transform: $-\\text{sign}(h_i) \\cdot \\log_{10}(|h_i| + \\epsilon)$.",
        "3. **Fitur Geometris**: 12 dimensi",
        "   - Aspek rasio, kepadatan tinta (*ink occupancy*), rasio area kontur, centroid relatif, dan profil proyeksi horizontal/vertikal.",
        "",
        "> ⚖️ **Balanced Sub-vector L2 Normalization:**",
        "> Setiap sub-vektor dinormalisasi ke unit L2-norm sebelum digabung:",
        "> $$\\mathbf{f} = [\\hat{\\mathbf{f}}_{HOG}, \\hat{\\mathbf{f}}_{Hu}, \\hat{\\mathbf{f}}_{Geom}]$$",
        "> Total dimensi fitur: $8.100 + 7 + 12 = 8.119$ dimensi."
    ])

    add_code([
        "from src.feature_extraction import (",
        "    extract_features,",
        "    extract_hog_features,",
        "    extract_hu_moments,",
        "    extract_geometric_features",
        ")",
        "",
        "hog_feat, hog_vis = extract_hog_features(processed_img, return_visualization=True)",
        "hu_feat = extract_hu_moments(processed_img)",
        "geom_feat = extract_geometric_features(processed_img)",
        "combined_feat = extract_features(processed_img)",
        "",
        "print(f'Dimensi HOG: {hog_feat.shape}')",
        "print(f'Dimensi Hu Moments: {hu_feat.shape} -> Nilai: {np.round(hu_feat, 4)}')",
        "print(f'Dimensi Geometri: {geom_feat.shape}')",
        "print(f'Dimensi Total Fitur Gabungan: {combined_feat.shape}')",
        "",
        "# Visualisasi HOG Gradient Map",
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
        "axes[0].imshow(processed_img, cmap='gray')",
        "axes[0].set_title('Preprocessed Binary (128x128)')",
        "axes[0].axis('off')",
        "",
        "axes[1].imshow(hog_vis, cmap='inferno')",
        "axes[1].set_title('HOG Stroke Gradient Descriptors')",
        "axes[1].axis('off')",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # -------------------------------------------------------------
    # Feature Caching
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 5. Caching Fitur Dataset CEDAR",
        "Untuk efisiensi komputasi, fitur seluruh 2.640 citra dihitung dan disimpan di `data/processed/cedar_feature_cache.npy`."
    ])

    add_code([
        "cache_file = BASE_DIR / 'data' / 'processed' / 'cedar_feature_cache.npy'",
        "",
        "if cache_file.exists():",
        "    print(f'Memuat feature cache yang telah tersedia dari {cache_file}...')",
        "    feature_cache = np.load(cache_file, allow_pickle=True).item()",
        "else:",
        "    print('Mengekstraksi fitur seluruh dataset CEDAR...')",
        "    feature_cache = {}",
        "    for idx, row in df_metadata.iterrows():",
        "        p = row['filepath']",
        "        proc = preprocess_signature_image(p)",
        "        comb = extract_features(proc)",
        "        feature_cache[p] = {'combined': comb}",
        "    cache_file.parent.mkdir(parents=True, exist_ok=True)",
        "    np.save(cache_file, feature_cache)",
        "",
        "print(f'Total citra dalam cache: {len(feature_cache)}')",
        "sample_item = list(feature_cache.keys())[0]",
        "print(f'Contoh bentuk fitur: {feature_cache[sample_item][\"combined\"].shape}')"
    ])

    # -------------------------------------------------------------
    # Pairwise Dataset Construction
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 6. Konstruksi Sampel Pasangan (*Pairwise Dataset*)",
        "Untuk melatih model SVM (*Model B*), kita membentuk pasangan tanda tangan:",
        "- **Label 1 (Same Writer):** Pasangan citra asli dari penulis yang sama ($G_A$ vs $G_A$).",
        "- **Label 0 (Different Writer):** Pasangan citra asli dari dua penulis berbeda ($G_A$ vs $G_B$).",
        "- **Label 0 (Skilled Forgery):** Pasangan citra asli vs pemalsuan ahli yang meniru tanda tangan tersebut ($G_A$ vs $F_A$).",
        "",
        "Vektor representasi pasangan dihitung menggunakan:",
        "$$\\mathbf{x}_{pair} = [|\\mathbf{f}_A - \\mathbf{f}_B|, \\mathbf{f}_A \\odot \\mathbf{f}_B, \\text{CosineSimilarity}(\\mathbf{f}_A, \\mathbf{f}_B)]$$",
        "Dimensi: $8.119 + 8.119 + 1 = 16.239$ fitur."
    ])

    add_code([
        "from src.dataset_loader import create_pairwise_dataset",
        "",
        "combined_cache = {p: feature_cache[p]['combined'] for p in feature_cache}",
        "",
        "print('Membangun pairwise training samples dari 35 train writers...')",
        "X_train_pairs, y_train_pairs = create_pairwise_dataset(",
        "    df_metadata, combined_cache, train_wids, pairs_per_writer=35, random_seed=42",
        ")",
        "",
        "print('Membangun pairwise validation samples dari 10 val writers...')",
        "X_val_pairs, y_val_pairs = create_pairwise_dataset(",
        "    df_metadata, combined_cache, val_wids, pairs_per_writer=30, random_seed=42",
        ")",
        "",
        "print('Membangun pairwise test samples dari 10 unseen test writers...')",
        "X_test_pairs, y_test_pairs = create_pairwise_dataset(",
        "    df_metadata, combined_cache, test_wids, pairs_per_writer=30, random_seed=42",
        ")",
        "",
        "print(f'Train Pairs: {X_train_pairs.shape}, Label distribution: {np.bincount(y_train_pairs)}')",
        "print(f'Val Pairs:   {X_val_pairs.shape}, Label distribution: {np.bincount(y_val_pairs)}')",
        "print(f'Test Pairs:  {X_test_pairs.shape}, Label distribution: {np.bincount(y_test_pairs)}')"
    ])

    # -------------------------------------------------------------
    # Training Model B (SVM)
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 7. Pelatihan Model Klasifikasi SVM (Model B)",
        "Kita menggunakan `scikit-learn` Pipeline yang terdiri atas:",
        "1. `StandardScaler`: Normalisasi mean dan varians setiap fitur pasangan.",
        "2. `LinearSVC` dengan probabilitas terkalibrasi (`CalibratedClassifierCV`): Menghasilkan estimasi probabilitas keaslian $P(y=1|\\mathbf{x})$ yang stabil dan cepat."
    ])

    add_code([
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.svm import LinearSVC",
        "from sklearn.calibration import CalibratedClassifierCV",
        "",
        "print('Melatih Linear SVM Pipeline dengan Platt Calibration...')",
        "start_time = time.time()",
        "",
        "svm_pipeline = Pipeline([",
        "    ('scaler', StandardScaler(copy=False)),",
        "    ('svm', CalibratedClassifierCV(LinearSVC(dual=False, C=1.0, max_iter=2500, random_state=42), cv=3))",
        "])",
        "",
        "svm_pipeline.fit(X_train_pairs, y_train_pairs)",
        "train_duration = time.time() - start_time",
        "print(f'Pelatihan selesai dalam {train_duration:.2f} detik!')"
    ])

    # -------------------------------------------------------------
    # Evaluation & Metrics
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 8. Evaluasi Model & Analisis Metrik",
        "Evaluasi dilakukan pada dataset pengujian yang berisi **penulis yang belum pernah dilihat sebelumnya (Unseen Test Writers)**.",
        "Metrik yang dihitung mencakup:",
        "- **Akurasi, Presisi, Recall, F1-Score**",
        "- **Equal Error Rate (EER)**: Titik di mana *False Acceptance Rate (FAR)* sama dengan *False Rejection Rate (FRR)*.",
        "- **ROC Curve & AUC Score**",
        "- **Confusion Matrix**"
    ])

    add_code([
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix",
        "",
        "def evaluate_predictions(y_true, y_probs, threshold=0.5):",
        "    y_pred = (y_probs >= threshold).astype(int)",
        "    acc = accuracy_score(y_true, y_pred)",
        "    prec = precision_score(y_true, y_pred, zero_division=0)",
        "    rec = recall_score(y_true, y_pred, zero_division=0)",
        "    f1 = f1_score(y_true, y_pred, zero_division=0)",
        "    ",
        "    fpr, tpr, thresholds = roc_curve(y_true, y_probs)",
        "    fnr = 1.0 - tpr",
        "    idx_eer = np.nanargmin(np.abs(fpr - fnr))",
        "    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)",
        "    eer_thresh = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else 0.5",
        "    roc_auc = float(auc(fpr, tpr))",
        "    cm = confusion_matrix(y_true, y_pred)",
        "    ",
        "    return {",
        "        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,",
        "        'eer': eer, 'eer_threshold': eer_thresh, 'roc_auc': roc_auc,",
        "        'cm': cm, 'fpr': fpr, 'tpr': tpr",
        "    }",
        "",
        "# Evaluasi pada Data Uji (Unseen Writers)",
        "test_probs = svm_pipeline.predict_proba(X_test_pairs)[:, 1]",
        "metrics = evaluate_predictions(y_test_pairs, test_probs, threshold=0.5)",
        "",
        "print('=== HASIL EVALUASI PADA UNSEEN TEST WRITERS ===')",
        "print(f'Accuracy:      {metrics[\"accuracy\"]*100:.2f}%')",
        "print(f'Precision:     {metrics[\"precision\"]*100:.2f}%')",
        "print(f'Recall:        {metrics[\"recall\"]*100:.2f}%')",
        "print(f'F1-Score:      {metrics[\"f1\"]*100:.2f}%')",
        "print(f'EER:           {metrics[\"eer\"]*100:.2f}%')",
        "print(f'EER Threshold: {metrics[\"eer_threshold\"]:.4f}')",
        "print(f'ROC AUC:       {metrics[\"roc_auc\"]:.4f}')"
    ])

    add_markdown([
        "### 📊 Visualisasi Kurva ROC & Confusion Matrix"
    ])

    add_code([
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))",
        "",
        "# 1. ROC Curve",
        "axes[0].plot(metrics['fpr'], metrics['tpr'], color='#1E40AF', lw=2, label=f'Linear SVM (AUC = {metrics[\"roc_auc\"]:.3f})')",
        "axes[0].plot([0, 1], [0, 1], color='#9CA3AF', linestyle='--', label='Random Guess')",
        "axes[0].set_xlabel('False Positive Rate (FAR)')",
        "axes[0].set_ylabel('True Positive Rate (1 - FRR)')",
        "axes[0].set_title('Receiver Operating Characteristic (ROC)')",
        "axes[0].legend(loc='lower right')",
        "axes[0].grid(True, alpha=0.3)",
        "",
        "# 2. Confusion Matrix",
        "cm = metrics['cm']",
        "im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)",
        "axes[1].set_title('Confusion Matrix (Threshold = 0.50)')",
        "plt.colorbar(im, ax=axes[1])",
        "tick_marks = np.arange(2)",
        "axes[1].set_xticks(tick_marks)",
        "axes[1].set_xticklabels(['Different/Forg', 'Same Writer'])",
        "axes[1].set_yticks(tick_marks)",
        "axes[1].set_yticklabels(['Different/Forg', 'Same Writer'])",
        "",
        "thresh = cm.max() / 2.",
        "for i in range(cm.shape[0]):",
        "    for j in range(cm.shape[1]):",
        "        axes[1].text(j, i, format(cm[i, j], 'd'),",
        "                     ha='center', va='center',",
        "                     color='white' if cm[i, j] > thresh else 'black')",
        "axes[1].set_ylabel('True Label')",
        "axes[1].set_xlabel('Predicted Label')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # -------------------------------------------------------------
    # Simulation: 1:1 Verification & 1:N Identification
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 9. Simulasi One-Shot Verification (1:1) & Identification (1:N)",
        "Kita mensimulasikan skenario dunia nyata di mana setiap pengguna hanya mendaftarkan 1 template tanda tangan."
    ])

    add_code([
        "from src.similarity import compute_pair_features",
        "from src.verification import verify_signature",
        "from src.identification import identify_signature",
        "",
        "# Pilih 5 pengguna dari grup test sebagai pengguna terdaftar",
        "sim_users = test_wids[:5]",
        "templates = []",
        "",
        "for uid in sim_users:",
        "    gen_path = df_metadata[(df_metadata['writer_id'] == uid) & (df_metadata['is_forgery'] == 0)]['filepath'].iloc[0]",
        "    templates.append({",
        "        'user_id': uid,",
        "        'name': f'User_{uid}',",
        "        'features': combined_cache[gen_path],",
        "        'image_path': gen_path",
        "    })",
        "",
        "print(f'Terdaftar {len(templates)} template tanda tangan (1 template per pengguna).')",
        "",
        "# Tes 1:1 Verification (Benar vs Salah)",
        "probe_true = df_metadata[(df_metadata['writer_id'] == sim_users[0]) & (df_metadata['is_forgery'] == 0)]['filepath'].iloc[1]",
        "probe_false = df_metadata[(df_metadata['writer_id'] == sim_users[1]) & (df_metadata['is_forgery'] == 0)]['filepath'].iloc[1]",
        "",
        "f_probe_true = combined_cache[probe_true]",
        "f_probe_false = combined_cache[probe_false]",
        "",
        "res_match = verify_signature(f_probe_true, templates[0]['features'], claimed_user_name=templates[0]['name'], method='svm', svm_model=svm_pipeline)",
        "res_diff = verify_signature(f_probe_false, templates[0]['features'], claimed_user_name=templates[0]['name'], method='svm', svm_model=svm_pipeline)",
        "",
        "print('=== TEST 1:1 VERIFICATION ===')",
        "print(f'1. Probe Penulis Asli -> Decision: {res_match[\"decision\"]}, Skor: {res_match[\"similarity_score\"]:.4f}')",
        "print(f'2. Probe Penulis Lain -> Decision: {res_diff[\"decision\"]}, Skor: {res_diff[\"similarity_score\"]:.4f}')",
        "",
        "# Tes 1:N Identification",
        "id_res = identify_signature(f_probe_true, templates, method='svm', svm_model=svm_pipeline)",
        "print('\\n=== TEST 1:N IDENTIFICATION ===')",
        "print(f'Keputusan: {id_res[\"decision\"]}')",
        "print(f'Teridentifikasi sebagai: {id_res[\"predicted_name\"]} (Skor: {id_res[\"top_score\"]:.4f}, Margin: {id_res[\"margin\"]:.4f})')"
    ])

    # -------------------------------------------------------------
    # Model Export
    # -------------------------------------------------------------
    add_markdown([
        "---",
        "## 10. Ekspor Model ke Disk (`.joblib`)",
        "Simpan model yang telah dilatih ke folder `data/models/` agar langsung dapat digunakan oleh aplikasi web Streamlit."
    ])

    add_code([
        "model_dir = BASE_DIR / 'data' / 'models'",
        "model_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "save_path_lin = model_dir / 'linear_svm_pipeline.joblib'",
        "save_path_best = model_dir / 'best_model.joblib'",
        "",
        "joblib.dump(svm_pipeline, save_path_lin)",
        "joblib.dump(svm_pipeline, save_path_best)",
        "",
        "print(f'✅ Model berhasil disimpan ke:')",
        "print(f'  1. {save_path_lin} (Ukuran: {save_path_lin.stat().st_size / 1024:.1f} KB)')",
        "print(f'  2. {save_path_best} (Ukuran: {save_path_best.stat().st_size / 1024:.1f} KB)')",
        "print('\\nModel siap digunakan pada aplikasi Streamlit (`app.py`)!')"
    ])

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = Path("notebooks/Signature_Recognition_Training.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Jupyter Notebook successfully created at {out_path.resolve()}!")

if __name__ == "__main__":
    create_notebook()
