# Fast Food Image Classification

Sistem klasifikasi gambar 6 kategori makanan cepat saji menggunakan Convolutional Neural Network (CNN) dengan transfer learning MobileNetV2, dilengkapi pipeline skrip modular dan ekspor model ke SavedModel, TensorFlow Lite, dan TensorFlow.js.

## Ringkasan Proyek

Model dilatih pada 12.000 gambar dunia nyata beresolusi tidak seragam dari dataset Kaggle [`utkarshsaxenadn/fast-food-classification-dataset`](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset) (`Fast Food Classification V2`).

- **Jumlah kelas:** 6 (`Baked Potato`, `Burger`, `Crispy Chicken`, `Donut`, `Fries`, `Pizza`)
- **Total citra:** 12.000 gambar (2.000 gambar per kelas) dengan 474 variasi resolusi unik
- **Pembagian data:** 80% Train (9.600), 10% Validation (1.200), 10% Test (1.200) dengan stratifikasi per kelas (`seed=42`)
- **Backbone:** MobileNetV2 (bobot ImageNet, frozen) + custom head (`Conv2D`, `MaxPooling2D`, `GlobalAveragePooling2D`, `Dense`, `Dropout`)
- **Format ekspor:** TensorFlow SavedModel, TensorFlow Lite (`model.tflite` + `label.txt`), dan TensorFlow.js (`model.json` + `group1-shard1of1.bin`)

## Dataset

Data mentah dari subfolder `Train`, `Valid`, dan `Test` digabung per kelas, diverifikasi integritasnya menggunakan Pillow, lalu dibagi ulang secara seimbang (80:10:10).

| Kelas | Jumlah Gambar | Variasi Resolusi Unik | Contoh Resolusi (W x H) |
| :--- | ---: | ---: | :--- |
| Baked Potato | 2.000 | 194 | 178x283, 204x247, 225x224 |
| Burger | 2.000 | 218 | 433x116, 206x244, 315x160 |
| Crispy Chicken | 2.000 | 206 | 178x283, 205x246, 315x160 |
| Donut | 2.000 | 292 | 920x920, 736x736, 176x286 |
| Fries | 2.000 | 207 | 176x286, 204x247, 243x208 |
| Pizza | 2.000 | 227 | 322x157, 178x283, 205x246 |
| **Total** | **12.000** | **474 (keseluruhan)** | Non-uniform |

Augmentasi data (`rotation_range=25`, `width_shift_range=0.15`, `height_shift_range=0.15`, `shear_range=0.15`, `zoom_range=0.15`, `horizontal_flip=True`) hanya diterapkan pada data training. Validation dan test set hanya melalui normalisasi piksel `1./255` serta resize ke `224x224`.

## Arsitektur Model & Konfigurasi Pelatihan

Model disusun dengan Keras `Sequential` di atas backbone `MobileNetV2` (`input_shape=(224, 224, 3)`, `include_top=False`, `weights="imagenet"`, `trainable=False`):

| Layer | Output Shape | Jumlah Parameter |
| :--- | :--- | ---: |
| `MobileNetV2` (frozen) | `(None, 7, 7, 1280)` | 2.257.984 |
| `Conv2D(128, 3x3, relu, same)` | `(None, 7, 7, 128)` | 1.474.688 |
| `MaxPooling2D(2x2)` | `(None, 3, 3, 128)` | 0 |
| `GlobalAveragePooling2D` | `(None, 128)` | 0 |
| `Dense(256, relu)` | `(None, 256)` | 33.024 |
| `Dropout(0.4)` | `(None, 256)` | 0 |
| `Dense(6, softmax)` | `(None, 6)` | 1.542 |

- **Total parameter:** 3.767.238 (14,37 MB)
- **Trainable parameter:** 1.509.254 (5,76 MB)
- **Non-trainable parameter:** 2.257.984 (8,61 MB)
- **Optimizer & Loss:** Adam (`learning_rate=1e-3`), `categorical_crossentropy`
- **Callbacks:**
  - `ThresholdCallback(target_acc=0.95)`: menghentikan pelatihan lebih awal jika `accuracy` dan `val_accuracy` sama-sama mencapai >=95%.
  - `ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6)`: menurunkan learning rate ketika `val_loss` stagnan.

## Hasil Evaluasi & Artefak Model

Hasil yang tercatat pada eksperimen di [notebook.ipynb](notebook.ipynb):

- **Kurva pelatihan (15 epoch):** akurasi training mencapai ~94,7% dan akurasi validasi stabil di ~93,2%.
- **Evaluasi Test Set (1.200 gambar):** **Test Accuracy 90,92%** dengan **Test Loss 0,3705**.
- **Uji inferensi TFLite (5 sampel acak Test Set):** 5 dari 5 prediksi benar (Donut 100,0%, Burger 95,0%, Crispy Chicken 97,1%, Baked Potato 71,2%, Pizza 99,8%).

Artefak model yang tersedia di folder `models/`:

| Format | Lokasi | Berkas Utama | Ukuran Total | Target Penggunaan |
| :--- | :--- | :--- | ---: | :--- |
| SavedModel | `models/saved_model/` | `saved_model.pb`, `variables/` | ~15,3 MB | Server / TensorFlow Serving |
| TensorFlow Lite | `models/tflite/` | `model.tflite`, `label.txt` | ~3,84 MB | Mobile, edge, dan CLI lokal |
| TensorFlow.js | `models/tfjs_model/` | `model.json`, `group1-shard1of1.bin` | ~14,7 MB | Aplikasi web / browser |

## Struktur Direktori

```text
fast-food-image-classification/
├── .gitignore
├── README.md
├── requirements.txt
├── notebook.ipynb              # Notebook eksperimen lengkap beserta output eksekusi
├── src/
│   ├── data_preparation.py     # Unduh dataset via kagglehub, verifikasi resolusi, & split 80:10:10
│   ├── train.py                # Definisi arsitektur model, callbacks, training, evaluasi, & ekspor model
│   └── inference.py            # CLI prediksi gambar baru menggunakan model TFLite
└── models/
    ├── saved_model/            # Model format TensorFlow SavedModel
    ├── tflite/                 # Model TFLite teroptimasi + label.txt
    └── tfjs_model/             # Model format TensorFlow.js (model.json + shard .bin)
```

## Prasyarat & Instalasi

- Python 3.10 atau lebih baru
- Akun dan kredensial Kaggle aktif untuk mengunduh dataset lewat `kagglehub` (hanya dibutuhkan saat menjalankan `src/data_preparation.py`)
- GPU CUDA direkomendasikan untuk menjalankan pelatihan ulang

Buat virtual environment dan pasang dependensi:

```bash
python -m venv .venv
```

Aktifkan virtual environment:

- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

Pasang seluruh paket yang dibutuhkan:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### 1. Persiapan Dataset

Mengunduh dataset dari Kaggle via `kagglehub`, memverifikasi gambar yang valid, mencetak ringkasan variasi resolusi, lalu membaginya menjadi `train/`, `val/`, dan `test/` (80:10:10):

```bash
python src/data_preparation.py --raw-dir data/raw_dataset --split-dir data/dataset_split --seed 42
```

Jika dataset `Fast Food Classification V2` sudah tersedia secara lokal, gunakan `--source-dir`:

```bash
python src/data_preparation.py --source-dir /path/to/"Fast Food Classification V2"
```

### 2. Pelatihan, Evaluasi, & Ekspor Model

Melatih model dari hasil split dataset, menyimpan kurva akurasi/loss ke berkas gambar, mengevaluasi pada test set, dan mengekspor ketiga format model ke `models/`:

```bash
python src/train.py \
  --data-dir data/dataset_split \
  --models-dir models \
  --epochs 15 \
  --batch-size 32 \
  --lr 0.001 \
  --target-acc 0.95 \
  --plot-path training_history.png
```

### 3. Inferensi CLI dengan Model TFLite

Menjalankan prediksi pada satu berkas gambar atau seluruh gambar dalam satu folder menggunakan `models/tflite/model.tflite`:

```bash
python src/inference.py path/to/sample_pizza.png
```

Opsi tambahan:

- `--top-k 3`: jumlah prediksi teratas yang ditampilkan (default `3`)
- `--model models/tflite/model.tflite`: path model TFLite kustom
- `--labels models/tflite/label.txt`: path file label kustom
- `--json`: menampilkan output dalam format JSON

Contoh output CLI (hasil uji pada sampel gambar `Pizza`):

```text
Image: sample_pizza.png
  1. Pizza             99.82%
  2. Baked Potato       0.12%
  3. Donut              0.03%
```

### 4. Notebook Eksperimen

Seluruh alur eksplorasi, pelatihan di GPU, plot akurasi/loss, konversi model, dan uji inferensi terdokumentasi di [notebook.ipynb](notebook.ipynb).

> **Catatan sebelum publikasi:** Uji ulang langkah instalasi (`pip install -r requirements.txt` dan `python src/inference.py`) di direktori bersih sebelum memublikasikan repositori ke GitHub.
