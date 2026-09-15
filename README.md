# Retinal Disease Detection

A machine-learning project that classifies retinal fundus images into one of eight eye-health categories. The core inference pipeline uses a CatBoost multiclass classifier with PCA-reduced grayscale image features, served through an interactive Streamlit interface.

> **Medical disclaimer:** This is an educational screening prototype, not a clinical diagnostic device. Do not use its predictions as medical advice or for treatment decisions.

## What it does

- Upload a JPG or PNG fundus image.
- Preprocess the image to match the trained CatBoost model.
- Show the highest-probability class and confidence score.
- Display probabilities for every supported class.
- Provide optional debug information for troubleshooting preprocessing and model assets.

## Supported classes

The included model uses the following class order:

| Label | Description |
| --- | --- |
| `normal` | Normal |
| `hypertension` | Hypertensive retinal finding |
| `cataract` | Cataract |
| `others` | Other retinal finding |
| `glaucoma` | Glaucoma |
| `myopia` | Myopia |
| `ageDegeneration` | Age-related macular degeneration |
| `diabetes` | Diabetic retinal finding |

## Inference pipeline

```text
Fundus image
    -> RGB to grayscale conversion
    -> resize to 64 × 64 pixels
    -> flatten into 4,096 features
    -> PCA transformation (95% explained-variance target)
    -> CatBoost multiclass prediction
    -> class probabilities
```

The preprocessing and PCA artifact must come from the same training run as the CatBoost model.

## Repository structure

```text
.
├── app.py                     # Streamlit prediction interface
├── bestmodel.cbm              # Trained CatBoost model (local, git-ignored)
├── catboost_pca.pkl           # Fitted PCA transformer
├── catboost_class_names.pkl   # Ordered class names
├── DL_miniproject.ipynb       # Dataset preparation, training, and evaluation
├── datasets/                  # Local class-folder image dataset (git-ignored)
├── frontend/                  # Separate React/Vite UI prototype
└── requirements.txt           # Python dependencies
```

## Run locally

### Prerequisites

- Python 3.10 or a compatible TensorFlow-supported Python version
- `bestmodel.cbm`, `catboost_pca.pkl`, and `catboost_class_names.pkl` in the project root

### Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start the app

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, upload a fundus image, and select **Predict**.

## Training workflow

`DL_miniproject.ipynb` records the CatBoost baseline workflow:

1. Downloads the `tanjemahamed/odir5k-classification` dataset from Kaggle.
2. Creates an 80/20 stratified train-test split.
3. Converts images to grayscale `64 × 64` arrays.
4. Uses PCA to retain 95% of explained variance.
5. Trains a CatBoost multiclass classifier for 1,000 iterations.
6. Saves the trained model as `bestmodel.cbm`.

The notebook’s saved run reports 12,784 images across eight classes, 88.11% test accuracy, and a weighted F1 score of 0.8817. These are historical experiment outputs—not clinical-validation results—and may vary by dataset version, environment, and training run. Because the data is class-imbalanced, evaluate per-class precision, recall, F1 scores, and the confusion matrix in addition to overall accuracy.

## Frontend note

`frontend/` is a separate React/Vite prototype that expects a `POST /predict` API endpoint. That FastAPI endpoint is not included in this repository; use the Streamlit application for the working local interface.

## Responsible use

Before any clinical use, a medical-image model would require independent external validation, calibration, fairness and bias analysis, privacy safeguards, and applicable regulatory review.
