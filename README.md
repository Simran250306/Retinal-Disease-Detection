# Retinal Disease Detection (Streamlit)

This project runs a Streamlit app (`app.py`) for retinal disease detection from fundus images.  
It supports two model types:
- **CatBoost (`.cbm`)** with optional PCA preprocessing.
- **Keras (`.h5`)** with optional internal preprocessing layers.

## Quick Start

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Launch the app:
   ```powershell
   streamlit run app.py
   ```

## Files Used by `app.py`

Required/optional model assets in the project root:
- `bestmodel.cbm` or `retinal_model.cbm` (CatBoost model), or `retinal_model.h5` (Keras model)
- `catboost_pca.pkl` (optional; used to match CatBoost training pipeline)
- `catboost_class_names.pkl` or `catboost_clas_names.pkl` (class names for CatBoost)
- `class_names.pkl` (class names for Keras)

## How `app.py` Works

- **Model selection**: prefers `.cbm` if present, otherwise falls back to `.h5`.
- **Image preprocessing**:
  - CatBoost: converts to grayscale 64x64 and flattens; then applies PCA if available.
  - Keras: resizes to 300x300; normalizes based on backbone.
- **Prediction**:
  - Softmax models show top-1.
  - Multi-label models show all classes above threshold.

## Notes

- If you see a `ModuleNotFoundError: No module named 'sklearn'`, install scikit-learn:
  ```powershell
  pip install scikit-learn
  ```

