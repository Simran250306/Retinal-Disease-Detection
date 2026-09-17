# Retinal Disease Detection

This project is a Streamlit app that analyzes fundus images of the eye and predicts the most likely retinal condition from a trained machine-learning model.

In simple terms: you upload an eye image, the app preprocesses it to match the model it finds, and then it shows the prediction plus the probability for every class.

## What the app does

- Accepts a fundus image uploaded by the user.
- Runs the image through a trained retinal-disease model.
- Displays the most likely diagnosis.
- Shows class-wise probabilities so the result is easier to interpret.
- Supports both single-label and multi-label style outputs, depending on the model.

## Supported model types

The app automatically chooses the first model it finds in this order:

1. `bestmodel.cbm`
2. `retinal_model.cbm`
3. `retinal_model.h5`

### CatBoost model

- Uses a grayscale `64x64` image.
- Flattens the image before prediction.
- Optionally applies PCA if `catboost_pca.pkl` is available.
- Uses stored class names from:
  - `catboost_class_names.pkl`
  - `catboost_clas_names.pkl`

### Pipeline model

This project also supports a pipeline-style CatBoost workflow:

- image preprocessing happens first,
- optional PCA is applied next,
- CatBoost makes the final prediction.

In other words, the "pipeline model" is the CatBoost model plus its preprocessing steps bundled together at inference time.

### Keras model

- Resizes the image to `300x300`.
- Normalizes the image before prediction unless the model already contains internal preprocessing.
- Uses class names from `class_names.pkl`.

## How prediction works

1. The user uploads a retina image in the app.
2. The app converts the image into the format expected by the selected model.
3. The model returns class probabilities.
4. The app either:
   - shows the top class for softmax-style outputs, or
   - lists every class above the confidence threshold for multi-label outputs.

## Project files

- `app.py` - Streamlit application and prediction logic.
- `bestmodel.cbm` - primary CatBoost model, if present.
- `retinal_model.cbm` - alternate CatBoost model, if present.
- `retinal_model.h5` - alternate Keras model, if present.
- `catboost_pca.pkl` - optional PCA object used with the CatBoost pipeline.
- `catboost_class_names.pkl` / `catboost_clas_names.pkl` - CatBoost class labels.
- `class_names.pkl` - class labels for the Keras model.
- `requirements.txt` - Python dependencies.

## Setup

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Run the app

```powershell
streamlit run app.py
```

## What the user sees

When the app runs, the interface lets the user:

- upload a fundus image,
- preview the uploaded image,
- click Predict,
- view the predicted disease class,
- inspect all class probabilities,
- optionally open debug information.

## Notes

- This project is meant for screening and demonstration, not as a replacement for a medical diagnosis.
- The prediction quality depends on the model file and class-label files available in the project root.
- If `catboost` or `scikit-learn` is missing, install the dependencies listed in `requirements.txt`.

## About the code

The app is built around `app.py`, which handles:

- model loading,
- image preprocessing,
- prediction,
- Streamlit UI rendering.

If you want, I can also add a screenshot section or a short "How it works" diagram to make the README even easier to understand.
