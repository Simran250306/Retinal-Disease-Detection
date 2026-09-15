"""Streamlit inference app for CNN and CatBoost retinal-image classifiers."""

from pathlib import Path
import ast
import pickle

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import ops as kops

CNN_IMAGE_SIZE = 224
CATBOOST_IMAGE_SIZE = 64


def available_models():
    """Return locally available models, with the CNN presented first."""
    candidates = [
        ("CNN (recommended)", Path("retinal_cnn.keras"), "cnn"),
        ("CNN (legacy H5)", Path("retinal_cnn.h5"), "cnn"),
        ("CatBoost baseline", Path("bestmodel.cbm"), "catboost"),
        ("CatBoost baseline (alternate)", Path("retinal_model.cbm"), "catboost"),
        ("Keras model (alternate)", Path("retinal_model.h5"), "cnn"),
    ]
    return [item for item in candidates if item[1].exists()]


class TrueDivide(tf.keras.layers.Layer):
    """Compatibility layer for older exported Keras models."""

    @classmethod
    def from_config(cls, config):
        return cls()

    def call(self, inputs, y=None):
        if y is not None:
            return kops.divide(inputs, y)
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return kops.divide(inputs[0], inputs[1])
        return kops.divide(inputs, 255.0)


class DenseCompat(tf.keras.layers.Dense):
    """Compatibility layer for models saved by older Keras versions."""

    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


@st.cache_resource
def load_model(model_path, model_type):
    if model_type == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(model_path)
        with open("catboost_pca.pkl", "rb") as file:
            model._pca = pickle.load(file)
        return model

    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"Dense": DenseCompat, "TrueDivide": TrueDivide},
    )


@st.cache_data
def load_class_names(model_type):
    candidates = (
        ["catboost_class_names.pkl", "catboost_clas_names.pkl"]
        if model_type == "catboost"
        else ["cnn_class_names.pkl", "class_names.pkl"]
    )
    for filename in candidates:
        path = Path(filename)
        if path.exists():
            with path.open("rb") as file:
                try:
                    return list(pickle.load(file))
                except Exception:
                    file.seek(0)
                    return list(ast.literal_eval(file.read().decode("utf-8").strip()))
    raise FileNotFoundError(f"No class-name file found. Expected one of: {', '.join(candidates)}")


def preprocess_image(image, model_type):
    if model_type == "catboost":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (CATBOOST_IMAGE_SIZE, CATBOOST_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32).reshape(1, -1)

    resized = image.convert("RGB").resize((CNN_IMAGE_SIZE, CNN_IMAGE_SIZE))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(image, model, model_type):
    features = preprocess_image(image, model_type)
    if model_type == "catboost":
        return model.predict_proba(model._pca.transform(features))[0]
    return model.predict(features, verbose=0)[0]


st.set_page_config(page_title="Retinal Disease Detection", page_icon="👁️", layout="centered")
st.title("Retinal Disease Detection")
st.caption("CNN-based fundus-image classification with a CatBoost baseline for comparison.")

models = available_models()
if not models:
    st.error(
        "No model file found. Train the CNN with `python train_cnn.py` or place the CatBoost "
        "artifacts (`bestmodel.cbm`, `catboost_pca.pkl`) in the project root."
    )
    st.stop()

labels = [name for name, _, _ in models]
selected_name = st.selectbox("Prediction model", labels)
_, selected_path, selected_type = next(item for item in models if item[0] == selected_name)

try:
    model = load_model(str(selected_path), selected_type)
    class_names = load_class_names(selected_type)
except Exception as error:
    st.error(f"Could not load {selected_path.name}: {error}")
    st.stop()

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])
show_debug = st.checkbox("Show debug information")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded fundus image", use_container_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            probabilities = predict(image, model, selected_type)

        if len(probabilities) != len(class_names):
            st.error("The model output does not match the number of class labels.")
            st.stop()

        top_index = int(np.argmax(probabilities))
        st.success(f"Top prediction: **{class_names[top_index]}** ({probabilities[top_index]:.1%})")
        st.subheader("Class probabilities")
        st.bar_chart({label: float(probability) for label, probability in zip(class_names, probabilities)})

        if show_debug:
            st.subheader("Debug information")
            st.write(f"Model: `{selected_path.name}` ({selected_type})")
            st.write(f"Input shape: {preprocess_image(image, selected_type).shape}")
            st.write(f"Probability sum: {float(np.sum(probabilities)):.4f}")
