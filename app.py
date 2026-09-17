import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import ast
from tensorflow.keras import ops as kops
from pathlib import Path
import cv2

# -------------------------------
# Load model and class names
# -------------------------------
def pick_model_path():
    if Path("bestmodel.cbm").exists():
        return "bestmodel.cbm"
    if Path("retinal_model.cbm").exists():
        return "retinal_model.cbm"
    if Path("retinal_model.h5").exists():
        return "retinal_model.h5"
    return None

MODEL_PATH = pick_model_path()
MODEL_TYPE = "catboost" if MODEL_PATH.endswith(".cbm") else "keras"
CATBOOST_PCA_PATH = "catboost_pca.pkl"
CATBOOST_CLASS_NAMES_PATH = "catboost_class_names.pkl"
CATBOOST_CLASS_NAMES_PATH_ALT = "catboost_clas_names.pkl"

@st.cache_resource
def load_model():
    if MODEL_TYPE == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise RuntimeError(
                "CatBoost is required to load .cbm models. "
                "Install it with: pip install catboost"
            ) from e

        cbm = CatBoostClassifier()
        cbm.load_model(MODEL_PATH)
        cbm._pca = None
        if Path(CATBOOST_PCA_PATH).exists():
            with open(CATBOOST_PCA_PATH, "rb") as f:
                cbm._pca = pickle.load(f)
        cbm._class_names = None
        class_path = None
        if Path(CATBOOST_CLASS_NAMES_PATH).exists():
            class_path = CATBOOST_CLASS_NAMES_PATH
        elif Path(CATBOOST_CLASS_NAMES_PATH_ALT).exists():
            class_path = CATBOOST_CLASS_NAMES_PATH_ALT
        if class_path:
            with open(class_path, "rb") as f:
                cbm._class_names = pickle.load(f)
        elif hasattr(cbm, "classes_"):
            cbm._class_names = list(cbm.classes_)
        cbm._has_internal_preprocess = False
        return cbm

    class TrueDivide(tf.keras.layers.Layer):
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
        @classmethod
        def from_config(cls, config):
            # Older TF/Keras versions don't accept quantization_config
            config.pop("quantization_config", None)
            return super().from_config(config)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"Dense": DenseCompat, "TrueDivide": TrueDivide},
    )
    # Detect internal preprocessing layers (e.g., TrueDivide)
    has_internal_preprocess = any(
        ("true_divide" in layer.name.lower()) or (layer.__class__.__name__ == "TrueDivide")
        for layer in model.layers
    )
    model._has_internal_preprocess = has_internal_preprocess
    return model

@st.cache_data
def load_classes():
    if MODEL_TYPE == "catboost":
        class_path = None
        if Path(CATBOOST_CLASS_NAMES_PATH).exists():
            class_path = CATBOOST_CLASS_NAMES_PATH
        elif Path(CATBOOST_CLASS_NAMES_PATH_ALT).exists():
            class_path = CATBOOST_CLASS_NAMES_PATH_ALT
        if class_path:
            with open(class_path, "rb") as f:
                return pickle.load(f)
        if hasattr(model, "_class_names") and model._class_names:
            return list(model._class_names)

    with open("class_names.pkl", "rb") as f:
        try:
            classes = pickle.load(f)
        except Exception:
            # Fallback for plain-text list stored in .pkl
            f.seek(0)
            text = f.read().decode("utf-8", errors="ignore").strip()
            classes = ast.literal_eval(text)
    return classes

model = load_model()
class_names = load_classes()

# -------------------------------
# Image preprocessing
# -------------------------------
IMG_SIZE = 300  # EfficientNetB3/VGG16 input size
CATBOOST_IMG_SIZE = 64  # Per DL_miniproject (grayscale 64x64)
BACKBONE = "efficientnetb3"  # Must match train_model.py

def preprocess_image(image):
    if MODEL_TYPE == "catboost":
        # DL_miniproject spec: grayscale 64x64, no normalization, then flatten
        image = np.array(image, dtype=np.uint8)
        # PIL gives RGB; convert to BGR to match cv2.imread + BGR2GRAY
        bgr = image[..., ::-1]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (CATBOOST_IMG_SIZE, CATBOOST_IMG_SIZE), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32)
        return gray.flatten()[None, :]

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)

    if not getattr(model, "_has_internal_preprocess", False):
        if BACKBONE == "vgg16":
            # VGG16 expects BGR with mean subtraction
            image = image[..., ::-1]
            image = image - np.array([103.939, 116.779, 123.68], dtype=np.float32)
        else:
            # EfficientNet expects inputs in [0, 1]
            image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Prediction function
# -------------------------------
def predict(image):
    processed = preprocess_image(image)
    if MODEL_TYPE == "catboost":
        if getattr(model, "_pca", None) is not None:
            processed = model._pca.transform(processed)
        preds = model.predict_proba(processed)[0]
        return preds
    preds = model.predict(processed)[0]
    return preds

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Retinal Disease Detection", layout="centered")

st.title("Retinal Disease Detection from Fundus Images")
st.write("Upload a fundus image to detect possible eye diseases.")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
show_debug = st.checkbox("Show Debug Info", value=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", width=600)
    
    if st.button("Predict"):
        preds = predict(image)

        st.subheader("Prediction Results")

        # Auto-handle softmax vs. multi-label outputs
        is_softmax = (len(preds) == len(class_names)) and (0.9 <= float(np.sum(preds)) <= 1.1)

        if is_softmax:
            top_idx = int(np.argmax(preds))
            st.write(f"Prediction: **{class_names[top_idx]}** ({preds[top_idx]:.2f})")
        else:
            threshold = 0.5
            detected = []

            for i, prob in enumerate(preds):
                if prob > threshold:
                    detected.append((class_names[i], prob))

            if len(detected) == 0:
                st.write("No disease detected with high confidence.")
            else:
                for disease, prob in detected:
                    st.write(f"Prediction: **{disease}** ({prob:.2f})")

        # Show all probabilities
        st.subheader("All Class Probabilities")
        for i, prob in enumerate(preds):
            st.write(f"{class_names[i]}: {prob:.2f}")

        if show_debug:
            st.subheader("Debug Info")
            st.write(f"Model type: {MODEL_TYPE}")
            st.write(f"Model path: {MODEL_PATH}")
            if Path(MODEL_PATH).exists():
                stat = Path(MODEL_PATH).stat()
                st.write(f"Model size (bytes): {stat.st_size}")
                st.write(f"Model modified: {stat.st_mtime}")
            st.write(f"Classes loaded: {len(class_names)}")
            st.write(f"Class order: {class_names}")

            processed = preprocess_image(image)
            st.write(f"Processed input shape: {processed.shape}")

            if MODEL_TYPE == "catboost":
                pca_loaded = getattr(model, "_pca", None) is not None
                st.write(f"PCA loaded: {pca_loaded}")
                if pca_loaded:
                    st.write(f"PCA components: {model._pca.n_components_}")
                    st.write(f"PCA input dim: {model._pca.n_features_in_}")
                st.write(f"Probs sum: {float(np.sum(preds)):.4f}")
                top_idx = int(np.argmax(preds))
                top3 = np.argsort(preds)[::-1][:3]
                st.write(f"Top-1 index: {top_idx}")
                st.write(f"Top-3 indices: {top3.tolist()}")
                top2 = top3[:2]
                if len(top2) == 2:
                    margin = float(preds[top2[0]] - preds[top2[1]])
                    st.write(f"Top-2 margin: {margin:.4f}")

            # Sanity check on dataset samples (if available)
            dataset_root = Path("datasets")
            if dataset_root.exists():
                st.write("Dataset sanity check (optional)")
                class_options = [p.name for p in dataset_root.iterdir() if p.is_dir()]
                if class_options:
                    selected_class = st.selectbox("Pick a class", class_options)
                    if st.button("Run Sanity Check"):
                        files = list((dataset_root / selected_class).glob("*"))
                        if files:
                            sample_path = files[0]
                            sample_img = Image.open(sample_path).convert("RGB")
                            sample_preds = predict(sample_img)
                            st.write(f"Expected class: {selected_class}")
                            top_idx = int(np.argmax(sample_preds))
                            st.write(f"Predicted class: {class_names[top_idx]}")
                            st.write(f"Expected index (from class order): {class_names.index(selected_class) if selected_class in class_names else 'not found'}")
                        else:
                            st.write("No files found in that class folder.")
