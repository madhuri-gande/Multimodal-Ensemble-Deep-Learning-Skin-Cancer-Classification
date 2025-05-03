import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf

# === Load Models ===
densenet_model = tf.keras.models.load_model('models/densenet_multimodal_model.keras')
efficientnet_model = tf.keras.models.load_model('models/efficientnetv2_multimodal_model.keras')
inceptionresnet_model = tf.keras.models.load_model('models/inceptionresnetv2_multimodal_model.keras')
meta_model = joblib.load('models/meta_model.pkl')

# === Helper Functions ===
def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # Shape: (1, 224, 224, 3)

def preprocess_metadata(df):
    # Debugging: Show original metadata columns
    #st.write("Original Metadata Columns:", df.columns)
    
    # Drop non-numeric and label columns
    drop_cols = ['patient_id', 'lesion_id', 'img_id', 'image_path', 'diagnostic_grouped']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Debugging: Show remaining columns after dropping
    #st.write("Remaining Columns after Drop:", df.columns)
    
    return df.values.astype(np.float32)  # Convert to numeric type

def predict(image, metadata):
    # Predict using each model (DenseNet, EfficientNetV2, InceptionResNetV2)
    pred1 = densenet_model.predict([image, metadata])[0]
    pred2 = efficientnet_model.predict([image, metadata])[0]
    pred3 = inceptionresnet_model.predict([image, metadata])[0]

    # Concatenate predictions from all three models and reshape for final meta-model prediction
    stacked_input = np.concatenate([pred1, pred2, pred3]).reshape(1, -1)
    final_pred = meta_model.predict(stacked_input)[0]
    final_proba = meta_model.predict_proba(stacked_input)[0]

    return final_pred, final_proba

def map_label(pred):
    return "Cancer (0)" if pred == 0 else "Disease (1)"

# === Streamlit UI ===
st.set_page_config(page_title="Skin Cancer Prediction", layout="centered")
st.title("Multimodal Skin Cancer Predictor")
st.markdown("Upload an **image** and a **CSV with one row of metadata** to classify as **Cancer(0) or Disease(1)**.")

# File upload options
uploaded_image = st.file_uploader("Upload Skin Image", type=["jpg", "png"])
uploaded_csv = st.file_uploader("Upload Metadata (1-row CSV)", type=["csv"])

if uploaded_image and uploaded_csv:
    try:
        # Preprocess image
        image = preprocess_image(uploaded_image)
        
        # Read the uploaded CSV file
        metadata_df = pd.read_csv(uploaded_csv)

        # Ensure there's exactly one row in the CSV file
        if metadata_df.shape[0] != 1:
            st.error("Please upload a CSV with exactly one row of metadata.")
        else:
            # Preprocess metadata
            metadata = preprocess_metadata(metadata_df)

            # Get the prediction
            pred_class, pred_proba = predict(image, metadata)
            label = map_label(pred_class)

            # Display the results
            st.image(uploaded_image, caption="Uploaded Skin Image", width=300)
            st.write("Metadata Preview:")
            st.dataframe(metadata_df)

            st.subheader(f"🩺 Predicted Diagnosis: **{label}**")
            st.markdown(f"Confidence (Cancer vs Disease): `{pred_proba[0]:.2f} / {pred_proba[1]:.2f}`")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
