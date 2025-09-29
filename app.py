import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------- Load Model --------------------
@st.cache_resource
def load_model(model_path):
    """Load a Keras model and cache it."""
    return tf.keras.models.load_model(model_path)

# Change this path to your trained model
model_path = "/Users/apple/Documents/Projects/fish image classification/images.cv_jzk6llhf18tm3k0kyttxz/data/mobilenet_trained_model.h5"
model = load_model(model_path)

# -------------------- Class Names --------------------
# Replace with your own class labels
class_names = [
    'fish sea_food trout', 'fish sea_food striped_red_mullet', 'fish sea_food shrimp',
    'fish sea_food sea_bass', 'fish sea_food red_sea_bream', 'fish sea_food red_mullet',
    'fish sea_food hourse_mackerel', 'fish sea_food gilt_head_bream',
    'fish sea_food black_sea_sprat', 'animal fish', 'animal fish bass'
]

# -------------------- Streamlit UI --------------------
st.markdown(
    "<h1 style='text-align: center; color: orange;'>FISH CLASSIFIER</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='color: #8B4513;'>Upload a fish image to predict its species</h3>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # -------------------- Preprocess Image --------------------
    # Automatically match model input size
    input_h, input_w = model.input_shape[1:3]
    image_resized = image.resize((input_w, input_h))
    image_array = np.array(image_resized) / 255.0  # normalize
    image_batch = np.expand_dims(image_array, axis=0)  # add batch dimension

    # -------------------- Prediction --------------------
    prediction = model.predict(image_batch)[0]
    pred_index = np.argmax(prediction)
    confidence = prediction[pred_index]

    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {class_names[pred_index]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # -------------------- Confidence Scores --------------------
    st.subheader("Confidence Scores (Top Predictions)")
    top_indices = np.argsort(prediction)[::-1]  # sort descending
    for i in top_indices:
        st.write(f"{class_names[i]}: {prediction[i]:.2f}")
