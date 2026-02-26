import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Long Hair Detection", layout="centered")

st.title("💇 Long Hair Detection System")
st.write("Age + Gender + Hair Length Detection")

# Load models
@st.cache_resource
def load_models():
    age_model = load_model("face_age.h5")
    gender_model = load_model("face_gender.h5")
    return age_model, gender_model

age_model, gender_model = load_models()

# Preprocess function
def preprocess_face(face_img):
    face = cv2.resize(face_img, (128, 128))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Age prediction
def predict_face_age(face_img):
    processed = preprocess_face(face_img)
    prediction = age_model.predict(processed)
    age_index = np.argmax(prediction)
    
    age_ranges = ["0-10", "11-20", "20–30", "31-40", "41-50", "51+"]
    return age_ranges[age_index], float(np.max(prediction))

# Gender prediction
def predict_face_gender(face_img):
    processed = preprocess_face(face_img)
    prediction = gender_model.predict(processed)
    return "Male" if prediction[0][0] > 0.5 else "Female"

# Hair detection heuristic
def detect_hair_length(face_img):
    h, w, _ = face_img.shape
    lower_half = face_img[h//2:, :]
    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < 80)
    return "Long" if dark_pixels > 0.25 * gray.size else "Short"

# Combined logic
def classify_person(face_img):
    age_range, age_conf = predict_face_age(face_img)
    predicted_gender = predict_face_gender(face_img)
    hair_length = detect_hair_length(face_img)

    if age_range == "20–30":
        final_gender = "Female" if hair_length == "Long" else "Male"
    else:
        final_gender = predicted_gender

    return age_range, final_gender, hair_length, age_conf


# Upload section
uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    age, gender, hair, conf = classify_person(image)

    st.subheader("🔍 Prediction Results")
    st.write(f"**Age Range:** {age}")
    st.write(f"**Hair Length:** {hair}")
    st.write(f"**Final Gender:** {gender}")
    st.write(f"**Age Confidence:** {conf:.2f}")