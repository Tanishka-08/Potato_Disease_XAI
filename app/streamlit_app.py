import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.lime_utils import generate_lime
from src.shap_utils import generate_shap
from data.disease_info import DISEASE_INFO

st.set_page_config(
    page_title="Potato Leaf Disease Detection",
    page_icon="🥔",
    layout="centered"
)

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_my_model():
    return load_model("models/efficientnet_model.h5")

model = load_my_model()

class_names = ["early_blight", "healthy", "late_blight"]

st.title("🥔 Potato Leaf Disease Detection")

st.markdown("""
Deep Learning-based potato leaf disease classification using **EfficientNetB0**
with **LIME** and **SHAP Explainable AI**.

Upload a potato leaf image below to begin prediction.
""")

st.divider()

explanation_type = st.radio(
    "Select Explainability Method",
    ["LIME", "SHAP", "Both"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Choose a potato leaf image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    img = img.resize((224,224))

    img_array = image.img_to_array(img)
    original_img = img_array.copy()

    input_img = np.expand_dims(img_array, axis=0)
    input_img = preprocess_input(input_img)

    prediction = model.predict(input_img, verbose=0)

    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    disease_key = class_names[predicted_class]
    disease = DISEASE_INFO[disease_key]

    st.divider()

    left, right = st.columns([1.5,1])

    with left:
        st.image(
            img,
            caption="Uploaded Potato Leaf",
            use_container_width=True
        )

    with right:
        st.subheader("🩺 Prediction Result")

        st.success(f"**Disease**\n\n{disease['name']}")

        st.info(f"**Confidence**\n\n{confidence:.2f}%")

        st.markdown("### 🧠 Explainability")
        st.write(f"**Selected Method:** {explanation_type}")

    st.divider()

    if explanation_type == "LIME":

        st.subheader("🧠 LIME Explanation")
        st.caption("Yellow highlighted regions indicate the areas that most influenced the prediction.")

        with st.spinner("Generating LIME explanation..."):
            lime_fig = generate_lime(
                model=model,
                img_array=original_img,
                predicted_class=predicted_class
            )

        st.pyplot(lime_fig)
        st.success("LIME explanation generated successfully!")

    elif explanation_type == "SHAP":

        st.subheader("📊 SHAP Explanation")
        st.caption("🔴 Red regions contribute positively while 🔵 Blue regions contribute negatively.")

        with st.spinner("Generating SHAP explanation..."):
            shap_path = generate_shap(
                model=model,
                img_array=original_img
            )

        st.image(
            shap_path,
            caption="SHAP Explanation",
            use_container_width=True
        )

        st.success("SHAP explanation generated successfully!")

    else:

        st.subheader("🧠 LIME Explanation")
        st.caption("Yellow highlighted regions indicate the areas that most influenced the prediction.")

        with st.spinner("Generating LIME explanation..."):
            lime_fig = generate_lime(
                model=model,
                img_array=original_img,
                predicted_class=predicted_class
            )

        st.pyplot(lime_fig)

        st.divider()

        st.subheader("📊 SHAP Explanation")
        st.caption("🔴 Red regions contribute positively while 🔵 Blue regions contribute negatively.")

        with st.spinner("Generating SHAP explanation..."):
            shap_path = generate_shap(
                model=model,
                img_array=original_img
            )

        st.image(
            shap_path,
            caption="SHAP Explanation",
            use_container_width=True
        )

    st.divider()

    st.subheader("🌿 Disease Details")

    st.markdown(f"### {disease['name']}")

    st.markdown("#### 🦠 Cause")
    st.write(disease["cause"])

    st.markdown("#### 🍂 Symptoms")
    for symptom in disease["symptoms"]:
        st.markdown(f"🔸 {symptom}")

    st.markdown("#### ✅ Recommendation")
    for recommendation in disease["recommendation"]:
        st.markdown(f"✅ {recommendation}")

    st.markdown("""
---
<div style='text-align:center;color:gray'>
<h4>🥔 Potato Leaf Disease Detection</h4>

Developed using <b>EfficientNetB0</b> • <b>TensorFlow</b> •
<b>LIME</b> • <b>SHAP</b> • <b>Streamlit</b>

<br><br>
© 2026 Final Year Project
</div>
""", unsafe_allow_html=True)
