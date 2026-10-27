import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Recognition - MobileNetV2",
    page_icon="🌿",
    layout="wide"
)

# 38 Class labels
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Cache model so it loads only once into RAM
@st.cache_resource
def load_disease_model():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(model_dir, "mobilenet_v2_plant_disease.keras"),
        os.path.join(model_dir, "trained_plant_disease_model.keras"),
        "mobilenet_v2_plant_disease.keras",
        "trained_plant_disease_model.keras"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                # Check for legacy HDF5 header
                with open(path, "rb") as f:
                    header = f.read(4)
                if header.startswith(b"\x89HDF"):
                    import shutil
                    temp_h5 = path + ".temp.h5"
                    shutil.copyfile(path, temp_h5)
                    model = tf.keras.models.load_model(temp_h5)
                    try:
                        model.save(path)
                    except Exception:
                        pass
                    if os.path.exists(temp_h5):
                        os.remove(temp_h5)
                    return model, path
                
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                st.warning(f"Note: Error loading {path}: {e}")
                continue
    return None, None

def format_class_name(raw_name):
    """Converts raw dataset class name like 'Tomato___Early_blight' to human-friendly text."""
    parts = raw_name.split("___")
    plant = parts[0].replace("_", " ").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    return f"{plant} - {condition.title()}"

def model_prediction(test_image, model):
    # Detect target image shape dynamically (224x224 for MobileNetV2, 128x128 for Custom CNN)
    if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) >= 3:
        target_size = (model.input_shape[1], model.input_shape[2])
    else:
        target_size = (224, 224)
        
    image = Image.open(test_image).convert("RGB")
    image = image.resize(target_size)
    input_arr = np.array(image, dtype=np.float32)
    input_arr = np.expand_dims(input_arr, axis=0)
    
    predictions = model.predict(input_arr, verbose=0)[0]
    result_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100.0
    return result_index, confidence

# Sidebar navigation
st.sidebar.title("Dashboard")
st.sidebar.caption("🚀 Model Engine: **MobileNetV2 (Transfer Learning)**")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ----------------- HOME PAGE -----------------
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.subheader("Powered by MobileNetV2 Deep Learning")
    
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "home_page.jpeg")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌿🔍
    
    This application utilizes **MobileNetV2 Transfer Learning** to quickly and accurately identify 38 different plant leaf health conditions and diseases from images.

    ### 🌟 Key Advantages of MobileNetV2
    - **Ultra Lightweight:** Only ~24 MB model size (compared to ~90 MB for standard CNNs).
    - **High Accuracy:** Pre-trained on ImageNet and fine-tuned for plant disease detection.
    - **Fast Inference:** Optimized depthwise separable convolutions deliver predictions in milliseconds.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload a leaf image.
    2. **Analysis:** The MobileNetV2 neural network analyzes image features at $224 \\times 224$ resolution.
    3. **Results:** View immediate disease prediction and confidence percentage.

    ### Get Started
    Select **Disease Recognition** in the sidebar to begin!
    """)

# ----------------- ABOUT PAGE -----------------
elif app_mode == "About":
    st.header("About the Model & Dataset")
    st.markdown("""
    #### 🧠 MobileNetV2 Architecture
    - **Base:** MobileNetV2 (Pre-trained on ImageNet)
    - **Architecture Highlights:** Inverted Residuals & Linear Bottlenecks, Depthwise Separable Convolutions
    - **Input Resolution:** $224 \\times 224 \\times 3$ RGB
    - **Parameters:** ~2.3 Million (Ultra-compact ~24 MB weights)
    
    #### 📊 About the Dataset
    This model was trained and evaluated on the **New Plant Diseases Dataset (PlantVillage)**:
    - **Total Images:** ~87,000 augmented leaf images
    - **Categories:** 38 distinct plant and disease classes
    - **Split:** 80% Training, 20% Validation
    
    #### 🌱 Supported Plant Categories (38 Classes):
    - **Apple:** Scab, Black Rot, Cedar Apple Rust, Healthy
    - **Blueberry:** Healthy
    - **Cherry:** Powdery Mildew, Healthy
    - **Corn (Maize):** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
    - **Grape:** Black Rot, Esca (Black Measles), Leaf Blight, Healthy
    - **Orange:** Huanglongbing (Citrus Greening)
    - **Peach:** Bacterial Spot, Healthy
    - **Pepper (Bell):** Bacterial Spot, Healthy
    - **Potato:** Early Blight, Late Blight, Healthy
    - **Raspberry:** Healthy
    - **Soybean:** Healthy
    - **Squash:** Powdery Mildew
    - **Strawberry:** Leaf Scorch, Healthy
    - **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy
    """)

# ----------------- DISEASE RECOGNITION PAGE -----------------
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    model, model_path = load_disease_model()
    if model is None:
        st.error("⚠️ Model file not found! Please run `python download_model.py` or `run.bat` to download the pre-trained MobileNetV2 weights.")
    else:
        model_name = os.path.basename(model_path)
        st.caption(f"Active Model: `{model_name}` (MobileNetV2)")
        st.info("💡 Tip: Test images are available in the `sample_test_images/` directory.")
        
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns([1, 1])
        
        if test_image is not None:
            with col1:
                st.subheader("Uploaded Leaf Image")
                st.image(test_image, use_container_width=True)
            
            with col2:
                st.subheader("Diagnosis")
                if st.button("Analyze Leaf", type="primary"):
                    with st.spinner("Analyzing with MobileNetV2..."):
                        result_index, confidence = model_prediction(test_image, model)
                        raw_label = CLASS_NAMES[result_index]
                        friendly_label = format_class_name(raw_label)
                        
                        st.snow()
                        if "healthy" in raw_label.lower():
                            st.success(f"🌱 **Diagnosis: {friendly_label}**")
                        else:
                            st.warning(f"⚠️ **Diagnosis: {friendly_label}**")
                        
                        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                        st.caption(f"Raw Label: `{raw_label}` (Class Index: {result_index})")
        else:
            st.write("Please upload a leaf image to begin.")
