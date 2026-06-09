import streamlit as st
from PIL import Image
import io

# Import your optimized, lightweight OpenCV and API mechanics
from model_utils import (
    load_inference_model, 
    predict_crop_disease, 
    generate_urdu_audio_api, 
    URDU_DIAGNOSTICS_MAP
)

# --- UI Page Configuration ---
st.set_page_config(
    page_title="AgriGuard - Crop Disease Diagnostics",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 AgriGuard: AI Crop Disease Assistant")
st.write("Upload a leaf image of **Cotton, Rice, or Wheat** to detect diseases and listen to localized remedies in Urdu.")

# --- Cache Model Loading ---
@st.cache_resource
def get_model():
    try:
        return load_inference_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Initialize the OpenCV DNN framework network
net = get_model()

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Choose a crop leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded asset cleanly
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Crop Leaf Image", use_container_width=True)
    
    st.write("🔄 Running diagnostics matrix...")
    
    if net is not None:
        try:
            # 1. Run inference using the OpenCV DNN function wrapper
            class_name, confidence = predict_crop_disease(net, image)
            
            # 2. Display localized results panel
            st.subheader("📊 Diagnostic Results")
            st.metric(label="Detected Condition", value=class_name.replace("___", " - "))
            st.write(f"**Confidence Level:** {confidence:.2f}%")
            
            # 3. Retrieve and present localized Urdu advice text
            urdu_advice = URDU_DIAGNOSTICS_MAP.get(class_name, "معلومات دستیاب نہیں ہیں۔")
            
            st.markdown("---")
            st.subheader("📢 اردو میں معلوماتی رہنمائی (Urdu Advisory)")
            st.info(urdu_advice)
            
            # 4. Stream Audio via Cloud Inference API safely without local PyTorch
            with st.spinner("🔊 Generating audio advice..."):
                audio_bytes, sampling_rate = generate_urdu_audio_api(urdu_advice)
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav", start_time=0)
                else:
                    st.warning("⚠️ Could not generate audio at this moment. Please read the advisory text above.")
                    
        except Exception as e:
            st.error(f"An error occurred during image assessment: {e}")
    else:
        st.error("AI engine is unavailable because the model file failed to load.")
