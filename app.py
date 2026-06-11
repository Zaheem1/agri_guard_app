import streamlit as st
from PIL import Image
import io
import soundfile as sf
import model_utils as utils

# Set page characteristics
st.set_page_config(
    page_title="AgriGuard Pakistan: Smart Crop Diagnostic System",
    page_icon="🌱",
    layout="centered"
)

# Custom Localized CSS UI Enhancement Styling 
st.markdown("""
    <style>
    .main { background-color: #f7f9fb; }
    .stButton>button { width: 100%; background-color: #2e7d32; color: white; font-weight: bold; }
    .report-card { padding: 20px; border-radius: 10px; background-color: white; border-left: 5px solid #2e7d32; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 AgriGuard Pakistan")
st.markdown("### **Automated Crop Disease Diagnosis with Native Urdu Voice Feedback**")
st.write("Supported Varieties: **Wheat (گندم)** | **Rice (چاول)** | **Cotton (کپاس)**")
st.markdown("---")

# Optimize loading operations via resource caching
@st.cache_resource
def get_cached_models():
    model = utils.load_tflite_model()
    tokenizer, tts_model, device = utils.load_mms_tts_engine()
    return model, tokenizer, tts_model, device

# Boot up models
model, tokenizer, tts_model, device = get_cached_models()

if model is None:
    st.error("❌ Missing Asset: 'crop_disease_model_native.pt' not found in project folder. Please pull the file from Google Colab and place it here.")
else:
    # 📸 Step 1: User Upload Intercept Loop
    st.markdown("### 📸 1. Upload Field Leaf Image")
    uploaded_file = st.file_uploader("Upload image (PNG, JPG, JPEG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and show file
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Specimen View", use_container_width=True)
        
        # 🔬 Step 2: Running Inference Execution Check
        st.markdown("---")
        st.markdown("### 🎯 2. Diagnostic Summary Results")
        
        with st.spinner("Processing image pixels via native math tensors..."):
            predicted_class, confidence_score = utils.predict_crop_disease(model, image)
            
        # Format strings cleanly for the user display card
        crop_type, condition = predicted_class.split("___")
        clean_condition = condition.replace("_", " ")
        
        # Output Presentation Card Box layout
        st.markdown(f"""
        <div class="report-card">
            <h4>📋 Diagnostics Analysis Card</h4>
            <p><b>Target Crop Family:</b> {crop_type}</p>
            <p><b>Classification Label:</b> {clean_condition}</p>
            <p><b>Statistical Match Confidence:</b> {confidence_score:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("") # Margin spacing block
        
        # 🗣️ Step 3: Meta MMS Urdu Voice Feedback Generation Run
        st.markdown("### 🗣️ 3. Localized Remedy Recommendation (Urdu Voice)")
        urdu_text = utils.URDU_DIAGNOSTICS_MAP[predicted_class]
        
        # Display the Urdu text instruction window
        st.text_area("Prescription Text (اردو ترجمہ):", value=urdu_text, height=95, disabled=True)
        
        with st.spinner("Synthesizing native phonetics via Meta MMS VITS framework..."):
            try:
                # Synthesize text directly to sound vectors
                audio_array, sampling_rate = utils.generate_urdu_audio(tokenizer, tts_model, device, urdu_text)
                
                # Write back wave streams to a virtual memory buffer file array 
                virtual_buffer = io.BytesIO()
                sf.write(virtual_buffer, audio_array, samplerate=sampling_rate, format='WAV')
                virtual_buffer.seek(0)
                
                # Mount interactive audio player controller to the browser DOM interface
                st.audio(virtual_buffer, format="audio/wav")
                st.toast("🔊 Audio warning instructions loaded successfully!", icon="✅")
                
            except Exception as e:
                st.error(f"Voice Synthesizer runtime disruption encountered: {e}")
