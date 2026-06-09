import os
import numpy as np
import tflite_runtime.interpreter as tflite
import requests
from PIL import Image

# 1. Exact alphabetical class indices matching your production model output layer
CLASS_NAMES = [
    'Cotton___Bacterial_Blight', 
    'Cotton___Healthy', 
    'Rice___Blast', 
    'Rice___Brown_Spot', 
    'Rice___Healthy', 
    'Wheat___Healthy', 
    'Wheat___Leaf_Rust', 
    'Wheat___Septoria_Leaf_Blotch'
]

# 2. Urdu localization diagnostic alerts map for farmers
URDU_DIAGNOSTICS_MAP = {
    'Cotton___Bacterial_Blight': "کپاس میں بیکٹیریل بلائٹ کی بیماری پائی گئی ہے۔ پودوں میں فاصلہ رکھیں، نائٹروجن کھاد کم کریں، اور تانبے والی دوائی کا سپرے کریں۔",
    'Cotton___Healthy': "آپ کی کپاس کی فصل بالکل صحت مند اور تندرست ہے۔ صفائی کا خاص خیال رکھیں۔",
    'Rice___Blast': "چاول میں بلاسٹ کی بیماری دیکھی گئی ہے۔ نائٹروجن کا استعمال کم کریں اور فوری طور پر فنگسائڈ کا سپرے کریں۔",
    'Rice___Brown_Spot': "چاول کے پتے پر بھورے دھبے دیکھے گئے ہیں۔ یہ عام طور پر غذائیت کی کمی کی علامت ہے۔ کھاد کا متوازن استعمال یقینی بنائیں۔",
    'Rice___Healthy': "آپ کے چاول کی فصل بالکل ٹھیک اور صحت مند ہے۔ پانی اور کھاد کا باقاعدہ شیڈول جاری رکھیں۔",
    'Wheat___Healthy': "گندم کی فصل بالکل صحت مند اور تندرست ہے۔ بہترین پیداوار کے لیے وقت پر پانی دیں۔",
    'Wheat___Leaf_Rust': "گندم پر پتے کی کنگی یعنی لیف رسٹ کی بیماری کی علامات ہیں۔ دھوپ کی موجودگی میں موزوں فنگسائڈ کا فوری سپرے کریں۔",
    'Wheat___Septoria_Leaf_Blotch': "گندم میں سیپٹوریا لیف بلاچ کی علامات ہیں۔ متاثرہ پتوں کو الگ کریں اور سفارش کردہ فنگسائڈ کا استعمال کریں۔"
}

def load_inference_model():
    """
    Loads the lightweight quantized TFLite interpreter instead of full TensorFlow.
    This protects Streamlit Cloud containers from running out of RAM.
    """
    model_path = "crop_disease_model_quantized.tflite"  
    
    if os.path.exists(model_path):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        raise FileNotFoundError(f"❌ Quantized TFLite model file '{model_path}' not found in repo root folder!")

def predict_crop_disease(interpreter, pil_image):
    """
    Processes the raw image and runs inference using the TFLite Interpreter.
    Maintains raw [0, 255] float scaling required natively by EfficientNet base layers.
    """
    # Get structural input and output details from the TFLite graph
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess the input image to match EfficientNet dimensions (224x224)
    resized_img = pil_image.resize((224, 224))
    img_array = np.array(resized_img, dtype=np.float32)
    
    # Append Channels-Last batch dimension: shape transforms to (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Bind the preprocessed image data array to the input tensor slot
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    
    # Run the TFLite math engine calculation forward pass
    interpreter.invoke()
    
    # Retrieve the raw prediction probabilities matrix from the output tensor layer
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    return CLASS_NAMES[predicted_idx], confidence

def generate_urdu_audio_api(text_prompt):
    """
    Generates localized Urdu speech audio bytes using Hugging Face's cloud inference API.
    Bypasses the need to install a 1GB+ local PyTorch speech framework package.
    """
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-urd"
    
    try:
        response = requests.post(API_URL, json={"inputs": text_prompt}, timeout=15)
        if response.status_code == 200:
            # Return raw audio bytes from the server and standard 16kHz sampling rate
            return response.content, 16000
    except Exception as e:
        print(f"Audio cloud generation fallback triggered: {e}")
        
    return None, None
