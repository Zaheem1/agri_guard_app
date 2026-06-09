import os
import numpy as np
import cv2
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

# 2. Urdu localization diagnostic alerts map
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
    Loads the model using OpenCV's highly stable DNN framework.
    Completely bypasses TensorFlow and TFLite installation runtime crashes.
    """
    model_path = "crop_disease_model_quantized.tflite"  
    
    if os.path.exists(model_path):
        # OpenCV reads the TFLite graph directly into an optimized CPU forward network
        net = cv2.dnn.readNetFromTFLite(model_path)
        return net
    else:
        raise FileNotFoundError(f"❌ Model file '{model_path}' not found in repo root folder!")

def predict_crop_disease(net, pil_image):
    """Processes image and handles inference using OpenCV blob calculations"""
    # 1. Convert PIL image to NumPy RGB array
    img_rgb = np.array(pil_image.convert('RGB'))
    
    # 2. Use OpenCV blobFromImage to resize to 224x224 and preserve raw float values
    blob = cv2.dnn.blobFromImage(
        img_rgb, 
        scalefactor=1.0, 
        size=(224, 224), 
        mean=(0, 0, 0), 
        swapRB=False, 
        crop=False
    )
    
    # 3. Pass the blob input directly into the network graph
    net.setInput(blob)
    
    # 4. Execute the forward pass propagation
    predictions = net.forward()
    
    # 5. Extract results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    return CLASS_NAMES[predicted_idx], confidence

def generate_urdu_audio_api(text_prompt):
    """Generates localized Urdu speech using Hugging Face's cloud inference API"""
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-urd"
    try:
        response = requests.post(API_URL, json={"inputs": text_prompt}, timeout=15)
        if response.status_code == 200:
            return response.content, 16000
    except Exception as e:
        print(f"Audio cloud generation fallback triggered: {e}")
    return None, None
