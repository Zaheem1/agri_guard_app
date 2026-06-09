import os
import numpy as np
import tensorflow as tf
import torch
from transformers import VitsModel, AutoTokenizer
from PIL import Image

# Exact class indices matching your Keras alphabetical dataset layout
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

URDU_DIAGNOSTICS_MAP = {
    'Cotton___Bacterial_Blight': "کپاس میں بیکٹیریل بلائٹ کی بیماری پائی گئی ہے۔ پودوں میں فاصلہ رکھیں، نائٹروجن کھاد کم کریں، اور تانبے والی دوائی کا سپرے کریں۔",
    'Cotton___Healthy': "آپ کی کپاس کی فصل بالکل صحت مند اور تندرست ہے۔ صفائی کا خاص خیال رکھیں۔",
    'Rice___Blast': "چاول میں بلاسٹ کی بیماری دیکھی گئی ہے۔ نائٹروجن کا استعمال کم کریں اور فوری طور پر فنگسائڈ کا سپرے کریں۔",
    'Rice___Brown_Spot': "چاول کے پتے پر بھورے دھبے دیکھے گئے ہیں۔ یہ عام طور پر غذائیت کی کمی کی علامت ہے۔ کھاد کا متوازن استعمال یقینی بنائیں۔",
    'Rice___Healthy': "آپ کے چاول کی فصل بالکل ٹھیک اور صحت مند ہے۔ پانی اور کھاد کا باقاعدگی سے استعمال جاری رکھیں۔",
    'Wheat___Healthy': "آپ کی گندم کی فصل بالکل تندرست ہے۔ موسمی حالات کے مطابق پانی دیتے رہیں۔",
    'Wheat___Leaf_Rust': "گندم میں لیف رسٹ (کنگی) کی بیماری دیکھی گئی ہے۔ فصل کا معائنہ بڑھائیں اور فوری طور پر سفارش کردہ فنگسائڈ سپرے کریں۔",
    'Wheat___Septoria_Leaf_Blotch': "گندم کے پتے پر جھلساؤ (سیپٹوریا) کی علامات ہیں۔ متاثرہ پتوں کو دور رکھیں اور کیمیائی سپرے کا استعمال کریں۔"
}

def load_inference_model():
    """Loads the core trained Keras model containing the EfficientNet base and top heads"""
    model_path = "crop_disease_model.keras"  # Place your real exported model file here
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}. Please upload it to your workspace root.")

def predict_crop_disease(model, pil_image):
    """Processes image and handles inputs exactly matching TensorFlow EfficientNet pipeline rules"""
    # Step A: Resize precisely to matches network input dimensions
    resized_img = pil_image.resize((224, 224))
    
    # Step B: Cast image object to structural float32 array
    img_array = np.array(resized_img, dtype=np.float32)
    
    # Step C: CRITICAL FIX: DO NOT normalize or divide by 255.0 for EfficientNet-B0!
    # EfficientNet has native scaling layers embedded in its core graph layer trees.
    
    # Step D: Append Channels-Last batch dimension: shape maps to (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Step E: Execute graph evaluation
    predictions = model.predict(img_tensor)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    return CLASS_NAMES[predicted_idx], confidence

def load_voice_components():
    """Loads Meta MMS VITS model elements for Urdu TTS speech rendering"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-urd")
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-urd").to(device)
    return tokenizer, tts_model, device

def generate_urdu_audio(tokenizer, tts_model, device, text_prompt):
    """Generates a localized audio vector sequence from Urdu diagnostic mappings"""
    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = tts_model(**inputs)
        # Extract native raw audio waveform array values
        audio_data = output.waveform[0].cpu().numpy()
    
    return audio_data, tts_model.config.sampling_rate