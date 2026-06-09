import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VitsModel, AutoTokenizer
from PIL import Image

# Exact class indices matching your production model output layer
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
    'Rice___Healthy': "آپ کے چاول کی فصل بالکل ٹھیک اور صحت مند ہے۔ پانی اور کھاد کا باقاعدہ شیڈول جاری رکھیں۔",
    'Wheat___Healthy': "گندم کی فصل بالکل صحت مند اور تندرست ہے۔ بہترین پیداوار کے لیے وقت پر پانی دیں۔",
    'Wheat___Leaf_Rust': "گندم پر پتے کی کنگی یعنی لیف رسٹ کی بیماری کی علامات ہیں۔ دھوپ کی موجودگی میں موزوں فنگسائڈ کا فوری سپرے کریں۔",
    'Wheat___Septoria_Leaf_Blotch': "گندم میں سیپٹوریا لیف بلاچ کی علامات ہیں۔ متاثرہ پتوں کو الگ کریں اور سفارش کردہ فنگسائڈ کا استعمال کریں۔"
}

# Standalone network container mapping out forward pass calculations
class AgriGuardNativeNet(nn.Module):
    def __init__(self, input_dim=1280):
        super(AgriGuardNativeNet, self).__init__()
        self.classifier = nn.Linear(input_dim, 8) 
        
    def forward(self, x):
        # Global mean feature evaluation
        x = x.mean(dim=[-2, -1]) if len(x.shape) == 4 else x
        x = torch.flatten(x, 1)
        return self.classifier(x)

def load_tflite_model(model_path="crop_disease_model_native.pt"):
    """Loads raw PyTorch model weight vectors safely into memory"""
    if not os.path.exists(model_path):
        return None
    
    # Load raw dictionary checkpoints
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Infer feature mapping dimensions dynamically based on checkpoint size
    weight_key = 'classifier.weight' if 'classifier.weight' in checkpoint else list(checkpoint.keys())[0]
    input_features_dim = checkpoint[weight_key].shape[1]
    
    model = AgriGuardNativeNet(input_dim=input_features_dim)
    
    # Handle direct state updates manually to preserve strict dictionary key boundaries
    try:
        model.load_state_dict(checkpoint)
    except:
        # Fallback dictionary parser block if key roots vary slightly
        clean_state = {'classifier.weight': checkpoint[weight_key], 'classifier.bias': checkpoint[list(checkpoint.keys())[1]]}
        model.load_state_dict(clean_state)
        
    model.eval()
    return model

import os
import numpy as np
import tensorflow as tf
import requests
from PIL import Image

# ... Keep your CLASS_NAMES and URDU_DIAGNOSTICS_MAP exactly as they are ...

def load_inference_model():
    """Loads the core trained Keras model containing the EfficientNet base and top heads"""
    model_path = "crop_disease_model.keras"  
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"❌ Model file '{model_path}' not found in repo root folder!")

def predict_crop_disease(model, pil_image):
    """Processes image and handles inputs exactly matching TensorFlow EfficientNet pipeline rules"""
    resized_img = pil_image.resize((224, 224))
    img_array = np.array(resized_img, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_tensor)
    predicted_idx = np.argmax(predictions[0])import os
import numpy as np
import tensorflow as tf
import requests
from PIL import Image

# 1. Exact class indices matching your production model output layer
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
    'Rice___Brown_Spot': "چاول کے پتے پر بھورے دھبے دیدهے گئے ہیں۔ یہ عام طور پر غذائیت کی کمی کی علامت ہے۔ کھاد کا متوازن استعمال یقینی بنائیں۔",
    'Rice___Healthy': "آپ کے چاول کی فصل بالکل ٹھیک اور صحت مند ہے۔ پانی اور کھاد کا باقاعدہ شیڈول جاری رکھیں۔",
    'Wheat___Healthy': "گندم کی فصل بالکل صحت مند اور تندرست ہے۔ بہترین پیداوار کے لیے وقت پر پانی دیں۔",
    'Wheat___Leaf_Rust': "گندم پر پتے کی کنگی یعنی لیف رسٹ کی بیماری کی علامات ہیں۔ دھوپ کی موجودگی میں موزوں فنگسائڈ کا فوری سپرے کریں۔",
    'Wheat___Septoria_Leaf_Blotch': "گندم میں سیپٹوریا لیف بلاچ کی علامات ہیں۔ متاثرہ پتوں کو الگ کریں اور سفارش کردہ فنگسائڈ کا استعمال کریں۔"
}

def load_inference_model():
    """Loads the core trained Keras model containing the EfficientNet base and top heads"""
    model_path = "crop_disease_model.keras"  
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"❌ Model file '{model_path}' not found in repo root folder!")

def predict_crop_disease(model, pil_image):
    """Processes image and handles inputs exactly matching TensorFlow EfficientNet pipeline rules"""
    resized_img = pil_image.resize((224, 224))
    img_array = np.array(resized_img, dtype=np.float32)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_tensor)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100
    
    return CLASS_NAMES[predicted_idx], confidence

def generate_urdu_audio_api(text_prompt):
    """Generates localized Urdu speech using Hugging Face's lightweight cloud inference API"""
    # Free server endpoint for the Meta MMS Urdu text-to-speech engine
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-urd"
    
    try:
        response = requests.post(API_URL, json={"inputs": text_prompt}, timeout=15)
        if response.status_code == 200:
            # Return raw audio bytes from the server and standard 16kHz sampling rate
            return response.content, 16000
    except Exception as e:
        print(f"Audio generation fallback triggered: {e}")
    return None, None
    confidence = predictions[0][predicted_idx] * 100
    
    return CLASS_NAMES[predicted_idx], confidence

def generate_urdu_audio_api(text_prompt):
    """Generates localized Urdu speech using Hugging Face's lightweight cloud inference API"""
    # Free server endpoint for the Meta MMS Urdu text-to-speech engine
    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-urd"
    
    try:
        response = requests.post(API_URL, json={"inputs": text_prompt}, timeout=15)
        if response.status_code == 200:
            # Return raw audio bytes from the server and standard 16kHz sampling rate
            return response.content, 16000
    except Exception as e:
        print(f"Audio generation fallback triggered: {e}")
    return None, None
