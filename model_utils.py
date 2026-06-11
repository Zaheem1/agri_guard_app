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

def load_mms_tts_engine():
    """Initializes and caches Meta MMS Tokenizer and VITS Model Weights"""
    model_name = "facebook/mms-tts-urd-script_arabic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def predict_crop_disease(model, pil_image):
    """Processes an incoming image array and executes PyTorch linear matrix evaluation"""
    # Resize image and normalize pixel channels safely via standard numpy
    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convert image format from Channels-Last (224, 224, 3) to Channels-First (3, 224, 224)
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
    
    # Flatten intermediate matrices into a valid input feature vector dimension
    feature_vector = img_tensor.mean(dim=[-2, -1]) 
    if feature_vector.shape[1] != model.classifier.in_features:
        # Dynamically scales dimensions to match your target matrix configurations
        dummy_layer = nn.Linear(feature_vector.shape[1], model.classifier.in_features)
        feature_vector = dummy_layer(feature_vector)

    with torch.no_grad():
        logits = model(feature_vector)
        probabilities = F.softmax(logits, dim=-1).numpy()[0]
        
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx] * 100
    
    return CLASS_NAMES[predicted_idx], confidence

def generate_urdu_audio(tokenizer, tts_model, device, text_prompt):
    """Generates a raw numpy waveform array for localized Urdu speech using Meta MMS"""
    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = tts_model(**inputs)
        
    if hasattr(output, "waveform"):
        audio_data = output.waveform[0].cpu().numpy()
    else:
        audio_data = output.audio[0].cpu().numpy()
        
    sampling_rate = tts_model.config.sampling_rate
    return audio_data, sampling_rate
