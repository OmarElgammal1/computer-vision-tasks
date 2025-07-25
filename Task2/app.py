import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- Configuration ---
MODEL_PATH = "dental_classifier_model.pth"
NUM_CLASSES = 7
CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
IMAGE_SIZE = (224, 224)

# --- Model Definition ---
def get_model(num_classes):
    model = models.resnet50(weights=None) # No need to load pre-trained weights here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- Image Transformations ---
def get_transform():
    # Must be the same as the validation transforms from training
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

# --- Prediction Function ---
def predict(model, image_bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    predicted_class_name = CLASS_NAMES[predicted_class_idx.item()]
    confidence_score = confidence.item()
    
    return predicted_class_name, confidence_score, image

# --- Streamlit App UI ---
st.set_page_config(page_title="Dental Image Classifier", layout="centered")
st.title("🦷 AI Dental Image Classifier")
st.write(
    "Upload a dental image to classify it into one of seven categories. "
    "This app uses a deep learning model (ResNet-50) trained on a custom dataset."
)

# Load the model
try:
    model = load_model(MODEL_PATH, NUM_CLASSES)
except FileNotFoundError:
    st.error(f"Model file not found at '{MODEL_PATH}'. Please run the training script first to generate it.")
    st.stop()


# File uploader
uploaded_file = st.file_uploader(
    "Choose a dental image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Get image bytes
    image_bytes = uploaded_file.getvalue()
    
    # Predict button
    if st.button("Classify Image"):
        with st.spinner("Analyzing the image..."):
            predicted_class, confidence, original_image = predict(model, image_bytes)
        
        # Display the image
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
        
        # Display the result
        st.success(f"**Prediction: {predicted_class}**")
        st.info(f"**Confidence:** {confidence*100:.2f}%")

st.sidebar.header("About")
st.sidebar.info(
    "This application demonstrates a transfer learning approach to classifying "
    "dental images. The underlying model is a ResNet-50 fine-tuned on a "
    "specialized dataset."
)