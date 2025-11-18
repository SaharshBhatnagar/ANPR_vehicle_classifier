import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import cv2
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. CONFIGURATION
DEVICE = "cuda"
ANPR_PATH = 'anpr_best.pt'
CLS_PATH = 'vehicle_classifier.pth'
CLASS_FILE = 'vehicle_classes.txt'

# Model Settings
IMG_SIZE = 224
NUM_CLASSES = 6 
MODEL_NAME = 'mobilenetv3_large_100'


# 2. LOAD AI MODELS

@st.cache_resource
def load_models():
    # A. Load Class Names
    with open(CLASS_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # B. Load Vehicle Classifier
    classifier = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES)
    classifier.load_state_dict(torch.load(CLS_PATH, map_location='cpu'))
    classifier.to(DEVICE).eval()
    
    # C. Load License Plate Detector
    anpr = YOLO(ANPR_PATH)
    anpr.to(DEVICE)
    
    return classifier, anpr, class_names


# 3. IMAGE PREPROCESSING
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(), 
    ToTensorV2()
])


# 4. MAIN APP UI

st.title("License Plate Detection With Vehicle Classification", anchor=False)

# Load models once
classifier, anpr, class_names = load_models()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # -- Read Image --
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image_pil, caption="Input Image", use_container_width=True)
    
    col1, col2 = st.columns(2)

    # -- TASK 1: Vehicle Type --
    with col1:
        st.subheader("Vehicle Type")
        
        # Prepare image for AI
        input_tensor = transform(image=image_np)['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = probs.topk(1)
        
        # Get results
        v_type = class_names[top_idx.item()]
        v_conf = top_prob.item() * 100
        
        st.success(f"**{v_type}** ({v_conf:.2f}%)")

        st.write("Top predictions:")
        all_probs = probs.cpu().numpy().flatten() # Get all probabilities as a list
        
        # Loop through all classes and their probabilities
        for name, prob in zip(class_names, all_probs):
            st.write(f"{name}: {prob*100:.2f}%")

    # -- TASK 2: License Plate --
    with col2:
        st.subheader("License Plate")
        
        # YOLO Detection
        results = anpr(image_bgr, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            st.warning("No Plate Detected")
        else:
            st.success(f"Found {len(boxes)} Plate(s)")
            
            annotated = image_bgr.copy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw Box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Crop and Show Plate
                plate_crop = image_pil.crop((x1, y1, x2, y2))
                st.image(plate_crop, width=150)

            # Show full annotated image
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)