import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import cv2
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tempfile

# 1. CONFIGURATION
DEVICE = "cuda"
ANPR_PATH = 'anpr_best.pt'
CLS_PATH = 'vehicle_classifier.pth'
CLASS_FILE = 'vehicle_classes.txt'
IMG_SIZE = 224
NUM_CLASSES = 6 
MODEL_NAME = 'mobilenetv3_large_100'

# 2. LOAD AI MODELS
@st.cache_resource
def load_models():
    # A. Load Class Names
    with open(CLASS_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # B. Load Vehicle Classifier (MobileNet - Checks "Type")
    classifier = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES)
    classifier.load_state_dict(torch.load(CLS_PATH, map_location='cpu'))
    classifier.to(DEVICE).eval()
    
    # C. Load License Plate Detector (Custom YOLO - Finds "Plate")
    anpr = YOLO(ANPR_PATH)
    anpr.to(DEVICE)

    # D. Load Vehicle Detector (Standard YOLO - Finds "Vehicle")
    # 'n' version is small and fast. It knows 'car', 'bus', 'truck' by default.
    vehicle_detector = YOLO('yolov8n.pt')
    vehicle_detector.to(DEVICE)
    
    return classifier, anpr, vehicle_detector, class_names

# 3. PREPROCESSING
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(), 
    ToTensorV2()
])

# Helper function to process a single frame
def process_frame(frame_bgr, classifier, anpr, vehicle_detector, class_names):
    # --- TASK 1: Find and Draw Vehicle (Blue Box) ---
    # We tell YOLO to only look for: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
    vehicle_results = vehicle_detector(frame_bgr, classes=[2, 3, 5, 7], verbose=False)
    
    for box in vehicle_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 1. Draw Blue Box for Vehicle
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue Color
        
        # 2. Double Check Type with MobileNet
        # We crop the vehicle area and ask MobileNet exactly what it is
        vehicle_crop = frame_bgr[y1:y2, x1:x2]
        if vehicle_crop.size > 0:
            # Convert crop to RGB for MobileNet
            crop_rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
            input_tensor = transform(image=crop_rgb)['image'].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                top_prob, top_idx = probs.topk(1)
            
            v_type = class_names[top_idx.item()]
            v_conf = top_prob.item() * 100
            
            # Write Label above the Blue Box
            label = f"{v_type} {v_conf:.0f}%"
            cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 0, 0), 2)

    # --- Find and Draw Plate (Green Box) ---
    plate_results = anpr(frame_bgr, verbose=False)
    for box in plate_results[0].boxes:
        px1, py1, px2, py2 = map(int, box.xyxy[0])
        
        # Draw Green Box for Plate
        cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), (0, 255, 0), 3) # Green Color
        cv2.putText(frame_bgr, "Plate", (px1, py2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

    return frame_bgr

# 4. MAIN APP UI
st.title("Vehicle & Plate Detection System", anchor=False)
classifier, anpr, vehicle_detector, class_names = load_models()

tab1, tab2 = st.tabs(["📷 Image Mode", "🎥 Video Mode"])

# --- TAB 1: IMAGE MODE ---
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        processed_frame = process_frame(image_bgr.copy(), classifier, anpr, vehicle_detector, class_names)
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)

# --- TAB 2: VIDEO MODE ---
with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        stop_button = st.button("Stop Processing")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = process_frame(frame, classifier, anpr, vehicle_detector, class_names)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
        cap.release()