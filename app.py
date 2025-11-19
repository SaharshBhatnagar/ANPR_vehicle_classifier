import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import cv2
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tempfile # NEW: Handles temporary video files

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
    # Load Class Names
    with open(CLASS_FILE, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Load Vehicle Classifier
    classifier = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES)
    classifier.load_state_dict(torch.load(CLS_PATH, map_location='cpu'))
    classifier.to(DEVICE).eval()
    
    # Load License Plate Detector
    anpr = YOLO(ANPR_PATH)
    anpr.to(DEVICE)
    
    return classifier, anpr, class_names

# 3. PREPROCESSING (Same as before)
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(), 
    ToTensorV2()
])

# Helper function to process a single frame
def process_frame(frame_bgr, classifier, anpr, class_names):
    # Convert BGR to RGB for Classifier
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # --- TASK 1: Classification ---
    # Prepare input tensor
    input_tensor = transform(image=frame_rgb)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = probs.topk(1)
    
    v_type = class_names[top_idx.item()]
    v_conf = top_prob.item() * 100
    
    # Draw Classification Text on Frame
    text = f"{v_type}: {v_conf:.1f}%"
    cv2.putText(frame_bgr, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, (0, 255, 0), 3)

    # --- TASK 2: Detection ---
    results = anpr(frame_bgr, verbose=False)
    boxes = results[0].boxes
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame_bgr, "Plate", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)

    return frame_bgr

# 4. MAIN APP UI
st.title("AI Vehicle System: Image & Video", anchor=False)
classifier, anpr, class_names = load_models()

# Create Tabs for Mode Selection
tab1, tab2 = st.tabs(["📷 Image Mode", "🎥 Video Mode"])

# --- TAB 1: IMAGE MODE  ---
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        processed_frame = process_frame(image_bgr.copy(), classifier, anpr, class_names)
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Analyzed Image", use_container_width=True)


# --- TAB 2: VIDEO MODE  ---
with tab2:
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        # Save uploaded video to a temporary file (OpenCV needs a file path)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder for the video frame
        
        stop_button = st.button("Stop Processing")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            
            # Note: We skip frames if needed for speed (e.g. process every 3rd frame)
            annotated_frame = process_frame(frame, classifier, anpr, class_names)
            
            # Show in Streamlit (Convert BGR back to RGB for display)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
        cap.release()