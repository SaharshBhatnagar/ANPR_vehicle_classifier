import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import cv2
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANPR_MODEL_PATH = 'anpr_best.pt'
CLASSIFIER_MODEL_PATH = 'vehicle_classifier.pth'
CLASSIFIER_CLASSES_PATH = 'vehicle_classes.txt'
IMG_SIZE = 224
NUM_CLASSES = 6 # *** THIS IS THE CORRECTED VALUE ***
MODEL_NAME = 'mobilenetv3_large_100'

# --- Load Classification Model ---
@st.cache_resource
def load_classifier_model():
    print("Loading classifier model...")
    # Load class names
    try:
        with open(CLASSIFIER_CLASSES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error(f"Error: '{CLASSIFIER_CLASSES_PATH}' not found. Please train the classifier first.")
        return None, None
    
    # Check if number of classes matches
    if len(class_names) != NUM_CLASSES:
        st.warning(f"Warning: Loaded {len(class_names)} classes, but NUM_CLASSES is set to {NUM_CLASSES}. Check 'vehicle_classes.txt'.")
        
    # Load model architecture
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location='cpu'))
    except FileNotFoundError:
        st.error(f"Error: '{CLASSIFIER_MODEL_PATH}' not found. Please train the classifier first.")
        return None, None
        
    model = model.to(DEVICE)
    model.eval()
    print("Classifier model loaded.")
    return model, class_names

# --- Load ANPR Model ---
@st.cache_resource
def load_anpr_model():
    print("Loading ANPR model...")
    try:
        model = YOLO(ANPR_MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Error: '{ANPR_MODEL_PATH}' not found. Please train the ANPR model first.")
        return None
        
    model = model.to(DEVICE)
    print("ANPR model loaded.")
    return model

# --- Preprocessing for Classifier ---
classifier_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- Main App Logic ---

# Load models
anpr_model = load_anpr_model()
classifier_model, class_names = load_classifier_model()

# Streamlit UI
st.title("Vehicle ANPR and Classification System")
st.write(f"Using device: {DEVICE}")

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and anpr_model is not None and classifier_model is not None:
    # 1. Load Image
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # OpenCV format

    st.image(image_pil, caption="Uploaded Image", use_container_width=True)
    
    col1, col2 = st.columns(2)

    with col1:
        # --- 2. Run Vehicle Classification ---
        st.subheader("Vehicle Type Prediction")
        with st.spinner("Classifying vehicle..."):
            # Preprocess
            augmented = classifier_transforms(image=image_np)
            image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = classifier_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_idx = probabilities.topk(1, dim=1)
            
            pred_class = class_names[top_idx.item()]
            pred_conf = top_prob.item() * 100
            
            st.success(f"**{pred_class}** ({pred_conf:.2f}%)")
            
            # Show top-k predictions
            st.write("Top predictions:")
            probs_np = probabilities.cpu().numpy().flatten()
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {probs_np[i]*100:.2f}%")

    with col2:
        # --- 3. Run ANPR Detection ---
        st.subheader("License Plate Detection")
        with st.spinner("Detecting license plate..."):
            # Predict
            anpr_results = anpr_model(image_np_bgr) # Pass BGR image
            
            annotated_image = image_np_bgr.copy()
            detected_plates = []
            
            # Check results
            if len(anpr_results[0].boxes) == 0:
                st.warning("No license plates detected.")
            else:
                st.success(f"Found {len(anpr_results[0].boxes)} license plate(s).")
                
                # Loop through detected boxes
                for box in anpr_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    
                    # Draw box on image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Plate: {conf*100:.1f}%"
                    cv2.putText(annotated_image, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Crop plate
                    cropped_plate = image_pil.crop((x1, y1, x2, y2))
                    detected_plates.append(cropped_plate)

                # Display annotated image
                st.subheader("Annotated Image")
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_container_width=True)
                
                # Display cropped plates
                if detected_plates:
                    st.subheader("Cropped Plate(s)")
                    if len(detected_plates) > 4:
                        # Use columns for many plates
                        cols = st.columns(4)
                        for i, plate in enumerate(detected_plates):
                            cols[i % 4].image(plate)
                    else:
                        # Use columns for few plates
                        cols = st.columns(len(detected_plates))
                        for i, plate in enumerate(detected_plates):
                            cols[i].image(plate)
elif uploaded_file is None:
    st.info("Please upload an image to begin.")
else:
    st.error("Models failed to load. Please check the terminal for errors and ensure you have trained the models.")