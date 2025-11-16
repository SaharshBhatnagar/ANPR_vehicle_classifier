from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8n (nano) model
    # 'n' is the smallest and fastest. 's' or 'm' might be more accurate.
    model = YOLO('yolov8n.pt') 

    # Train the model
    print("Starting ANPR model training...")
    model.train(
        data='anpr_data.yaml',
        epochs=50,          
        imgsz=640,          
        batch=16,           
        name='yolov8n_anpr_custom'
    )
    print("ANPR model training complete.")
    # The best model will be saved in: runs/detect/yolov8n_anpr_custom/weights/best.pt

if __name__ == '__main__':
    main()