from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8n (nano) model
    # 'n' is the smallest and fastest. 's' or 'm' might be more accurate.
    model = YOLO('yolov8n.pt') 

    # Train the model
    # Your RTX 4060 will be automatically detected and used (device=0)
    print("Starting ANPR model training...")
    model.train(
        data='anpr_data.yaml',
        epochs=50,          # 50 epochs is a good start
        imgsz=640,          # Image size
        batch=16,           # Adjust batch size based on your 8GB VRAM
        name='yolov8n_anpr_custom'
    )
    print("ANPR model training complete.")
    # The best model will be saved in: runs/detect/yolov8n_anpr_custom/weights/best.pt

if __name__ == '__main__':
    main()