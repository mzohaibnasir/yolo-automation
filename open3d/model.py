from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n")  # Use "yolov8n.yaml" for YOLOv8 Nano, change if you want a different version

# Train the model
model.train(
    data="/home/zohaib/pytorch3d-renderer/open3d/data.yaml",  # Path to your dataset YAML file
    epochs=70,  # Number of epochs (adjust based on your needs)
    imgsz=640,  # Image size (adjust based on your needs)
    batch=24,  # Batch size (adjust based on your GPU memory)
    project="yolo_training",  # Project directory
    name="experiment",  # Experiment name
    save_period=25,  # Save model checkpoints every epoch
    augment=True
)

