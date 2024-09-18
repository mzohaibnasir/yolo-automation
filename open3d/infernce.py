from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model (you can replace this with your custom YOLO model)
model = YOLO("/home/zohaib/pytorch3d-renderer/open3d/yolo_training/experiment14/weights/best.pt")  # Path to your trained model

# Set webcam stream
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB (YOLO expects this)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference with YOLOv8
    results = model.track(img_rgb, conf=0.5)

    # Get the detection results
    detected_results = results[0].boxes  # Get detected boxes from the first image (as this is real-time video)

    # Loop through the detected results and draw bounding boxes
    for box in detected_results:
        # Get coordinates and class info
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0])  # Class ID

        # Draw the bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Get the label for the detected object
        label = f"{model.names[class_id]}: {confidence:.2f}"
        
        # Put the label above the bounding box
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the result frame with bounding boxes
    cv2.imshow('YOLO Webcam Inference', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close the windows
cap.release()
cv2.destroyAllWindows()

