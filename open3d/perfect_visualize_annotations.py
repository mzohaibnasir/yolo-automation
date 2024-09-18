import cv2
import numpy as np
import os
import glob

# Parameters
image_width = 1920
image_height = 1080
output_annotation_dir = "yolo_annotations"
output_visualization_dir = "annotated_images"
os.makedirs(output_annotation_dir, exist_ok=True)
os.makedirs(output_visualization_dir, exist_ok=True)

# Load all images
image_files = glob.glob("smooth_motion_images/*.png")

for image_file in image_files:
    # Read image
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare annotation file
    annotation_file = os.path.join(output_annotation_dir, os.path.basename(image_file).replace('.png', '.txt'))

    # Initialize variables to track the maximum bounding box
    max_area = 0
    max_bbox = None
    
    # Find the bounding box with the largest area
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            max_bbox = (x, y, w, h)

    # Write the maximum bounding box to YOLO annotation file
    if max_bbox is not None:
        with open(annotation_file, 'w') as f:
            x, y, w, h = max_bbox
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            width = w / image_width
            height = h / image_height
            # Write the max bounding box in YOLO format
            f.write(f"0 {center_x} {center_y} {width} {height}\n")

    # Visualize the max bounding box with a green rectangle
    annotated_image = image.copy()
    if max_bbox is not None:
        x, y, w, h = max_bbox
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save visualization image
    vis_image_file = os.path.join(output_visualization_dir, os.path.basename(image_file))
    cv2.imwrite(vis_image_file, annotated_image)

    print(f"Max bounding box annotations and visualization saved for {os.path.basename(image_file)}")

print(f"All images annotated and saved in {output_annotation_dir} and {output_visualization_dir}.")

