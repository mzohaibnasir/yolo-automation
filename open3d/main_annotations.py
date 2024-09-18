import cv2
import numpy as np
import os
import glob

# Parameters
image_width = 1920
image_height = 1080
output_annotation_dir = "train/labels"
output_visualization_dir = "annotated_images"
os.makedirs(output_annotation_dir, exist_ok=True)
os.makedirs(output_visualization_dir, exist_ok=True)

def get_class_mapping(root_dir):
    """Generate a mapping of class names to IDs based on subdirectories."""
    class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    class_mapping = {class_name: idx for idx, class_name in enumerate(class_names)}
    return class_mapping

def get_class_id(class_name, class_mapping):
    """Get the class ID from the class name based on the class mapping."""
    return class_mapping.get(class_name, -1)  # -1 indicates an unknown class

def create_data_yaml(class_mapping, yaml_path):
    """Create a data.yaml file for YOLO format."""
    with open(yaml_path, 'w') as f:

        f.write("train: open3d/train\n")
        f.write("val: open3d/train/\n")
        f.write("nc: {}\n".format(len(class_mapping)))
        f.write("names:\n")
        for class_name, class_id in class_mapping.items():
            f.write("  {}: {}\n".format(class_id, class_name))

# Directory containing subdirectories for each class
models_directory = 'models_/'
class_mapping = get_class_mapping(models_directory)

# Create data.yaml file
yaml_file_path = 'data.yaml'
create_data_yaml(class_mapping, yaml_file_path)

train_path = 'train/images'
# Load all images
image_files = glob.glob("train/images/*.png")

for image_file in image_files:
    # Extract class name from filename
    class_name = os.path.basename(image_file).split('_')[0]
    class_id = get_class_id(class_name, class_mapping)

    if class_id == -1:
        print(f"Unknown class for {image_file}, skipping.")
        continue

    print(f"Processing {image_file} with class {class_name} (ID {class_id})")

    # Read image
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to track the largest bounding box
    max_area = 0
    largest_bbox = None

    # Iterate through contours to find the largest bounding box
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            largest_bbox = (x, y, w, h)

    if largest_bbox:
        x, y, w, h = largest_bbox

        # Convert to YOLO format
        center_x = (x + w / 2) / image_width
        center_y = (y + h / 2) / image_height
        width = w / image_width
        height = h / image_height

        # Prepare annotation file
        annotation_file = os.path.join(output_annotation_dir, os.path.basename(image_file).replace('.png', '.txt'))
        with open(annotation_file, 'w') as f:
            f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

        # Visualize bounding box
        annotated_image = image.copy()
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save visualization image
        vis_image_file = os.path.join(output_visualization_dir, os.path.basename(image_file))
        cv2.imwrite(vis_image_file, annotated_image)

        print(f"Annotations and visualization saved for {os.path.basename(image_file)}")
    else:
        print(f"No suitable bounding box found in {image_file}")

print(f"All images annotated and saved in {output_annotation_dir} and {output_visualization_dir}.")
print(f"Data YAML file saved at {yaml_file_path}.")

