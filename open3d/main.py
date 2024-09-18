import open3d as o3d
import numpy as np
import os
import cv2
import glob

def adjust_to_upright(mesh):
    # Rotate the mesh to align Z-up (Peel3D) to Y-up (Open3D)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz(np.radians([-20, 180, 90]))
    mesh.rotate(rotation_matrix)
    return mesh

def manual_adjustment(mesh):
    # Visualization for manual orientation adjustment
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    print("Adjust the orientation of the mesh manually using the Open3D viewer.")
    print("Press 'Q' to quit the viewer when you're done.")
    vis.run()
    vis.destroy_window()

def capture_images(mesh, output_dir, num_images, angle_step):
    # Create a visualizer for capturing images
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)

    for i in range(num_images):
        current_angle = i * angle_step
        rotation_matrix = mesh.get_rotation_matrix_from_xyz(np.radians([np.random.uniform(-6, 6), current_angle, 0]))
        mesh.rotate(rotation_matrix)
        
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        image_path = os.path.join(output_dir, f"image{i+1:03d}.png")
        vis.capture_screen_image(image_path)
        print(f"{i+1}/{num_images} image saved...")

    vis.destroy_window()
    print(f"All {num_images} images saved in {output_dir}.")

def annotate_images(image_dir, annotation_dir, visualization_dir, image_width, image_height):
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    image_files = glob.glob(f"{image_dir}/*.png")

    for image_file in image_files:
        print(f"Processing {image_file}")
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
        if max_bbox:
            annotation_file = os.path.join(annotation_dir, os.path.basename(image_file).replace('.png', '.txt'))
            x, y, w, h = max_bbox
            center_x = (x + w / 2) / image_width
            center_y = (y + h / 2) / image_height
            width = w / image_width
            height = h / image_height
            with open(annotation_file, 'w') as f:
                f.write(f"0 {center_x} {center_y} {width} {height}\n")

            # Visualize the max bounding box with a green rectangle
            annotated_image = image.copy()
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vis_image_file = os.path.join(visualization_dir, os.path.basename(image_file))
            cv2.imwrite(vis_image_file, annotated_image)

            print(f"Max bounding box annotations and visualization saved for {os.path.basename(image_file)}")

    print(f"All images annotated and saved in {annotation_dir} and {visualization_dir}.")

# Paths and parameters
obj_file_path = '/home/zohaib/pytorch3d-renderer/peel_3d_objects/Scan Models/Sprite Can/Sprite can.obj'
image_output_dir = "smooth_motion_images"
output_annotation_dir = "yolo_annotations"
output_visualization_dir = "annotated_images"
num_images = 150
image_width = 1920
image_height = 1080

# Ensure output directories exist
os.makedirs(image_output_dir, exist_ok=True)

# Load and adjust the mesh
mesh = o3d.io.read_triangle_mesh(obj_file_path, True)
mesh.compute_vertex_normals()
mesh = adjust_to_upright(mesh)
manual_adjustment(mesh)

# Capture images of the mesh
angle_step = 360 / num_images
capture_images(mesh, image_output_dir, num_images, angle_step)

# Annotate images with bounding boxes
annotate_images(image_output_dir, output_annotation_dir, output_visualization_dir, image_width, image_height)

