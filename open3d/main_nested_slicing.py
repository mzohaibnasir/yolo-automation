import open3d as o3d
import numpy as np
import os
import random

def adjust_to_upright(mesh):
    # Rotate the mesh to align Z-up (Peel3D) to Y-up (Open3D)
    rotation_matrix_x = mesh.get_rotation_matrix_from_xyz(np.radians([-90, 0, 0]))
    # Apply an additional rotation for manual adjustment
    rotation_matrix_x = mesh.get_rotation_matrix_from_xyz(np.radians([-20, 180, 90]))
    mesh.rotate(rotation_matrix_x)
    return mesh

def manual_adjustment(mesh):
    # Visualization for manual orientation adjustment
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    # Display instructions
    print("Adjust the orientation of the mesh manually using the Open3D viewer.")
    print("Press 'Q' to quit the viewer when you're done.")
    # Run the visualization
    vis.run()
    vis.destroy_window()

def process_meshes_in_directory(root_dir):
    """Process all .obj files in the directory and its subdirectories."""
    # Resolve root directory to absolute path
    root_dir = os.path.abspath(root_dir)

    # Prepare output directory
    output_dir = os.path.abspath('./train/images/')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_file_path = os.path.join(subdir, file)
                print(f"Processing {obj_file_path}")

                # Load the mesh
                mesh = o3d.io.read_triangle_mesh(obj_file_path, True)
                mesh.compute_vertex_normals()

                # Adjust the mesh to an upright position
                mesh = adjust_to_upright(mesh)

                # Perform manual orientation adjustment
                manual_adjustment(mesh)

                # Parameters
                num_images = 150  # Number of images to capture
                angle_step = 360 / num_images  # Step size for rotation in degrees

                # Create a visualizer for capturing images
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(mesh)

                # Extract base filename without extension
                base_filename = os.path.splitext(file)[0]

                # Loop through each image
                for i in range(num_images):
                    # Apply the rotation incrementally for smooth continuous rotation
                    current_angle = i * angle_step
                    rotation_matrix = mesh.get_rotation_matrix_from_xyz(np.radians([random.uniform(-6, 6), current_angle, 0]))  # Rotate around y-axis
                    mesh.rotate(rotation_matrix)

                    # Update the visualization and save the image
                    vis.update_geometry(mesh)
                    vis.poll_events()
                    vis.update_renderer()
                    image_filename = f"{base_filename}_{i+1:02d}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    vis.capture_screen_image(image_path)
                    print(f"{i+1:02d}/{num_images} image saved... {image_path}")

                # Clean up
                vis.destroy_window()
                print(f"All {num_images} images saved in {output_dir}.")

# Example usage
root_directory = 'models_'  # Update the directory if needed
process_meshes_in_directory(root_directory)
















'''
import open3d as o3d
import numpy as np
import os
import random

def adjust_to_upright(mesh):
    # Rotate the mesh to align Z-up (Peel3D) to Y-up (Open3D)
    rotation_matrix_x = mesh.get_rotation_matrix_from_xyz(np.radians([-90, 0, 0]))
    rotation_matrix_x = mesh.get_rotation_matrix_from_xyz(np.radians([-20, 2*90,90]))
    mesh.rotate(rotation_matrix_x)
    return mesh

def manual_adjustment(mesh):
    # Visualization for manual orientation adjustment
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    # Display instructions
    print("Adjust the orientation of the mesh manually using the Open3D viewer.")
    print("Press 'Q' to quit the viewer when you're done.")
    # Run the visualization
    vis.run()
    vis.destroy_window()



def process_meshes_in_directory(root_dir):
    """Process all .obj files in the directory and its subdirectories."""
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_file_path = os.path.join(subdir, file)
                print(f"Processing {obj_file_path}")


                # Load the mesh
                mesh = o3d.io.read_triangle_mesh(obj_file_path, True)
                mesh.compute_vertex_normals()

                # Adjust the mesh to an upright position
                mesh = adjust_to_upright(mesh)

                # Perform manual orientation adjustment
                manual_adjustment(mesh)

                # Prepare output directory
                output_dir = './train/images/'
                os.makedirs(output_dir, exist_ok=True)

                # Parameters
                num_images = 150  # Number of images to capture
                angle_step = 360 / 3600  # Step size for rotation in degrees

                # Create a visualizer for capturing images
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(mesh)

                # Extract base filename without extension
                base_filename = os.path.splitext(file)[0]

                # Loop through each image
                for i in range(num_images):
                    print(os.getcwd())
                    # Apply the rotation incrementally for smooth continuous rotation
                    current_angle = i * angle_step
                    rotation_matrix = mesh.get_rotation_matrix_from_xyz(np.radians([random.uniform(-6,6), current_angle, 0]))  # Rotate around y-axis
                    mesh.rotate(rotation_matrix)

                    # Update the visualization and save the image
                    vis.update_geometry(mesh)
                    vis.poll_events()
                    vis.update_renderer()
                    image_filename = f"{base_filename}_{i+1:02d}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    vis.capture_screen_image(image_path)
                    print(f"{i+1:02d}/{num_images} image saved... {image_path}")

                # Clean up
                vis.destroy_window()
                print(f"All {num_images} images saved in {output_dir}.")

# Example usage
root_directory = 'models_'
process_meshes_in_directory(root_directory)
'''
