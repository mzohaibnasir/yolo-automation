import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(vertices), np.array(faces)

def slice_mesh(vertices, faces, plane_origin, plane_normal):
    # Calculate signed distances from vertices to the plane
    distances = np.dot(vertices - plane_origin, plane_normal)
    
    # Find faces that intersect the plane
    intersecting_faces = []
    for face in faces:
        if np.min(distances[face]) * np.max(distances[face]) <= 0:
            intersecting_faces.append(face)
    
    return np.array(intersecting_faces)

def main():
    obj_file_path = "/home/zohaib/pytorch3d-renderer/peel_3d_objects/pringles/pringles.obj"
    vertices, faces = load_obj(obj_file_path)

    output_dir = "pringles_slices"
    os.makedirs(output_dir, exist_ok=True)

    num_images = 120
    for i in range(num_images):
        # Random rotation
        angles = np.random.uniform(0, 360, size=3)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(np.radians(angles[0])), -np.sin(np.radians(angles[0]))],
                       [0, np.sin(np.radians(angles[0])), np.cos(np.radians(angles[0]))]])
        Ry = np.array([[np.cos(np.radians(angles[1])), 0, np.sin(np.radians(angles[1]))],
                       [0, 1, 0],
                       [-np.sin(np.radians(angles[1])), 0, np.cos(np.radians(angles[1]))]])
        Rz = np.array([[np.cos(np.radians(angles[2])), -np.sin(np.radians(angles[2])), 0],
                       [np.sin(np.radians(angles[2])), np.cos(np.radians(angles[2])), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        rotated_vertices = np.dot(vertices, R.T)

        # Random zoom
        zoom_factor = np.random.uniform(0.3, 2.0)
        zoomed_vertices = rotated_vertices * zoom_factor

        # Random slice
        plane_origin = np.mean(zoomed_vertices, axis=0)
        plane_normal = np.random.rand(3)
        plane_normal /= np.linalg.norm(plane_normal)
        sliced_faces = slice_mesh(zoomed_vertices, faces, plane_origin, plane_normal)

        # Visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(zoomed_vertices[sliced_faces], alpha=0.5)
        ax.add_collection3d(mesh)

        # Set axis limits
        all_pts = zoomed_vertices[sliced_faces].reshape(-1, 3)
        ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
        ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
        ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

        # Save image
        image_path = os.path.join(output_dir, f"slice_{i}.png")
        plt.savefig(image_path)
        plt.close()

        print(f"Slice {i} is done...")

    print(f"{num_images} sliced images saved in {output_dir}.")

if __name__ == "__main__":
    main()