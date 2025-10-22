'''
Converts image of circuit into waypoints for RaceTrack object
'''

# track_from_image.py

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def extract_track_from_image(
        image_path,
        resize_width=800,
        invert=False,
        blur_kernel=5,
        skeletonize_track=True,
        smooth_track=True,
        plot_steps=False):
    '''
    Extracts an ordered set of waypoints from a top-down racetrack image.

    Steps:
      1. Load image and preprocess (grayscale, resize, threshold)
      2. Detect binary mask of track
      3. Skeletonize to get 1-pixel-wide centerline
      4. Extract coordinates of the centerline pixels
      5. Order them to form a continuous loop
      6. Smooth the path and resample evenly
      7. Return waypoints (N x 2)
    '''

    # ---- Step 1. Load image ----
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Resize for consistency
    h, w = img.shape
    new_h = int(resize_width * (h / w))
    img = cv2.resize(img, (resize_width, new_h))

    if plot_steps:
        plt.imshow(img, cmap='gray')
        plt.title("Original grayscale")
        plt.show()

    # ---- Step 2. Threshold / binarize ----
    if invert:
        img = cv2.bitwise_not(img)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    binary = (binary > 0).astype(np.uint8)

    if blur_kernel:
        binary = cv2.medianBlur(binary * 255, blur_kernel)
        binary = (binary > 0).astype(np.uint8)

    if plot_steps:
        plt.imshow(binary, cmap='gray')
        plt.title("Binary mask")
        plt.show()

    # ---- Step 3. Skeletonize ----
    if skeletonize_track:
        skeleton = skeletonize(binary)
    else:
        skeleton = binary

    if plot_steps:
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeletonized centerline")
        plt.show()

    # ---- Step 4. Extract coordinates ----
    ys, xs = np.nonzero(skeleton)
    coords = np.stack([xs, ys], axis=1).astype(float)

    # ---- Step 5. Order points into a path ----
    waypoints = order_points_nearest_neighbor(coords)
    if smooth_track:
        waypoints = smooth_path(waypoints, n_points=1000)

    # ---- Step 6. Normalize coordinates ----
    waypoints -= waypoints.mean(axis=0)  # center
    waypoints[:, 1] *= -1                # flip Y axis for visualization consistency

    if plot_steps:
        plt.plot(waypoints[:,0], waypoints[:,1], '-r')
        plt.axis('equal')
        plt.title("Extracted track")
        plt.show()

    return waypoints

# Helper Functions
def order_points_nearest_neighbor(points):
    """Greedy nearest-neighbor traversal to order scattered skeleton points."""
    points = points.copy()
    ordered = [points[0]]
    points = np.delete(points, 0, axis=0)

    tree = cKDTree(points)
    while len(points) > 0:
        dist, idx = tree.query(ordered[-1], k=1)
        next_pt = points[idx]
        ordered.append(next_pt)
        points = np.delete(points, idx, axis=0)
        if len(points) > 0:
            tree = cKDTree(points)

    return np.array(ordered)

def smooth_path(pts, n_points=800):
    """Spline smooth and resample an ordered path."""
    tck, u = splprep(pts.T, s=5.0, per=True)
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack([x_new, y_new], axis=1)


