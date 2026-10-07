import cv2
import numpy as np
import open3d as o3d


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DEPTH_PATH = "data/depth/01_LivingDepth.png"


# --------------------------------------------------
# 2. TUM RGB-D camera parameters
# --------------------------------------------------

WIDTH = 640
HEIGHT = 480

FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5

DEPTH_SCALE = 5000.0


# --------------------------------------------------
# 3. Convert a 2D bounding box into 3D points
# --------------------------------------------------

def bbox_to_3d_points(bbox, depth, calibration):

    x1, y1, x2, y2 = bbox

    # Convert coordinates to integers
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))

    x2 = min(
        calibration.width - 1,
        int(x2)
    )

    y2 = min(
        calibration.height - 1,
        int(y2)
    )

    # Extract depth values inside the bounding box
    depth_region = depth[y1:y2 + 1, x1:x2 + 1]

    # Remove invalid depth values
    valid_depth = depth_region[depth_region > 0]

    if len(valid_depth) == 0:
        return np.empty((0, 3))

    # --------------------------------------------------
    # Remove extreme depth outliers
    # --------------------------------------------------

    lower = np.percentile(valid_depth, 5)
    upper = np.percentile(valid_depth, 95)

    mask = (
        (depth_region >= lower) &
        (depth_region <= upper) &
        (depth_region > 0)
    )

    # Get pixel coordinates inside the bounding box
    v_coords, u_coords = np.where(mask)

    # Convert local coordinates to image coordinates
    u_coords = u_coords + x1
    v_coords = v_coords + y1

    # Get corresponding depth values
    z = depth[v_coords, u_coords].astype(np.float32)

    # Convert millimetres to metres
    z = z / calibration.depth_scale

    # --------------------------------------------------
    # Convert pixels to 3D coordinates
    # --------------------------------------------------

    x = ((u_coords - calibration.cx) * z / calibration.fx)

    y = ((v_coords - calibration.cy) * z / calibration.fy)

    points = np.column_stack((x, y, z))

    return points


# --------------------------------------------------
# 4. Create an Open3D 3D bounding box
# --------------------------------------------------

def create_3d_bounding_box(points):

    if len(points) == 0:
        return None

    # Convert NumPy points to Open3D point cloud
    object_cloud = o3d.geometry.PointCloud()

    object_cloud.points = o3d.utility.Vector3dVector(points)

    # Calculate axis-aligned bounding box
    bbox = object_cloud.get_axis_aligned_bounding_box()

    bbox.color = [1.0, 0.0, 0.0]

    return bbox


# # --------------------------------------------------
# # 5. Test using one example bounding box
# # --------------------------------------------------

# if __name__ == "__main__":

#     # Load depth image
#     depth = cv2.imread(
#         DEPTH_PATH,
#         cv2.IMREAD_UNCHANGED
#     )

#     print("Depth shape:", depth.shape)

#     # Example bounding box.
#     # Replace these coordinates with one of the
#     # actual YOLO detections.
#     test_bbox = [489, 138, 639, 307]

#     # Convert to 3D
#     points = bbox_to_3d_points(
#         test_bbox,
#         depth
#     )

#     print("\n3D points generated:", len(points))

#     if len(points) > 0:

#         # Create 3D bounding box
#         bbox_3d = create_3d_bounding_box(points)

#         print("\n3D Bounding Box")
#         print("----------------")

#         print("Minimum XYZ:")
#         print(bbox_3d.get_min_bound())

#         print("\nMaximum XYZ:")
#         print(bbox_3d.get_max_bound())

#         print("\nDimensions:")
#         print(bbox_3d.get_extent())

from detector import detect_objects


# --------------------------------------------------
# Create 3D bounding boxes for all detections
# --------------------------------------------------

def create_object_boxes(rgb_path, depth_path, calibration):

    # Load depth image
    depth = cv2.imread(
        depth_path,
        cv2.IMREAD_UNCHANGED
    )
    if depth is None:
        raise FileNotFoundError(
            f"Could not read depth image: {depth_path}"
        )

    # Make sure depth is a 2D single-channel image
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    print("Depth shape:", depth.shape)
    print("Depth dtype:", depth.dtype)

    # Run YOLO
    detections = detect_objects(rgb_path)

    object_boxes = []

    for detection in detections:

        # Convert 2D bounding box to 3D points
        points = bbox_to_3d_points(
            detection["bbox"],
            depth,
            calibration
        )

        # Skip if no valid depth exists
        if len(points) == 0:
            continue

        # Create 3D bounding box
        bbox_3d = create_3d_bounding_box(points)

        object_boxes.append({
            "class": detection["class"],
            "confidence": detection["confidence"],
            "bbox_2d": detection["bbox"],
            "bbox_3d": bbox_3d
        })

    return object_boxes


# --------------------------------------------------
# Test bounding_box.py independently.
# --------------------------------------------------

# if __name__ == "__main__":

#     RGB_PATH = "data/rgb/01_LivingHigh.png"
#     DEPTH_PATH = "data/depth/01_LivingDepth.png"

#     object_boxes = create_object_boxes(
#         RGB_PATH,
#         DEPTH_PATH
#     )

#     print("\n3D Object Bounding Boxes")
#     print("------------------------")

#     for obj in object_boxes:

#         print(f"\nObject: {obj['class']}")
#         print(f"Confidence: {obj['confidence']:.2f}")

#         bbox = obj["bbox_3d"]

#         print("Minimum XYZ:")
#         print(bbox.get_min_bound())

#         print("Maximum XYZ:")
#         print(bbox.get_max_bound())

#         print("Dimensions:")
#         print(bbox.get_extent())