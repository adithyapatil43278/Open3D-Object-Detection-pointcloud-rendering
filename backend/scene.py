from PIL import Image, ImageDraw, ImageFont
import numpy as np
import open3d as o3d
# import open3d.visualization.gui as gui

from pointcloud import create_point_cloud
from bounding_box import create_object_boxes
from calibration import get_calibration


# --------------------------------------------------
# File paths
# --------------------------------------------------

RGB_PATH = "data/rgb/01_LivingHigh.png"
DEPTH_PATH = "data/depth/01_LivingDepth.png"


# --------------------------------------------------
# Convert Open3D bounding box into a visible LineSet
# --------------------------------------------------

# def bounding_box_to_lineset(bbox):

#     # Get the 8 corner points of the bounding box
#     points = bbox.get_box_points()

#     # Convert to Open3D vector
#     points = o3d.utility.Vector3dVector(points)

#     # 12 edges of a rectangular box
    # lines = [
    #     # Bottom face
    #     [0, 1],
    #     [1, 7],
    #     [7, 2],
    #     [2, 0],

    #     # Top face
    #     [3, 6],
    #     [6, 4],
    #     [4, 5],
    #     [5, 3],

    #     # Vertical edges
    #     [0, 3],
    #     [1, 6],
    #     [2, 5],
    #     [7, 4]
    # ]

#     # Create LineSet
#     line_set = o3d.geometry.LineSet()

#     line_set.points = points
#     line_set.lines = o3d.utility.Vector2iVector(lines)

#     # Make every line red
#     colors = [[1.0, 0.0, 0.0] for _ in lines]

#     line_set.colors = o3d.utility.Vector3dVector(colors)

#     return line_set

def bounding_box_to_lineset(bbox):

    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        bbox
    )

    line_set.paint_uniform_color([1.0, 0.0, 0.0])

    return line_set


#########################################################################
def create_text_point_cloud(
    text,
    position,
    scale=0.001
):
    """
    Creates a small 3D point cloud representing text.
    """

    font = ImageFont.truetype(
        "arial.ttf",
        50
    )

    # Determine text size
    bbox = font.getbbox(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Create black image
    image = Image.new(
        "L",
        (width + 10, height + 10),
        0
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (5, 5),
        text,
        fill=255,
        font=font
    )

    # Find pixels belonging to text
    pixels = np.array(image)

    y_coords, x_coords = np.where(pixels > 0)

    # Convert pixels to 3D coordinates
    points = []

    for x, y in zip(x_coords, y_coords):

        px = position[0] + x * scale
        py = position[1] - y * scale
        pz = position[2]

        points.append([px, py, pz])

    # Create Open3D point cloud
    text_cloud = o3d.geometry.PointCloud()

    text_cloud.points = o3d.utility.Vector3dVector(
        points
    )

    # Give text a color
    text_cloud.paint_uniform_color(
        [1.0, 1.0, 0.0]
    )

    return text_cloud


# --------------------------------------------------
# Create complete 3D scene
# --------------------------------------------------

# def create_scene(): # this one uses draw_geometries (legacy) to render the point cloud
#     calibration = get_calibration()

#     print("\nCreating point cloud...")

#     point_cloud = create_point_cloud(
#         RGB_PATH,
#         DEPTH_PATH,
#         calibration
#     )

#     print("Point cloud created.")

#     print("\nDetecting objects and creating 3D boxes...")

#     object_boxes = create_object_boxes(
#         RGB_PATH,
#         DEPTH_PATH,
#         calibration
#     )

#     print(
#         f"\nNumber of 3D boxes: {len(object_boxes)}"
#     )

#     geometries = [point_cloud]

#     # Add bounding boxes
#     for obj in object_boxes:

#         bbox = obj["bbox_3d"]

#         print(
#             f"Adding box: "
#             f"{obj['class']} "
#             f"(confidence={obj['confidence']:.2f})"
#         )

#         line_set = bounding_box_to_lineset(bbox)

#         geometries.append(line_set)

#     # Add coordinate frame for reference
#     coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.3,
#         origin=[0, 0, 0]
#     )

#     geometries.append(coordinate_frame)

#     # --------------------------------------------------
#     # Display
#     # --------------------------------------------------

#     print("\nOpening Open3D visualization...")

#     o3d.visualization.draw_geometries(
#         geometries,
#         window_name="3D Object Detection",
#         width=1280,
#         height=720
#     )

def create_scene(): # 3D rendering of point cloud with text labels on the bounding boxes

    calibration = get_calibration()

    print("\nCreating point cloud...")

    point_cloud = create_point_cloud(
        RGB_PATH,
        DEPTH_PATH,
        calibration
    )

    print("Point cloud created.")

    print("\nDetecting objects and creating 3D boxes...")

    object_boxes = create_object_boxes(
        RGB_PATH,
        DEPTH_PATH,
        calibration
    )

    print(
        f"\nNumber of 3D boxes: {len(object_boxes)}"
    )

    geometries = [point_cloud]

    # Add bounding boxes + text
    for obj in object_boxes:

        bbox = obj["bbox_3d"]

        print(
            f"Adding box: "
            f"{obj['class']} "
            f"(confidence={obj['confidence']:.2f})"
        )

        # -------------------------
        # Bounding box
        # -------------------------

        line_set = bounding_box_to_lineset(bbox)

        geometries.append(line_set)

        # -------------------------
        # Text label
        # -------------------------

        center = bbox.get_center()
        max_bound = bbox.get_max_bound()

        label_position = [
            center[0],
            max_bound[1] + 0.05,
            center[2]
        ]

        label_text = (
            f"{obj['class']} "
            f"{obj['confidence']:.2f}"
        )

        text_cloud = create_text_point_cloud(
            label_text,
            label_position
        )

        geometries.append(text_cloud)

    # Coordinate frame
    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3,
            origin=[0, 0, 0]
        )
    )

    geometries.append(coordinate_frame)

    # Display
    print("\nOpening Open3D visualization...")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="3D Object Detection",
        width=1280,
        height=720
    )

# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    create_scene()