import open3d as o3d
# from calibration import get_calibration

# calibration = get_calibration()

# --------------------------------------------------
# TUM RGB-D camera parameters
# --------------------------------------------------

# WIDTH = 640
# HEIGHT = 480

# FX = 525.0
# FY = 525.0
# CX = 319.5
# CY = 239.5

# DEPTH_SCALE = 5000.0


# --------------------------------------------------
# Create point cloud
# --------------------------------------------------

def create_point_cloud(rgb_path, depth_path, calibration):

    # Create camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        calibration.width,
        calibration.height,
        calibration.fx,
        calibration.fy,
        calibration.cx,
        calibration.cy
    )

    # Read RGB image
    color = o3d.io.read_image(rgb_path)

    # Read depth image
    depth = o3d.io.read_image(depth_path)

    # Create RGB-D image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=calibration.depth_scale,
        depth_trunc=calibration.depth_trunc,
        convert_rgb_to_intensity=False
    )

    # Convert RGB-D image into a 3D point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic
    )

    return point_cloud


# --------------------------------------------------
# Test pointcloud.py independently
# --------------------------------------------------

# if __name__ == "__main__":

#     RGB_PATH = "data/rgb/01_LivingHigh.png"
#     DEPTH_PATH = "data/depth/01_LivingDepth.png"

#     point_cloud = create_point_cloud(
#         RGB_PATH,
#         DEPTH_PATH
#     )

#     o3d.visualization.draw_geometries(
#         [point_cloud],
#         window_name="RGB-D Point Cloud"
#     )