from dataclasses import dataclass


@dataclass
class CameraCalibration:

    # Image dimensions
    width: int
    height: int

    # RGB camera intrinsic parameters
    fx: float
    fy: float
    cx: float
    cy: float

    # Depth scale
    depth_scale: float

    # Maximum depth used by Open3D
    depth_trunc: float


def get_calibration():

    print("\n==============================")
    print("   RGB-D Camera Calibration")
    print("==============================")

    print("\nImage dimensions")
    print("----------------")

    width = int(input("RGB width  : "))
    height = int(input("RGB height : "))

    print("\nRGB Camera Intrinsics")
    print("---------------------")

    fx = float(input("fx : "))
    fy = float(input("fy : "))
    cx = float(input("cx : "))
    cy = float(input("cy : "))

    print("\nDepth Configuration")
    print("-------------------")

    depth_scale = float(
        input("Depth scale (e.g. 5000): ")
    )

    depth_trunc = float(
        input("Maximum depth in metres (e.g. 3): ")
    )

    calibration = CameraCalibration(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc
    )

    print("\nCalibration loaded:")
    print("-------------------")
    print(f"Resolution   : {width} x {height}")
    print(f"fx           : {fx}")
    print(f"fy           : {fy}")
    print(f"cx           : {cx}")
    print(f"cy           : {cy}")
    print(f"Depth scale  : {depth_scale}")
    print(f"Depth trunc  : {depth_trunc} m")

    return calibration