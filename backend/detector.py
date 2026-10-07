from ultralytics import YOLO


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.1


# --------------------------------------------------
# Detect objects
# --------------------------------------------------

def detect_objects(image_path):

    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Run detection
    results = model(
        image_path,
        conf=CONFIDENCE_THRESHOLD
    )

    detections = []

    for result in results:

        for box in result.boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = result.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    return detections


# --------------------------------------------------
# Test detector.py independenlty
# --------------------------------------------------

# if __name__ == "__main__":

#     RGB_PATH = "data/rgb/01_LivingHigh.png"

#     detections = detect_objects(RGB_PATH)

#     print("\nDetected Objects")
#     print("----------------")

#     for detection in detections:

#         print(
#             f"{detection['class']:10s} "
#             f"confidence={detection['confidence']:.2f} "
#             f"bbox={detection['bbox']}"
#         )