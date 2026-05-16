from ultralytics import YOLO
import cv2
import os

# ==========================================
# MODEL PATH
# ==========================================

model_path = r"D:\ARMED_PERSON_DETECTION\best.pt"
model = YOLO(model_path)

# ==========================================
# VIDEO SOURCE
# ==========================================

video_path = r"D:\ARMED_PERSON_DETECTION\sample_data\test1.mp4"
# For webcam use: video_path = 0

cap = cv2.VideoCapture(video_path)

# ==========================================
# CLASS NAMES
# ==========================================

class_names = {
    0: "gun",
    1: "person",
    2: "person with a gun"
}

# ==========================================
# OUTPUT SETTINGS
# ==========================================

output_folder = r"D:\ARMED_PERSON_DETECTION\results"
os.makedirs(output_folder, exist_ok=True)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = os.path.join(output_folder, "output_detection.mp4")

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# ==========================================
# DETECTION LOOP
# ==========================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video Completed")
        break

    # YOLO Prediction
    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            label = class_names.get(class_id, "Unknown")

            # ==================================
            # COLOR BASED ON CLASS
            # ==================================

            if label == "gun":
                color = (0, 0, 255)

            elif label == "person":
                color = (0, 255, 0)

            elif label == "person with a gun":
                color = (255, 0, 0)

            else:
                color = (255, 255, 255)

            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label Text
            text = f"{label} {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # ==================================
            # ALERT LOGIC
            # ==================================

            if label == "person with a gun":
                cv2.putText(
                    frame,
                    "ALERT: ARMED PERSON DETECTED",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                print(f"ALERT: Armed Person Detected ({confidence:.2f})")

    # Show Video
    cv2.imshow("Armed Person Detection", frame)

    # Save Output
    out.write(frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# RELEASE
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()

print("====================================")
print("Detection Completed Successfully")
print(f"Saved at: {output_path}")
print("====================================")