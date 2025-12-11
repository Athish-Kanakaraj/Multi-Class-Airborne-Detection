import time
import cv2
from ultralytics import YOLO
from pygame import mixer

# Initialize pygame mixer for beep sound
mixer.init()
beep_sound = mixer.Sound(r"D:\PROJECT\Aerial Object Detection\buzzer.mp3")

# Load the YOLOv11 model with custom weights
model = YOLO(r'D:\PROJECT\Aerial Object Detection\Metrics\YOLOv11 Quantum\weights\best.pt')

# Class names from your dataset (based on your data.yaml)
class_names = ['Airplane', 'Bird', 'Drone', 'Helicopter']
threat_classes = ['Airplane', 'Drone', 'Helicopter']
threat_class_indices = [class_names.index(c) for c in threat_classes]

# Colors
red_color = (0, 0, 255)  # BGR format red

# Timing variables for beep control
threat_detected_start = None
beep_on = False

def draw_threat_box(img, box, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), red_color, 2)
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_color, 2)

# Replace this with your video source or image loader
cap = cv2.VideoCapture(r"D:\PROJECT\Aerial Object Detection\sample2.mp4")  # Webcam, or replace with your video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)[0]  # YOLO returns list, take first element

    threat_present = False
    # results.boxes contains bounding boxes and classes
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        if cls in threat_class_indices and conf > 0.5:
            # Draw red box with "threat" label
            draw_threat_box(frame, box.xyxy[0], 'threat')
            threat_present = True
        elif class_names[cls] == 'Bird':
            # No box for Bird, skip
            continue

    now = time.time()
    if threat_present:
        if threat_detected_start is None:
            threat_detected_start = now
        elapsed = now - threat_detected_start
        if elapsed > 0.5 and not beep_on:
            beep_sound.play(-1)  # play looping beep
            beep_on = True
    else:
        threat_detected_start = None
        if beep_on:
            beep_sound.stop()
            beep_on = False

    cv2.imshow("YOLOv11 Quantum Model Threat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if beep_on:
    beep_sound.stop()
