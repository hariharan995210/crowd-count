import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8n model (auto-downloads if not already available)
model = YOLO('yolov8n')

# Define COCO class labels
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Use video file or webcam (0 = webcam)
video_path = 'people5.mp4'
cap = cv2.VideoCapture(video_path)

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.6
MAX_PEOPLE_COUNT = 350

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

    # Extract predictions
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()

    # Count detected people
    current_people_count = sum(1 for i, cid in enumerate(class_ids)
                               if COCO_CLASSES[cid] == 'person' and confidences[i] > CONFIDENCE_THRESHOLD)

    # Calculate crowd percentage
    crowd_percentage = (current_people_count / MAX_PEOPLE_COUNT) * 100 if MAX_PEOPLE_COUNT > 0 else 0

    # Draw bounding boxes for people
    for i, cid in enumerate(class_ids):
        if COCO_CLASSES[cid] == 'person' and confidences[i] > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {confidences[i]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw count and percentage
    text = f'Crowd Count: {current_people_count} | Crowd Traffic: {crowd_percentage:.2f}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = 50, 30
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Display result
    cv2.imshow('Human Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
