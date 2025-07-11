

# 👥 Real-Time Human Detection and Crowd Counting with YOLOv8

This project performs real-time **human detection** and **crowd density analysis** from a video stream using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). It is designed for applications like event safety, crowd management, and people flow analysis.

## 🚀 Features

- Detects **people** in real-time from CCTV or webcam footage  
- Displays **live crowd count**  
- Shows **crowd traffic percentage** based on a configurable threshold  
- Uses **YOLOv8n** model for efficient and fast performance  
- Easy to extend for zone alerts, occupancy limits, etc.

---

## 🧠 Model

- **Model Used:** YOLOv8n (`yolov8n.pt`)
- **Detection Class:** Only `person` class is processed
- **Framework:** OpenCV + Ultralytics YOLOv8

---

## 📂 File Structure


```plaintext
.
├── people5.mp4              # Input video file (replace or use webcam)
├── detect_and_count.py      # Python script for detection & counting
├── README.md                # Project documentation
└── requirements.txt         # Dependency list


---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install ultralytics opencv-python numpy

