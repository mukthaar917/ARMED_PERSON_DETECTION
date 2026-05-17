# ARMED_PERSON_DETECTION
AI-based surveillance system using YOLOv8 to detect guns, persons, and armed individuals in real time. Processes video input (CCTV/MP4/webcam), generates bounding boxes, and triggers alerts for potential threats. Built with Python, OpenCV, and PyTorch.
# AI-Based Intelligent Camera Security System (Armed Person Detection)

## 1. Project Overview

This project presents an AI-based intelligent surveillance system designed to detect armed individuals in real-time using computer vision and deep learning. The system processes video input (CCTV, webcam, or recorded footage) and identifies objects such as guns, persons, and persons carrying weapons.

---

## 2. Problem Statement

Traditional surveillance systems rely on manual monitoring, which is inefficient and error-prone. This project aims to automate threat detection by identifying armed individuals in real-time to enhance security and response time.

---

## 3. Dataset Used

* Custom dataset created using Roboflow
* Classes:

  * Gun
  * Person
  * Person with a Gun
* Dataset split:

  * Train
  * Validation
  * Test

---

## 4. System Architecture

1. Input Video (CCTV / MP4 / Webcam)
2. Frame Extraction using OpenCV
3. YOLOv8 Model Inference
4. Object Detection (Gun, Person, Armed Person)
5. Alert Generation
6. Output Video with Bounding Boxes

---

## 5. Model Selection

* Model: YOLOv8 (Ultralytics)
* Reason:

  * Real-time performance
  * High accuracy
  * Easy deployment
* Trained using Roboflow Object Detection pipeline

---

## 6. Installation

```bash
pip install -r requirements.txt
```

---

## 7. How to Run

```bash
python detect.py
```

For webcam:

```python
video_path = 0
```

---

## 8. Training Details

* Platform: Roboflow
* Model Type: Object Detection (Fast)
* Classes: 3
* Training approach: Transfer learning (pretrained weights)
* Augmentation: Applied via Roboflow

* <img width="1007" height="492" alt="2" src="https://github.com/user-attachments/assets/ac320bf7-543f-4b21-8c71-f96c713cdfa9" />



  <img width="896" height="486" alt="3" src="https://github.com/user-attachments/assets/33e47331-902f-402b-ac96-278afbe89882" />

---

## 9. Inference Pipeline

* Load trained model (`best.pt`)
* Capture frames using OpenCV
* Run YOLO inference
* Draw bounding boxes and labels
* Generate alert for "person with a gun"
* Save output video

---

## 10. Results

* Successfully detects:

  * Gun
  * Person
  * Armed person
* Output:

  * Annotated video with bounding boxes
  * Alert message displayed on detection

---

## 11. Evaluation Metrics

* mAP@50: 68.1%
* Precision: 53.3%
* Recall: 80.5%
* F1 Score: 64.2%

<img width="991" height="342" alt="1" src="https://github.com/user-attachments/assets/16edcdee-74a1-46d6-a5cd-3ab09f25a53c" />

<img width="1006" height="329" alt="4" src="https://github.com/user-attachments/assets/84646e77-03be-4ccd-9828-a9754adc9caa" />

---

## 12. Limitations

* Limited dataset size
* False positives in crowded scenes
* Performance may drop in low lighting
* No multi-object relationship reasoning

---

## 13. Future Improvements

* Increase dataset size and diversity
* Add real-time alert system (email/SMS)
* Deploy on edge devices (CCTV cameras)
* Integrate tracking (DeepSORT)
* Improve accuracy using custom training

---

## 14. Model Download

* Model file: `best.pt`
  OR
* Provide download link (Roboflow / Google Drive)

---

## 15. Results Folder

Contains:

* Output video
* Detection screenshots

<img width="1309" height="767" alt="Screenshot 1" src="https://github.com/user-attachments/assets/11c499e7-7c81-4cc7-9a0c-994e0398b7ba" />

---

## 16. Requirements

```
ultralytics
opencv-python
numpy==1.26.4
torch
torchvision
```

---

## 17. Technical Explanation

The system uses YOLOv8 (You Only Look Once), a deep learning-based object detection algorithm that performs detection in a single forward pass. It divides the image into regions and predicts bounding boxes and class probabilities, enabling real-time detection.

---

## 18. References

* YOLOv8 Documentation (Ultralytics)
* Roboflow Dataset Platform
* OpenCV Documentation

---

## 19. Assumptions

* Input video quality is sufficient
* Objects are visible in the frame
* Model trained on relevant dataset

---
