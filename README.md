##Face Detection & Recognition System

This project is a simple AI-based application that detects and recognizes faces in images or videos using **Haar Cascade (OpenCV)** for face detection and **LBPH (Local Binary Pattern Histogram)** for face recognition.

---

## Features

- Detect faces in **images**
- Detect faces in **videos**
- Recognize trained (known) persons
- Display names on detected faces
- Simple menu-driven program (Image / Video / Exit)

---

## Technologies Used

- Python  
- OpenCV  
- NumPy  

---

## Project Structure
TASK-5_FACE-DETECTION-RECOGNITION/
│
├── known_faces/ # Subfolders of each person's images
│ ├── Person1/
│ ├── Person2/
│
├── output/ # Stores trained model (lbph_model.yml)
├── main.py # Main program file
└── README.md # Project documentation

---

## How It Works

1. The program reads face images from the **known_faces** folder.  
2. Faces are detected using **Haar Cascade**.  
3. The **LBPH algorithm** learns facial patterns.  
4. When testing, detected faces are matched with trained faces.  
5. Recognized faces are labeled with names on the screen.  

---

## Installation

Install required libraries:

```bash
pip install opencv-contrib-python numpy
```
## How to Run
```bash
python main.py
```
You will see:

1. Image
2. Video
3. Exit
## 🖼 Image Mode

- Choose **1**
- Enter the full path of the test image
- Faces will be detected and recognized

---

## Video Mode

- Choose **2**
- Enter the full path of the video
- Faces will be detected and recognized frame by frame

---

## Example Inputs

- A group photo containing known persons
- A short video with known faces

---

## Notes

- Add **multiple clear images per person** inside `known_faces`
- Good lighting improves recognition accuracy
- Model training happens automatically during first run

---

## Output

- Trained model file saved in `output/lbph_model.yml`
- Recognized face names shown on screen

---

## Author

**Samatham Sai Deepthi**

