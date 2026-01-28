import os
import cv2
import numpy as np
class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces_dir = "known_faces"
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_path = os.path.join(self.output_dir, "lbph_model.yml")
        self.label_map = {}
        self.is_trained = False
    def read_image(self, path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except:
            return None
    def detect_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        return faces, gray
    def train_model(self):
        X, y = [], []
        self.label_map = {}
        label_id = 0
        if not os.path.exists(self.known_faces_dir):
            print(f"Folder not found: {self.known_faces_dir}")
            return False
        for person_name in sorted(os.listdir(self.known_faces_dir)):
            person_path = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_path):
                continue
            self.label_map[label_id] = person_name
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = self.read_image(img_path)
                if img is None:
                    continue
                faces, gray = self.detect_faces(img)
                if len(faces) == 0:
                    continue
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y1, w, h = faces[0]
                face_roi = gray[y1:y1 + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))
                X.append(face_roi)
                y.append(label_id)
            label_id += 1
        if len(X) == 0:
            print(" No faces found for training.")
            return False
        self.recognizer.train(X, np.array(y))
        self.recognizer.save(self.model_path)
        self.is_trained = True
        return True
    def load_model(self):
        if self.is_trained:
            return True
        if os.path.exists(self.model_path):
            self.recognizer.read(self.model_path)
            if not self.label_map:
                return self.train_model()
            return True
        return self.train_model()
    def save_face_crop(self, face_img, person_name, count):
        filename = f"{person_name}_{count}.jpg"
        out_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(out_path, face_img)
    def recognize_and_draw(self, frame, save_crops=False):
        faces, gray = self.detect_faces(frame)
        recognized_names = set()
        person_counter = {} 
        for (x, y1, w, h) in faces:
            roi_gray = gray[y1:y1 + h, x:x + w]
            roi_color = frame[y1:y1 + h, x:x + w] 
            roi_gray = cv2.resize(roi_gray, (200, 200))
            label_id, confidence = self.recognizer.predict(roi_gray)
            name = self.label_map.get(label_id, "Unknown")
            if confidence > 120:
                name = "Unknown"
            recognized_names.add(name)
            if save_crops:
                if name not in person_counter:
                    person_counter[name] = 1
                else:
                    person_counter[name] += 1
                self.save_face_crop(roi_color, name, person_counter[name])
            cv2.rectangle(frame, (x, y1), (x + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame, recognized_names
    def image_mode(self):
        if not self.load_model():
            return
        img_path = input("\nEnter image path: ").strip().strip('"')
        if not os.path.exists(img_path):
            print(" Image not found!")
            return
        img = self.read_image(img_path)
        if img is None:
            print(" Unable to read image!")
            return
        result, names = self.recognize_and_draw(img, save_crops=True)
        known_names = [n for n in names if n != "Unknown"]
        if known_names:
            print("\nIn the above image the persons whose images are known are:", ", ".join(known_names))
        else:
            print("\nIn the above image no known faces are matched.")
        print("Cropped faces saved in output folder.")
        cv2.imshow("Face Recognition - IMAGE", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def video_mode(self):
        if not self.load_model():
            return
        video_path = input("\nEnter video path: ").strip().strip('"')
        if not os.path.exists(video_path):
            print(" Video not found!")
            return
        cap = cv2.VideoCapture(video_path)
        print("\n🎥 Video running... Press 'q' to stop")
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            save_now = (frame_no % 30 == 0)
            result, _ = self.recognize_and_draw(frame, save_crops=save_now)
            cv2.imshow("Face Recognition - VIDEO", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Video completed. Cropped faces saved in output folder.")
    def run(self):
        print("\nFace Detection & Recognition")
        while True:
            print("\nSelect an option:")
            print("1. Image")
            print("2. Video")
            print("3. Exit")
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == "1":
                self.image_mode()
            elif choice == "2":
                self.video_mode()
            elif choice == "3":
                print("Exiting...")
                break
            else:
                print(" Invalid choice! Try again.")
system = FaceRecognitionSystem()
system.run()
