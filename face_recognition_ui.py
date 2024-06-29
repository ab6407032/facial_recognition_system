import cv2
from tkinter import *
from PIL import Image, ImageTk

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the LBF model for facial landmark detection
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition System")
        self.root.geometry("1000x800")

        self.video_label = Label(root)
        self.video_label.pack()

        # Use DirectShow backend for video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("Error: Could not open video capture device.")
            self.root.quit()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"Grayscale frame shape: {frame_gray.shape} and dtype: {frame_gray.dtype}")  # Debug print
            
            # Detect faces
            try:
                faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                print(f"Number of faces detected: {len(faces)}")
                if len(faces) > 0:
                    try:
                        _, landmarks = facemark.fit(frame_gray, faces)
                        print(f"Number of landmarks detected: {len(landmarks)}")
                        for landmark in landmarks:
                            for x, y in landmark[0]:
                                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                    except Exception as e:
                        print(f"Error detecting landmarks: {e}")
                else:
                    print("No faces detected.")
            except Exception as e:
                print(f"Error detecting faces: {e}")

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
        else:
            print("Error: Could not read frame.")
        
        self.root.after(10, self.update_frame)

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
