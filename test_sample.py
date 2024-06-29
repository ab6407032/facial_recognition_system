import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the LBF model for facial landmark detection
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

def load_and_prepare_image(image_path):
    # Load the image file
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None, None
    # Resize image for easier processing
    image = cv2.resize(image, (640, 480))
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def test_opencv_face_and_landmark_detection(image_path):
    image, gray_image = load_and_prepare_image(image_path)
    if image is None or gray_image is None:
        return
    
    print(f"Image shape: {gray_image.shape}, dtype: {gray_image.dtype}")

    # Detect faces
    try:
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Number of faces detected: {len(faces)}")
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Detect landmarks
        _, landmarks = facemark.fit(gray_image, faces)
        for landmark in landmarks:
            for x, y in landmark[0]:
                cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
    except Exception as e:
        print(f"Error detecting faces or landmarks: {e}")

    # Display the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image_path = 'sample.jpg'  # Replace with your test image path
    test_opencv_face_and_landmark_detection(test_image_path)
