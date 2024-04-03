import cv2
import numpy as np

# Load face and eye detection models (pre-trained)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# Function to draw a rectangle with green color
def draw_green_rectangle(image, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color with thickness 2


# Capture video from webcam (or provide a video filename as argument)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw green rectangle around the face
        draw_green_rectangle(frame, x, y, w, h)

        # Extract the region of interest for the face
        roi_face = gray[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_face, 1.1, 3)

        for (ex, ey, ew, eh) in eyes:
            # Draw green rectangle around each eye
            draw_green_rectangle(frame, x + ex, y + ey, ew, eh)

    # Display the resulting frame
    cv2.imshow('Face and Eye Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
