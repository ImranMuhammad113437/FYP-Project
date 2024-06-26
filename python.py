import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def draw_green_rectangle(image, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color with thickness 2



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        draw_green_rectangle(frame, x, y, w, h)
        roi_face = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_face, 1.1, 3)

        for (ex, ey, ew, eh) in eyes:
            draw_green_rectangle(frame, x + ex, y + ey, ew, eh)

    cv2.imshow('Face and Eye Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
