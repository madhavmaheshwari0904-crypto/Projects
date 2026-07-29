import numpy as np
import cv2
import imutils
import datetime

cascade_path = "cascade.xml"
gun_cascade = cv2.CascadeClassifier(cascade_path)
if gun_cascade.empty():
    raise FileNotFoundError(f"Failed to load cascade file: {cascade_path}")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Cannot open the camera")

gun_exit = False
window_name = "Security feed"
print("Press 'q' to quit")

confirm_frames = 0
confirm_required = 5
no_confirm_frames = 0
no_confirm_required = 5

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame from camera")
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    guns = gun_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=10,
        minSize=(140, 140),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    valid_guns = [
        (x, y, w, h)
        for (x, y, w, h) in guns
        if w * h >= 10000 and w >= 100 and h >= 100
    ]

    if len(valid_guns) > 0:
        confirm_frames += 1
        no_confirm_frames = 0
    else:
        no_confirm_frames += 1
        confirm_frames = 0

    if confirm_frames >= confirm_required and not gun_exit:
        print("guns detected")
        gun_exit = True

    if no_confirm_frames >= no_confirm_required and gun_exit:
        print("guns didn't detected")
        gun_exit = False

    for (x, y, w, h) in valid_guns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()

