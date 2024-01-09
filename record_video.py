import cv2
import time

time.sleep(5)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'fast_mo.avi', fourcc, 30.0, (640, 480))
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
i = 0
while i < 200:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        exit()

    out.write(frame)
    cv2.imshow("Frame", frame)
    i += 1

cap.release()

