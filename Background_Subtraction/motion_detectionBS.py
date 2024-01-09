import cv2
import numpy as np

video = cv2.VideoCapture('./c05.mp4')
fgbg = cv2.createBackgroundSubtractorKNN()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while video.isOpened():
    ret, frame = video.read()

    if ret:
        fgmask = fgbg.apply(frame)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel=np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Moving Objects', frame)
        out.write(fgmask)
        cv2.imshow('Foreground mask', fgmask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        print('Video ended')
        break

out.release()
video.release()
cv2.destroyAllWindows()
