import cv2
import numpy as np

video_files = [
    {"path": "./fast_mo.avi", "name": "Recorded Video"},
    {"path": "./output_MOG2.avi", "name": "MOG2"},
    {"path": "./output_KNN.avi", "name": "KNN"},
    {"path": "./output_CNT.avi", "name": "CNT"},
    {"path": "./output_GMG.avi", "name": "GMG"},
    {"path": "./output_GSOC.avi", "name": "GSOC"},
    {"path": "./output_LSBP.avi", "name": "LSBP"},
    {"path": "./output_MOG.avi", "name": "MOG"}
]

cv2.namedWindow("Combined Videos", cv2.WINDOW_NORMAL)

video_captures = [cv2.VideoCapture(video["path"]) for video in video_files]

output_filename = "combined_output.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter(output_filename, fourcc, 30.0, (1920, 1080))


while True:
    combined_frame_top = None
    combined_frame_bottom = None

    for i, video_capture in enumerate(video_captures):
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 600))

        name = video_files[i]["name"]
        cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if i < 4:
            if combined_frame_top is None:
                combined_frame_top = frame
            else:
                combined_frame_top = np.concatenate((combined_frame_top, frame), axis=1)
        else:
            if combined_frame_bottom is None:
                combined_frame_bottom = frame
            else:
                combined_frame_bottom = np.concatenate((combined_frame_bottom, frame), axis=1)

    combined_frame = np.concatenate((combined_frame_top, combined_frame_bottom), axis=0)

    cv2.imshow("Combined Videos", combined_frame)

    combined_frame = cv2.resize(combined_frame, (1920, 1080))
    output_video.write(combined_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

for video_capture in video_captures:
    video_capture.release()

output_video.release()

cv2.destroyAllWindows()
