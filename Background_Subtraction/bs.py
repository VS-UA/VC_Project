import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil

#cap = cv2.VideoCapture(0) #tirar de comentário para usar a webcam

#fourcc = cv2.VideoWriter_fourcc(*'XVID')

background_subtractors = [cv2.createBackgroundSubtractorMOG2(),
                            cv2.createBackgroundSubtractorKNN(),
                            cv2.bgsegm.createBackgroundSubtractorCNT(),
                            cv2.bgsegm.createBackgroundSubtractorGMG(),
                            cv2.bgsegm.createBackgroundSubtractorGSOC(),
                            cv2.bgsegm.createBackgroundSubtractorLSBP(),
                            cv2.bgsegm.createBackgroundSubtractorMOG()
                            ]

subtractor_names = ["MOG2", "KNN", "CNT", "GMG", "GSOC", "LSBP", "MOG"]

ground_truth_path = "./background2.png"
ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

if ground_truth.shape[0] != 480 and ground_truth.shape[1] != 640:
    ground_truth = cv2.resize(ground_truth, (640, 480))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

true_positives = []
false_positives = []
true_negatives = []
false_negatives = []

precisions = []
recalls = []
accuracies = []
f1_scores = []

memory_usages = []
time_usages = []

for i in range(len(background_subtractors)):
    video_path = './fast_mo.avi'
    cap = cv2.VideoCapture(video_path)
    background_subtractor = background_subtractors[i]

    out = cv2.VideoWriter(f'output_{subtractor_names[i]}.avi', fourcc, 30.0, (640, 480))

    #num_of_processed_frames = 0

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    start = time.time()
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        fg_mask = background_subtractor.apply(frame)

        _, binary_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

        colored_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        ground_truth_matlike = ground_truth.astype(binary_mask.dtype)
        ground_truth_matlike = cv2.threshold(ground_truth_matlike, 128, 255, cv2.THRESH_BINARY)[1]

        true_positive += np.sum(np.logical_and(binary_mask == 255, ground_truth_matlike == 255))
        false_positive += np.sum(np.logical_and(binary_mask == 255, ground_truth_matlike == 0))
        true_negative += np.sum(np.logical_and(binary_mask == 0, ground_truth_matlike == 0))
        false_negative += np.sum(np.logical_and(binary_mask == 0, ground_truth_matlike == 255))

        cv2.imshow('Foreground Mask', fg_mask)
        out.write(colored_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        #num_of_processed_frames += 1
        #if num_of_processed_frames == 200: # o GMG sem parâmetros só funciona com mais de 100 frames
            #break
    end = time.time()
    out.release()
    
    print("True Positive:", true_positive)
    print("False Positive:", false_positive)
    print("True Negative:", true_negative)
    print("False Negative:", false_negative)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative) if (true_positive + false_positive + true_negative + false_negative) > 0 else 0
    f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1_score)
    print(f"Time: {(end - start):.2f} seconds")

    if i == 0:
        memory_usage = psutil.Process().memory_info().rss
        print(f"Memory usage: {(memory_usage / (1024 ** 2)):.2f} MB")
        memory_usages.append(memory_usage)
    else:
        memory_usage = psutil.Process().memory_info().rss - memory_usages[i-1]
        print(f"Memory usage: {(memory_usage / (1024 ** 2)):.2f} MB")
        memory_usages.append(memory_usage)
        
    true_positives.append(true_positive)
    false_positives.append(false_positive)
    true_negatives.append(true_negative)
    false_negatives.append(false_negative)
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    f1_scores.append(f1_score)
    time_usages.append(end - start)

    plt.imshow(binary_mask, cmap="gray")
    plt.show()

    plt.imshow(ground_truth_matlike, cmap="gray")
    plt.show()

    plt.imshow(fg_mask, cmap="gray")
    plt.show()

    plt.imshow(ground_truth, cmap="gray")
    plt.show()

cap.release()
cv2.destroyAllWindows()

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.bar(subtractor_names, [p * 100 for p in precisions], label='Precision %', alpha=0.7)
plt.title('Precision')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(subtractor_names, [r * 100 for r in recalls], label='Recall %', alpha=0.7)
plt.title('Recall')
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(subtractor_names, [a * 100 for a in accuracies], label='Accuracy %', alpha=0.7)
plt.title('Accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.bar(subtractor_names, [f1 for f1 in f1_scores], label='F1 Score', alpha=0.7)
plt.title('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.bar(subtractor_names, memory_usages, label='Memory Usage (MB)')
plt.title('Memory Usage')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(subtractor_names, time_usages, label='Time Elapsed (s)')
plt.title('Time')
plt.legend()

plt.tight_layout()
plt.show()

out.release()
cap.release()
cv2.destroyAllWindows()