import cv2
from slide_locator import (
    analyze_video_complexity,
    detect_slide_keypoints,
    extract_video_keypoints,
    draw_detection,
    YoloSlideDetector,
)

# 1. Определить, simple / hybrid / neural
report = analyze_video_complexity("C:\\Users\\dondu\\PycharmProjects\\VKR\\math\\math.mp4")
print(report.recommended_mode)
print(report.success_rate, report.median_confidence, report.temporal_jitter)

# 2. Найти точки на одном кадре
frame = cv2.imread(
    "C:\\Users\\dondu\\PycharmProjects\\automatic_conspect33\\runs\\math\\keyframes\\key_0000_356.00s.jpg")
det = detect_slide_keypoints(frame, mode="classic")
print(det.keypoints, det.confidence)

vis = draw_detection(frame, det)
cv2.imwrite("frame_detected.jpg", vis)

# 3. Работа в auto-режиме на видео
# если есть обученная YOLO-модель:
# neural = YoloSlideDetector("best.pt")
# report, results = extract_video_keypoints("lecture.mp4", mode="auto", neural_detector=neural)

# если нейросети пока нет, можно хотя бы получить оценку пригодности classical:
report, results = extract_video_keypoints("C:\\Users\\dondu\\PycharmProjects\\VKR\\math\\math.mp4", mode="auto", neural_detector=None)

for t, det in results[:5]:
    print(f"{t:.2f}s", det.mode_used, det.confidence, det.keypoints)