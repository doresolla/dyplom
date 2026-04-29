from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class SlideROIDetectorConfig:
    resize_max_side: int = 1280
    min_area_ratio: float = 0.08       # минимальная площадь ROI относительно кадра
    max_area_ratio: float = 0.98       # максимальная площадь ROI относительно кадра
    aspect_ratios: Tuple[float, ...] = (16 / 9, 16 / 10, 4 / 3)
    aspect_ratio_tolerance: float = 0.35
    canny_thr1: int = 50
    canny_thr2: int = 150
    morph_kernel: int = 7
    binarize_block_size: int = 31
    binarize_C: int = 7
    debug: bool = False


class SlideROIDetector:
    def __init__(self, cfg: SlideROIDetectorConfig | None = None):
        self.cfg = cfg or SlideROIDetectorConfig()

    def detect(self, frame: np.ndarray) -> Optional[BBox]:
        """
        На вход: BGR кадр (H, W, 3)
        На выход: bbox [x1, y1, x2, y2] в координатах исходного кадра
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame passed to detector")

        original_h, original_w = frame.shape[:2]
        scaled, scale = self._resize_keep_aspect(frame, self.cfg.resize_max_side)

        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1) Основной путь: edges -> morphology -> contours
        bbox = self._detect_by_contours(scaled, gray)
        if bbox is None:
            # 2) Fallback: бинаризация яркой/однородной области
            bbox = self._detect_by_threshold(scaled, gray)

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        # Приводим обратно к исходному размеру
        x1 = int(round(x1 / scale))
        y1 = int(round(y1 / scale))
        x2 = int(round(x2 / scale))
        y2 = int(round(y2 / scale))

        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(0, min(x2, original_w - 1))
        y2 = max(0, min(y2, original_h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def _detect_by_contours(self, img: np.ndarray, gray: np.ndarray) -> Optional[BBox]:
        edges = cv2.Canny(gray, self.cfg.canny_thr1, self.cfg.canny_thr2)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.morph_kernel, self.cfg.morph_kernel)
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        frame_area = h * w

        best_score = -1.0
        best_bbox = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_area_ratio * frame_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            if rect_area <= 0:
                continue

            area_ratio = rect_area / frame_area
            if not (self.cfg.min_area_ratio <= area_ratio <= self.cfg.max_area_ratio):
                continue

            contour_fill_ratio = area / rect_area
            aspect = bw / max(bh, 1)

            # Насколько ratio похож на типичный slide ratio
            ratio_error = min(abs(aspect - r) / r for r in self.cfg.aspect_ratios)

            # Слишком вытянутые/странные bbox отбрасываем
            if ratio_error > self.cfg.aspect_ratio_tolerance:
                continue

            # Небольшой бонус, если контур близок к 4-угольнику
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            quad_bonus = 0.15 if len(approx) == 4 else 0.0

            # На слайде обычно есть текст/контент, поэтому variance не должна быть совсем нулевая
            roi_gray = gray[y:y + bh, x:x + bw]
            variance = float(np.var(roi_gray)) / 255.0

            score = (
                2.5 * area_ratio +
                1.5 * contour_fill_ratio +
                1.0 * (1.0 - ratio_error) +
                0.15 * min(variance, 1.0) +
                quad_bonus
            )

            if score > best_score:
                best_score = score
                best_bbox = (x, y, x + bw, y + bh)

        return best_bbox

    def _detect_by_threshold(self, img: np.ndarray, gray: np.ndarray) -> Optional[BBox]:
        # Adaptive threshold помогает, если экран/слайд заметно светлее фона
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.cfg.binarize_block_size,
            self.cfg.binarize_C
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        frame_area = h * w

        best_score = -1.0
        best_bbox = None

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            rect_area = bw * bh
            if rect_area <= 0:
                continue

            area_ratio = rect_area / frame_area
            if not (self.cfg.min_area_ratio <= area_ratio <= self.cfg.max_area_ratio):
                continue

            aspect = bw / max(bh, 1)
            ratio_error = min(abs(aspect - r) / r for r in self.cfg.aspect_ratios)
            if ratio_error > self.cfg.aspect_ratio_tolerance:
                continue

            roi = gray[y:y + bh, x:x + bw]
            mean_val = float(np.mean(roi)) / 255.0

            # Во fallback слегка предпочитаем светлые области
            score = 2.0 * area_ratio + 1.2 * (1.0 - ratio_error) + 0.8 * mean_val

            if score > best_score:
                best_score = score
                best_bbox = (x, y, x + bw, y + bh)

        return best_bbox

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale == 1.0:
            return img.copy(), 1.0
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale


def draw_bbox(frame: np.ndarray, bbox: Optional[BBox]) -> np.ndarray:
    vis = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return vis


def get_aspect_ratio(rect):
  w, h = rect[0], rect[1]  # Ширина и высота
  return max(w/h, h/w)

def detect_canny(
        path="C:\\Users\\dondu\\Downloads\\keyframes\\frames\\seg_0002_f_0002820_00094.00s.jpg",
        out_path = "CANNY_DEBUG.jpg"
    ):
    image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    output_image = cv2.imread(path)
    blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 1.9)

    # 2. Расчет градиентов изображения с помощью Canny
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # 6. Математическая морфология
    # Операция расширения (для увеличения толщины границ)
    kernel = np.ones((5, 5), np.uint8)  # Создание ядра для морфологии
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Исходное изображение
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Параметры для отсечения пропорций экрана
    min_aspect_ratio = 2 / 3
    max_aspect_ratio = 16 / 9

    # Переменная для хранения прямоугольников экрана
    screen_rects = []

    # Проходим по всем контурам
    for contour in contours:
        # Аппроксимация контура до прямоугольника
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # Получаем 4 точки прямоугольника
        box = np.int32(box)
        # Вычисляем соотношение сторон
        aspect_ratio = get_aspect_ratio(rect[1])
        min_area_threshold = 70000
        max_area_threshold = 200000

        # Проверяем, соответствует ли соотношение сторон экрана
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            area = cv2.contourArea(box)
            # Если площадь больше порога, добавляем прямоугольник
            if max_area_threshold > area > min_area_threshold:
                screen_rects.append(box)

    canny_prediction = [0, 0, 0, 0]
    if screen_rects:
        # Функция для вычисления площади прямоугольника
        def rect_area(rect):
            # Используем cv2.contourArea, чтобы вычислить площадь
            return cv2.contourArea(rect)

        # Находим прямоугольник с максимальной площадью
        max_rect = max(screen_rects, key=rect_area)

        # Отображаем его на изображении (для наглядности)
        # cv2.polylines(output_image, [max_rect], isClosed=True, color=(255, 0, 0), thickness=3)
        x, y, w, h = cv2.boundingRect(max_rect)
        canny_prediction = [x, y, w, h]
        print("Canny prediction:", canny_prediction)
    cv2.rectangle(output_image,
                  (canny_prediction[0], canny_prediction[1]),
                  (canny_prediction[0] + canny_prediction[2], canny_prediction[1] + canny_prediction[3]),
                  (0, 0, 255), 2)  # Красный цвет (BGR)
    cv2.imwrite(out_path, output_image)

def detect_slide_ROI(
        image,
        save="roi_debug.jpg",
        max_side: int = 960,
        min_area_ratio: float = 0.12,
        max_area_ratio: float = 0.98,
        smooth_alpha: float = 0.75,
        keep_last_if_low_score: bool = True,
        enable_temporal_smoothing: bool = True,
    ):
    frame = cv2.imread(image)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image}")

    detector = SlideROIDetector()
    bbox = detector.detect(frame)

    print("bbox:", bbox)  # [x1, y1, x2, y2] либо None

    vis = draw_bbox(frame, bbox)
    if save:
        cv2.imwrite(save, vis)
    else:
        cv2.imshow("slide_roi", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

