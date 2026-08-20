import os
from functools import lru_cache

import cv2
import numpy as np


def get_aspect_ratio(size):
    """
    size = (w, h) из rect[1]
    Возвращает отношение меньшей стороны к большей,
    чтобы результат всегда был в диапазоне (0, 1].
    Для экрана 16:9 получится 9/16 = 0.5625,
    для 4:3 получится 0.75
    """
    w, h = size
    if w <= 0 or h <= 0:
        return 0.0
    return min(w, h) / max(w, h)

# def iou_metric(output_true, output_pred):
#
#     output_true = tf.cast(output_true, tf.float32)
#     output_pred = tf.cast(output_pred, tf.float32)
#
#     x_true, y_true, w_true, h_true = tf.split(output_true, 4, axis=-1)
#     x_pred, y_pred, w_pred, h_pred = tf.split(output_pred, 4, axis=-1)
#
#     x_left = tf.maximum(x_true, x_pred)
#     y_top = tf.maximum(y_true, y_pred)
#     x_right = tf.minimum(x_true + w_true, x_pred + w_pred)
#     y_bottom = tf.minimum(y_true + h_true, y_pred + h_pred)
#
#     intersection = tf.maximum(0.0, x_right - x_left) * tf.maximum(0.0, y_bottom - y_top)
#     union = (w_true * h_true) + (w_pred * h_pred) - intersection
#
#     return tf.reduce_mean(intersection / (union + tf.keras.backend.epsilon()))
#



def find_page_rect(img: np.ndarray, debug: bool = True):
    if img is None:
        raise ValueError(f"Не удалось открыть изображение")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Сглаживание и границы
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    # 2. Маска "пустого фона":
    # светлые пиксели, которые не попали в границы
    # Для такого типа страниц обычно 240 работает неплохо.
    bright_thr = 240
    free_mask = ((gray >= bright_thr) & (dilated_edges == 0)).astype(np.uint8) * 255

    # 3. Убираем мелкие детали и разрывы
    free_mask_open = cv2.morphologyEx(
        free_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    )

    # 4. Ищем крупнейшую связную компоненту,
    # но отбрасываем слишком верхние области панели
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(free_mask_open, 8)

    best = None
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]

        # отсечение мелочи
        if area < 0.10 * h * w:
            continue

        # верхнюю ленту интерфейса стараемся не брать
        if y < 0.05 * h:
            continue

        if best is None or area > best[0]:
            best = (area, x, y, ww, hh, i)

    if best is None:
        print("Не удалось найти область страницы")
        return img, img

    _, x0, y0, w0, h0, label_id = best
    x1 = x0
    y1 = y0
    x2 = x0 + w0 - 1
    y2 = y0 + h0 - 1

    # 5. Расширяем внутреннюю область до границ страницы
    # Берем профили яркости:
    # по столбцам - в диапазоне найденной области по Y,
    # по строкам - в диапазоне найденной области по X.
    col_mean = gray[y1:y2 + 1, :].mean(axis=0).astype(np.float32)
    row_mean = gray[:, x1:x2 + 1].mean(axis=1).astype(np.float32)

    col_mean_smooth = cv2.GaussianBlur(col_mean.reshape(1, -1), (31, 1), 0).ravel()
    row_mean_smooth = cv2.GaussianBlur(row_mean.reshape(-1, 1), (1, 31), 0).ravel()

    def expand_lr(profile, left, right, thr):
        while left > 0 and profile[left - 1] >= thr:
            left -= 1
        while right < len(profile) - 1 and profile[right + 1] >= thr:
            right += 1
        return left, right

    def expand_tb(profile, top, bottom, thr):
        while top > 0 and profile[top - 1] >= thr:
            top -= 1
        while bottom < len(profile) - 1 and profile[bottom + 1] >= thr:
            bottom += 1
        return top, bottom

    # Влево/вправо страница почти белая -> порог повыше
    x1, x2 = expand_lr(col_mean_smooth, x1, x2, thr=245)

    # Вверх у тебя есть заголовок и рисунок, поэтому порог помягче
    y1, y2 = expand_tb(row_mean_smooth, y1, y2, thr=220)

    # Снизу лучше чуть поджать, чтобы не захватить строку состояния
    while y2 > y1 and row_mean_smooth[y2] < 248:
        y2 -= 1

    # Дополнительная страховка от выхода за границы
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2))
    y2 = min(h - 1, int(y2))

    # 6. Debug-визуализация
    debug_img = img.copy()

    # найденная внутренняя связная компонента
    component_mask = np.zeros_like(gray, dtype=np.uint8)
    component_mask[labels == label_id] = 255

    # финальный прямоугольник
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # прямоугольник внутренней найденной области
    cv2.rectangle(debug_img, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)

    cv2.putText(
        debug_img,
        f"final: ({x1}, {y1}) - ({x2}, {y2})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    page_crop = img[y1:y2 + 1, x1:x2 + 1]

    if debug:
        cv2.imshow("01_gray", gray)
        cv2.imshow("02_edges", edges)
        cv2.imshow("03_dilated_edges", dilated_edges)
        cv2.imshow("04_free_mask", free_mask)
        cv2.imshow("05_free_mask_open", free_mask_open)
        cv2.imshow("06_largest_component", component_mask)
        cv2.imshow("07_final_rect", debug_img)
        cv2.imshow("08_page_crop", page_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (x1, y1, x2, y2), page_crop




def _largest_true_run(mask_1d: np.ndarray):
    """
    Ищет самый длинный непрерывный участок True в 1D-массиве.
    Возвращает (start, end), если не найдено -> (-1, -1)
    """
    best_s, best_e, best_len = -1, -1, 0
    s = -1

    for i, v in enumerate(mask_1d.astype(bool)):
        if v and s < 0:
            s = i
        elif (not v) and s >= 0:
            ln = i - s
            if ln > best_len:
                best_s, best_e, best_len = s, i - 1, ln
            s = -1

    if s >= 0:
        ln = len(mask_1d) - s
        if ln > best_len:
            best_s, best_e, best_len = s, len(mask_1d) - 1, ln

    return best_s, best_e


def _smooth_1d(arr, k=31, axis="row"):
    """
    Гауссово сглаживание 1D профиля.
    """
    arr = arr.astype(np.float32).ravel()
    k = max(3, int(k) | 1)

    if axis == "row":
        return cv2.GaussianBlur(arr.reshape(-1, 1), (1, k), 0).ravel()
    else:
        return cv2.GaussianBlur(arr.reshape(1, -1), (k, 1), 0).ravel()


def _rect_area(rect):
    x1, y1, x2, y2 = rect
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def _clip_rect(rect, w, h):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))
    return x1, y1, x2, y2


def _score_rect(frame, rect):
    """
    Общая оценка кандидата.
    Чем больше и "логичнее" прямоугольник относительно кадра, тем выше score.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = rect

    if x2 <= x1 or y2 <= y1:
        return -1e9

    area_ratio = _rect_area(rect) / float(h * w)
    aspect = (x2 - x1 + 1) / max(1.0, (y2 - y1 + 1))

    # слабое предпочтение слайдоподобным пропорциям
    aspect_penalty = min(
        abs(aspect - 16 / 9),
        abs(aspect - 4 / 3),
        abs(aspect - 2.0),
    )

    score = 2.5 * area_ratio - 0.25 * aspect_penalty

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inside_mean = gray[y1:y2 + 1, x1:x2 + 1].mean()

    pad = max(4, min(h, w) // 80)
    xx1 = max(0, x1 - pad)
    yy1 = max(0, y1 - pad)
    xx2 = min(w - 1, x2 + pad)
    yy2 = min(h - 1, y2 + pad)

    ring = gray[yy1:yy2 + 1, xx1:xx2 + 1].copy()
    ring[y1 - yy1:y2 - yy1 + 1, x1 - xx1:x2 - xx1 + 1] = 0

    outer = ring[ring > 0]
    if outer.size:
        outer_mean = outer.mean()
        score += 0.7 * abs(float(inside_mean) - float(outer_mean)) / 255.0

    return float(score)


# ---------------------------------------------------------
# 1. Детектор светлого холста / страницы внутри приложения
# ---------------------------------------------------------

def _mask_bright_canvas(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    bright_thr = int(np.clip(np.percentile(blur, 78) - 8, 165, 245))

    mask = ((blur >= bright_thr) & (edges == 0)).astype(np.uint8) * 255

    k1 = max(9, min(gray.shape) // 35)
    k2 = max(5, min(gray.shape) // 150)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (k1 | 1, k1 | 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2 | 1, k2 | 1))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    return mask


def _candidate_light_canvas(frame):
    """
    Ищет светлый холст / страницу / рабочую область.
    ВАЖНО: здесь ROI берется по крупнейшей связной компоненте маски,
    а не по проекциям и не по расширению вверх.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = _mask_bright_canvas(gray)

    # connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )

    best_rect = None
    best_score = -1e9

    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]

        if ww <= 0 or hh <= 0:
            continue

        area_ratio = area / float(h * w)
        bbox_area = ww * hh
        fill_ratio = area / float(max(1, bbox_area))
        aspect = ww / float(max(1, hh))

        # Отсекаем мусор
        if area_ratio < 0.08:
            continue

        # очень тонкие полосы не нужны
        if ww < 0.20 * w and hh < 0.20 * h:
            continue

        # слишком маленькая заполненность bbox -> обычно мусор/контуры
        if fill_ratio < 0.55:
            continue

        # штраф за касание границ кадра
        touches_border = 0
        if x <= 1:
            touches_border += 1
        if y <= 1:
            touches_border += 1
        if x + ww >= w - 1:
            touches_border += 1
        if y + hh >= h - 1:
            touches_border += 1

        # слабое предпочтение "слайдоподобным" формам
        aspect_penalty = min(
            abs(aspect - 16 / 9),
            abs(aspect - 4 / 3),
            abs(aspect - 2.0),
            abs(aspect - 2.5),
        )

        score = (
            2.5 * area_ratio +
            1.5 * fill_ratio -
            0.25 * aspect_penalty -
            0.35 * touches_border
        )

        if score > best_score:
            best_score = score
            best_rect = (x, y, x + ww - 1, y + hh - 1)

    if best_rect is None:
        return None, {"mode": "light", "mask": mask}

    # Небольшой аккуратный padding
    x1, y1, x2, y2 = best_rect
    pad = 2
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    rect = (x1, y1, x2, y2)

    info = {
        "mode": "light",
        "score": float(best_score),
        "mask": mask,
        "component_rect": rect,
    }
    return rect, info


# ---------------------------------------------------------
# 2. Детектор темного слайда / слайда на черном фоне
# ---------------------------------------------------------

def _mask_dark_foreground(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    dark_thr = int(np.clip(max(8, np.percentile(blur, 10) + 5), 8, 32))
    mask = (blur > dark_thr).astype(np.uint8) * 255

    k1 = max(9, min(gray.shape) // 45)
    k2 = max(5, min(gray.shape) // 160)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (k1 | 1, k1 | 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (k2 | 1, k2 | 1))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    return mask


def _candidate_dark_or_black_bg(frame):
    """
    Для темных презентаций и случаев с черной полосой справа / вебкой.
    Ключевая идея: игнорировать верхнюю часть кадра при поиске X,
    чтобы вебка не ломала правую границу слайда.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = _mask_dark_foreground(gray)
    m = (mask > 0).astype(np.float32)

    top_ignore = int(0.10 * h)
    bottom_ignore = int(0.04 * h)

    band = m[top_ignore:max(top_ignore + 1, h - bottom_ignore), :]
    col_ratio = band.mean(axis=0)

    x1, x2 = _largest_true_run(col_ratio >= 0.50)
    if x1 < 0:
        return None, {"mode": "dark"}

    while x1 > 0 and col_ratio[x1 - 1] >= 0.08:
        x1 -= 1
    while x2 < w - 1 and col_ratio[x2 + 1] >= 0.08:
        x2 += 1

    row_ratio = m[:, x1:x2 + 1].mean(axis=1)

    # если почти весь кадр и так слайд, берем всю высоту
    if row_ratio.mean() > 0.85:
        y1, y2 = 0, h - 1
    else:
        y1, y2 = _largest_true_run(row_ratio >= 0.30)
        if y1 < 0:
            y1, y2 = 0, h - 1

        while y1 > 0 and row_ratio[y1 - 1] >= 0.08:
            y1 -= 1
        while y2 < h - 1 and row_ratio[y2 + 1] >= 0.08:
            y2 += 1

    rect = _clip_rect((x1, y1, x2, y2), w, h)

    info = {
        "mode": "dark",
    }
    return rect, info


# ---------------------------------------------------------
# 3. Запасной вариант: отличие от цвета рамок по краям
# ---------------------------------------------------------

def _candidate_border_difference(frame):
    h, w = frame.shape[:2]

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)

    b = max(4, min(h, w) // 25)

    strips = [
        lab[:b, :, :].reshape(-1, 3),
        lab[h - b:, :, :].reshape(-1, 3),
        lab[:, :b, :].reshape(-1, 3),
        lab[:, w - b:, :].reshape(-1, 3),
    ]

    meds = np.array([np.median(s, axis=0) for s in strips], dtype=np.float32)

    d = np.min(
        np.sqrt(((lab[:, :, None, :].astype(np.float32) - meds[None, None, :, :]) ** 2).sum(axis=3)),
        axis=2
    )

    border_mask = np.zeros((h, w), np.uint8)
    border_mask[:b] = 1
    border_mask[h - b:] = 1
    border_mask[:, :b] = 1
    border_mask[:, w - b:] = 1

    thr = max(10.0, float(np.percentile(d[border_mask > 0], 95)) * 1.8)

    mask = (d > thr).astype(np.uint8) * 255

    k = max(7, min(h, w) // 60)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k | 1, k | 1), np.uint8))

    m = (mask > 0).astype(np.float32)

    col_ratio = m.mean(axis=0)
    x1, x2 = _largest_true_run(col_ratio >= 0.40)
    if x1 < 0:
        return None, {"mode": "border"}

    row_ratio = m[:, x1:x2 + 1].mean(axis=1)
    y1, y2 = _largest_true_run(row_ratio >= 0.25)
    if y1 < 0:
        return None, {"mode": "border"}

    rect = _clip_rect((x1, y1, x2, y2), w, h)

    info = {
        "mode": "border",
    }
    return rect, info


# ---------------------------------------------------------
# Главная функция
# ---------------------------------------------------------

def detect_slide_roi(frame, debug=False):
    """
    Универсальный поиск ROI слайда на одном кадре.

    Возвращает:
        rect = (x1, y1, x2, y2)
        info = словарь с диагностикой
    """
    h, w = frame.shape[:2]

    candidate_map = {}

    for fn in (
        _candidate_light_canvas,
        _candidate_dark_or_black_bg,
        _candidate_border_difference,
    ):
        rect, info = fn(frame)
        if rect is None:
            continue
        score = _score_rect(frame, rect)
        candidate_map[info["mode"]] = (score, rect, info)

    if not candidate_map:
        rect = (0, 0, w - 1, h - 1)
        return rect, {"mode": "fallback_full_frame"}

    light = candidate_map.get("light")
    dark = candidate_map.get("dark")

    # Спец-правило для "светлой страницы внутри окна приложения"
    if light is not None and dark is not None:
        _, lrect, _ = light
        _, drect, _ = dark

        l_area = _rect_area(lrect) / float(h * w)
        d_area = _rect_area(drect) / float(h * w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        l_mean = gray[lrect[1]:lrect[3] + 1, lrect[0]:lrect[2] + 1].mean()

        dark_is_almost_full = d_area > 0.95
        light_has_inner_margins = (lrect[0] > 0.02 * w) or (lrect[1] > 0.02 * h)

        if dark_is_almost_full and light_has_inner_margins and l_area > 0.45 and l_mean > 150:
            best_rect = lrect
            best_info = {
                "mode": "light",
                "score": float(light[0] + 0.3),
                "candidates": [
                    (float(s), r, m) for m, (s, r, _) in candidate_map.items()
                ]
            }

            if debug:
                dbg = frame.copy()
                cv2.rectangle(dbg, (best_rect[0], best_rect[1]), (best_rect[2], best_rect[3]), (0, 0, 255), 2)
                cv2.imshow("detect_slide_roi", dbg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return best_rect, best_info

    candidates = []

    for mode, (score, rect, info) in candidate_map.items():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mode == "light":
            inside = gray[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1]
            score += 0.2 * (inside.mean() / 255.0)

        elif mode == "dark":
            x1, y1, x2, y2 = rect
            if x2 < w - 5:
                right = gray[:, x2 + 1:]
                if right.size:
                    score += 0.4 * max(
                        0.0,
                        (gray[y1:y2 + 1, x1:x2 + 1].mean() - right.mean()) / 255.0
                    )

        candidates.append((score, rect, mode))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_rect, best_mode = candidates[0]

    info = {
        "mode": best_mode,
        "score": float(best_score),
        "candidates": [(float(s), r, m) for s, r, m in candidates]
    }

    if debug:
        dbg = frame.copy()
        cv2.rectangle(dbg, (best_rect[0], best_rect[1]), (best_rect[2], best_rect[3]), (0, 0, 255), 2)
        cv2.putText(
            dbg,
            f"mode={best_mode} score={best_score:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        cv2.imshow("detect_slide_roi", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return best_rect, info


# ---------------------------------------------------------
# Обертка для видео: один ROI на все видео
# ---------------------------------------------------------

def detect_slide_roi_from_video(
    video_path: str,
    sample_fps: float = 0.5,
    max_frames: int = 20,
    debug: bool = False,
):
    """
    Берет несколько кадров из видео, ищет ROI на каждом кадре,
    затем возвращает медианный ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / sample_fps)))

    rects = []
    infos = []

    frame_idx = 0
    used = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step == 0:
            try:
                rect, info = detect_slide_roi(frame, debug=False)
                rects.append(rect)
                infos.append(info)
                used += 1

                if debug:
                    dbg = frame.copy()
                    cv2.rectangle(dbg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                    # cv2.putText(
                    #     dbg,
                    #     f"{used}: {info['mode']}",
                    #     (20, 30),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.8,
                    #     (0, 0, 255),
                    #     2
                    # )
                    cv2.imshow("video_debug", dbg)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

            except Exception:
                pass

            if used >= max_frames:
                break

        frame_idx += 1

    cap.release()
    if debug:
        cv2.destroyAllWindows()

    if not rects:
        raise RuntimeError("ROI was not detected on sampled frames")

    rects_np = np.array(rects, dtype=np.int32)

    x1 = int(np.median(rects_np[:, 0]))
    y1 = int(np.median(rects_np[:, 1]))
    x2 = int(np.median(rects_np[:, 2]))
    y2 = int(np.median(rects_np[:, 3]))

    final_rect = (x1, y1, x2, y2)
    return final_rect, rects, infos



def detect_slide_keypoints(img: np.ndarray, debug: bool = True):
    rect, crop = find_page_rect(img, debug)

