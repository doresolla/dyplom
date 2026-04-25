import itertools
import math
from pathlib import Path

from typing import List, Optional

import cv2
import numpy as np


def clamp_xyxy(box, w, h):
    x1, y1, x2, y2 = box

    x1 = int(round(max(0, min(x1, w - 1))))
    y1 = int(round(max(0, min(y1, h - 1))))
    x2 = int(round(max(0, min(x2, w - 1))))
    y2 = int(round(max(0, min(y2, h - 1))))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]

def xyxy_to_xywh(box_xyxy) -> List[int]:
    x1, y1, x2, y2 = box_xyxy
    return [int(round(x1)), int(round(y1)), int(round(max(0, x2 - x1))), int(round(max(0, y2 - y1)))]

def bbox_xywh_to_quad(box_xywh) -> np.ndarray:
    x, y, w, h = [float(v) for v in box_xywh]
    return np.array([
        [x,     y],
        [x + w, y],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32)



def expand_model_roi_xywh(model_roi_xywh, image_shape, pad_ratio=0.10):
    """
    model_roi_xywh: [x, y, w, h] в абсолютных пикселях
    pad_ratio берется от размеров ВСЕГО изображения
    """
    h_img, w_img = image_shape[:2]

    x1, y1, x2, y2 = xywh_to_xyxy(model_roi_xywh)

    pad_x = int(round(pad_ratio * w_img))
    pad_y = int(round(pad_ratio * h_img))

    search_xyxy = [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]
    return clamp_xyxy(search_xyxy, w_img, h_img)


def shift_quad(quad, dx, dy):
    quad = np.asarray(quad, dtype=np.float32).copy()
    quad[:, 0] += dx
    quad[:, 1] += dy
    return quad


def draw_box(img, box_xyxy, color=(0, 255, 255), thickness=2, label: Optional[str] = None):
    out = img.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    if label:
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def resize_max(img, max_side=1200):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img, scale


def color_edge_map(img):
    """
    Карта границ по каналам Lab.
    Это устойчивее, чем опираться только на серое изображение.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)

    edge = np.zeros(img.shape[:2], dtype=np.uint8)

    for ch in channels:
        ch = cv2.GaussianBlur(ch, (5, 5), 0)
        med = np.median(ch)

        low = int(max(10, 0.66 * med))
        high = int(min(255, 1.33 * med))

        e = cv2.Canny(ch, low, high, L2gradient=True)
        edge = cv2.bitwise_or(edge, e)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edge


def detect_long_lines(edge, min_len=80):
    """
    Поиск длинных отрезков детектором LSD.
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = lsd.detect(edge)
    lines = out[0]

    result = []
    if lines is None:
        return result

    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(float, l)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length >= min_len:
            result.append((x1, y1, x2, y2))

    return result


def segment_angle(line):
    x1, y1, x2, y2 = line
    a = math.atan2(y2 - y1, x2 - x1)
    return (a + math.pi) % math.pi


def line_normal_form_from_points(p1, p2):
    """
    Прямая в виде:
        nx * x + ny * y + c = 0
    где (nx, ny) — единичная нормаль.
    """
    x1, y1 = p1
    x2, y2 = p2

    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None

    nx, ny = dy / norm, -dx / norm
    c = -(nx * x1 + ny * y1)

    # Канонический знак
    if c < 0:
        nx, ny, c = -nx, -ny, -c

    return nx, ny, c


def merge_collinear_lines(
        lines,
        angle_thresh_deg=6.0,
        dist_thresh=12.0,
        gap_thresh=80.0,
):
    """
    Объединяет близкие коллинеарные отрезки в одну сторону.
    Это нужно, когда человек или кафедра разрывают видимую границу.
    """
    if not lines:
        return []

    angle_thresh = math.radians(angle_thresh_deg)

    info = []
    for line in lines:
        x1, y1, x2, y2 = line
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)
        info.append(
            {
                "p1": p1,
                "p2": p2,
                "angle": segment_angle(line),
                "line_nf": line_normal_form_from_points((x1, y1), (x2, y2)),
                "length": float(np.hypot(x2 - x1, y2 - y1)),
            }
        )

    used = [False] * len(lines)
    order = sorted(range(len(lines)), key=lambda i: -info[i]["length"])

    merged = []

    for seed in order:
        if used[seed]:
            continue

        cluster = [seed]
        used[seed] = True
        changed = True

        while changed:
            changed = False

            pts = []
            for idx in cluster:
                pts.extend([info[idx]["p1"], info[idx]["p2"]])

            pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            vx, vy, x0, y0 = cv2.fitLine(
                pts_np, cv2.DIST_L2, 0, 0.01, 0.01
            ).flatten()

            direction = np.array([vx, vy], dtype=np.float32)
            direction = direction / np.linalg.norm(direction)
            origin = np.array([x0, y0], dtype=np.float32)

            angle_center = math.atan2(float(direction[1]), float(direction[0])) % math.pi

            nx, ny = float(direction[1]), float(-direction[0])
            norm = math.hypot(nx, ny)
            nx, ny = nx / norm, ny / norm
            c = -(nx * x0 + ny * y0)
            if c < 0:
                nx, ny, c = -nx, -ny, -c

            proj = [float(np.dot(p - origin, direction)) for p in pts]
            min_proj, max_proj = min(proj), max(proj)

            for i in order:
                if used[i]:
                    continue

                angle_i = info[i]["angle"]
                da = abs(angle_i - angle_center)
                da = min(da, math.pi - da)
                if da > angle_thresh:
                    continue

                p1 = info[i]["p1"]
                p2 = info[i]["p2"]

                dist1 = abs(nx * p1[0] + ny * p1[1] + c)
                dist2 = abs(nx * p2[0] + ny * p2[1] + c)
                if max(dist1, dist2) > dist_thresh:
                    continue

                q1 = float(np.dot(p1 - origin, direction))
                q2 = float(np.dot(p2 - origin, direction))
                s1, s2 = min(q1, q2), max(q1, q2)

                interval_gap = max(0.0, max(s1 - max_proj, min_proj - s2))
                if interval_gap > gap_thresh:
                    continue

                cluster.append(i)
                used[i] = True
                changed = True

        # Итоговая прямая по кластеру
        pts = []
        for idx in cluster:
            pts.extend([info[idx]["p1"], info[idx]["p2"]])

        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        vx, vy, x0, y0 = cv2.fitLine(
            pts_np, cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()

        direction = np.array([vx, vy], dtype=np.float32)
        direction = direction / np.linalg.norm(direction)
        origin = np.array([x0, y0], dtype=np.float32)

        proj = [float(np.dot(p - origin, direction)) for p in pts]
        min_proj, max_proj = min(proj), max(proj)

        p_start = origin + min_proj * direction
        p_end = origin + max_proj * direction

        nx, ny = float(direction[1]), float(-direction[0])
        norm = math.hypot(nx, ny)
        nx, ny = nx / norm, ny / norm
        c = -(nx * x0 + ny * y0)
        if c < 0:
            nx, ny, c = -nx, -ny, -c

        merged.append(
            {
                "segment": (
                    float(p_start[0]),
                    float(p_start[1]),
                    float(p_end[0]),
                    float(p_end[1]),
                ),
                "angle": math.atan2(float(direction[1]), float(direction[0])) % math.pi,
                "line_nf": (nx, ny, float(c)),
                "length": float(max_proj - min_proj),
                "members": cluster,
            }
        )

    return merged


def split_into_two_direction_groups(merged_lines):
    """
    Делит линии на два семейства направлений:
    например, верх/низ и лево/право.
    """
    if len(merged_lines) < 4:
        return [], []

    feats = []
    for line in merged_lines:
        a = line["angle"]
        feats.append([math.cos(2 * a), math.sin(2 * a)])

    Z = np.array(feats, dtype=np.float32)

    _, labels, _ = cv2.kmeans(
        Z,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4),
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    fam1 = [merged_lines[i] for i in range(len(merged_lines)) if labels[i, 0] == 0]
    fam2 = [merged_lines[i] for i in range(len(merged_lines)) if labels[i, 0] == 1]
    return fam1, fam2


def line_intersection(nf1, nf2):
    n1x, n1y, c1 = nf1
    n2x, n2y, c2 = nf2

    A = np.array([[n1x, n1y], [n2x, n2y]], dtype=np.float64)
    b = np.array([-c1, -c2], dtype=np.float64)

    det = np.linalg.det(A)
    if abs(det) < 1e-4:
        return None

    x, y = np.linalg.solve(A, b)
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def order_quad_pts(pts):
    """
    Упорядочивает 4 точки по кругу,
    затем начинает с левого верхнего угла.
    """
    pts = np.array(pts, dtype=np.float32)
    center = pts.mean(axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    idx = np.argsort(angles)
    pts = pts[idx]

    s = pts.sum(axis=1)
    i0 = np.argmin(s)
    pts = np.roll(pts, -i0, axis=0)
    return pts


def polygon_area(pts):
    pts = np.array(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex_quad(pts):
    pts = np.array(pts, dtype=np.float64)
    signs = []

    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)

    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def edge_distance_map(edge):
    inv = 255 - edge
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return dist


def sample_side(p1, p2, n=120):
    p1 = np.asarray(p1, np.float32)
    p2 = np.asarray(p2, np.float32)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return p1[None, :] * (1 - t) + p2[None, :] * t


def side_support_score(p1, p2, dist_map, keep_ratio=0.7):
    """
    Оценка стороны по карте расстояний до ближайшей границы.
    Берём только лучшую часть точек, чтобы закрытый участок не ломал оценку.
    """
    h, w = dist_map.shape
    pts = sample_side(p1, p2)

    vals = []
    for x, y in pts:
        xi = int(np.clip(round(float(x)), 0, w - 1))
        yi = int(np.clip(round(float(y)), 0, h - 1))
        vals.append(float(dist_map[yi, xi]))

    vals = np.sort(np.array(vals))
    k = max(1, int(len(vals) * keep_ratio))
    vals = vals[:k]

    return -float(vals.mean())


def quad_outside_penalty(quad, w, h):
    quad = np.array(quad, dtype=np.float32)
    xs = quad[:, 0]
    ys = quad[:, 1]

    dx = np.maximum(0, -xs) + np.maximum(0, xs - (w - 1))
    dy = np.maximum(0, -ys) + np.maximum(0, ys - (h - 1))
    return float(np.mean(dx + dy))


def line_signed_offset(line_nf, center):
    nx, ny, c = line_nf
    x, y = center
    return nx * x + ny * y + c


def clamp_quad_to_image(quad, w, h):
    quad = np.array(quad, dtype=np.float32).copy()
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return quad

def is_point_reasonable(p, w, h, max_abs_scale=3.0):
    """
    Отсекает явно мусорные пересечения:
    например x=-50000 при ширине 2000.
    """
    x, y = p

    if not np.isfinite(x) or not np.isfinite(y):
        return False

    max_abs_x = max_abs_scale * w
    max_abs_y = max_abs_scale * h

    if x < -max_abs_x or x > max_abs_x:
        return False
    if y < -max_abs_y or y > max_abs_y:
        return False

    return True

def build_quad_candidates(fam1, fam2, shape, top_k=10, outside_tol_ratio=0.10):
    """
    Строит кандидаты четырехугольника как пересечения
    двух линий из одного семейства и двух из другого.

    Защита:
    - сразу отбрасываются мусорные пересечения с огромными координатами;
    - кандидаты, слишком далеко выходящие за кадр, отбрасываются;
    - кандидаты с небольшим выходом можно подрезать внутрь кадра.
    """
    h, w = shape
    center = (w / 2.0, h / 2.0)
    outside_tol = outside_tol_ratio * min(w, h)

    fam1 = sorted(fam1, key=lambda d: d["length"], reverse=True)[:top_k]
    fam2 = sorted(fam2, key=lambda d: d["length"], reverse=True)[:top_k]

    for line in fam1:
        line["_off"] = line_signed_offset(line["line_nf"], center)

    for line in fam2:
        line["_off"] = line_signed_offset(line["line_nf"], center)

    quads = []

    for a, b in itertools.combinations(fam1, 2):
        if abs(a["_off"] - b["_off"]) < 0.08 * min(w, h):
            continue

        for c, d in itertools.combinations(fam2, 2):
            if abs(c["_off"] - d["_off"]) < 0.08 * min(w, h):
                continue

            pts = []
            valid = True

            for l1 in (a["line_nf"], b["line_nf"]):
                for l2 in (c["line_nf"], d["line_nf"]):
                    p = line_intersection(l1, l2)

                    if p is None:
                        valid = False
                        break

                    # Вот главный фильтр от x=-50000, y=70000 и т.п.
                    if not is_point_reasonable(p, w, h, max_abs_scale=3.0):
                        valid = False
                        break

                    pts.append(p)

                if not valid:
                    break

            if not valid:
                continue

            quad = order_quad_pts(pts)
            quad = np.array(quad, dtype=np.float32)

            xs = quad[:, 0]
            ys = quad[:, 1]

            # Если кандидат умеренно выходит за кадр — терпим,
            # если слишком далеко — отбрасываем
            if (
                np.any(xs < -outside_tol) or
                np.any(xs > (w - 1 + outside_tol)) or
                np.any(ys < -outside_tol) or
                np.any(ys > (h - 1 + outside_tol))
            ):
                continue

            # Умеренный выход внутрь кадра подрезаем
            quad = clamp_quad_to_image(quad, w, h)
            quad = order_quad_pts(quad)

            if polygon_area(quad) < 0.05 * w * h:
                continue

            quads.append(quad)

    uniq = []
    keys = set()

    for q in quads:
        center_q = q.mean(axis=0)
        area_q = polygon_area(q)
        key = (
            round(center_q[0] / 20),
            round(center_q[1] / 20),
            round(area_q / 5000),
        )
        if key in keys:
            continue
        keys.add(key)
        uniq.append(q)

    return uniq

def score_quad(quad, dist_map, shape):
    """
    Итоговая оценка четырехугольника.
    """
    h, w = shape
    quad = np.array(quad, dtype=np.float32)

    if not is_convex_quad(quad):
        return -1e18

    area = polygon_area(quad)
    if area < 0.05 * w * h:
        return -1e18

    score = 0.0

    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        score += 25.0 * side_support_score(p1, p2, dist_map, keep_ratio=0.7)

    score += 2.0 * math.sqrt(max(area, 1.0))

    lengths = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    ratio = max(lengths) / (min(lengths) + 1e-6)
    if ratio > 12:
        score -= 1000.0

    score -= 8.0 * quad_outside_penalty(quad, w, h)

    return score


def _detect_presentation_surface_full(
        img,
        max_side=1200,
        min_line_len=80,
):
    """
    Старый алгоритм без изменений:
    работает на том изображении, которое ему передали.
    """
    work, scale = resize_max(img, max_side=max_side)

    edge = color_edge_map(work)
    lines = detect_long_lines(edge, min_len=min_line_len)

    merged = merge_collinear_lines(
        lines,
        angle_thresh_deg=6.0,
        dist_thresh=12.0,
        gap_thresh=80.0,
    )

    fam1, fam2 = split_into_two_direction_groups(merged)

    info = {
        "scale": scale,
        "edge": edge,
        "lines": lines,
        "merged": merged,
        "fam1": fam1,
        "fam2": fam2,
    }

    if len(fam1) < 2 or len(fam2) < 2:
        return None, info

    quads = build_quad_candidates(fam1, fam2, work.shape[:2], top_k=10)
    dist_map = edge_distance_map(edge)

    best_quad = None
    best_score = -1e18

    for quad in quads:
        s = score_quad(quad, dist_map, work.shape[:2])
        if s > best_score:
            best_score = s
            best_quad = quad

    info["best_score"] = best_score

    if best_quad is None:
        return None, info

    if scale != 1.0:
        best_quad = best_quad / scale

    return best_quad, info


def detect_presentation_surface(
        img,
        max_side=1200,
        min_line_len=80,
        model_roi_xywh=None,
        roi_pad_ratio=0.20,
        fallback_to_full_image=False,
):
    """
    Если model_roi_xywh=None:
        алгоритм работает по всему изображению.

    Если model_roi_xywh задан:
        сначала вырезается область:
            model_roi_xywh + roi_pad_ratio * размеры всего изображения
        затем исходный алгоритм запускается только внутри нее,
        а найденные точки переносятся обратно в координаты полного кадра.
    """
    if model_roi_xywh is None:
        quad, info = _detect_presentation_surface_full(
            img,
            max_side=max_side,
            min_line_len=min_line_len,
        )
        info["search_xyxy"] = None
        info["crop_origin"] = (0, 0)
        info["used_local_roi"] = False
        return quad, info

    search_xyxy = expand_model_roi_xywh(
        model_roi_xywh=model_roi_xywh,
        image_shape=img.shape,
        pad_ratio=roi_pad_ratio,
    )
    x1, y1, x2, y2 = search_xyxy

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        if fallback_to_full_image:
            quad, info = _detect_presentation_surface_full(
                img,
                max_side=max_side,
                min_line_len=min_line_len,
            )
            info["search_xyxy"] = search_xyxy
            info["crop_origin"] = (0, 0)
            info["used_local_roi"] = False
            info["fallback_to_full_image"] = True
            return quad, info

        return None, {
            "scale": 1.0,
            "edge": None,
            "lines": [],
            "merged": [],
            "fam1": [],
            "fam2": [],
            "search_xyxy": search_xyxy,
            "crop_origin": (x1, y1),
            "used_local_roi": True,
            "fallback_to_full_image": False,
        }

    quad_local, info = _detect_presentation_surface_full(
        crop,
        max_side=max_side,
        min_line_len=min_line_len,
    )

    info["search_xyxy"] = search_xyxy
    info["crop_origin"] = (x1, y1)
    info["used_local_roi"] = True
    info["fallback_to_full_image"] = False

    if quad_local is not None:
        quad_full = shift_quad(quad_local, x1, y1)
        return quad_full, info

    if fallback_to_full_image:
        quad, info_full = _detect_presentation_surface_full(
            img,
            max_side=max_side,
            min_line_len=min_line_len,
        )
        info_full["search_xyxy"] = search_xyxy
        info_full["crop_origin"] = (0, 0)
        info_full["used_local_roi"] = False
        info_full["fallback_to_full_image"] = True
        return quad, info_full

    return None, info


def save_debug_images(img, info, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if info.get("search_xyxy") is not None:
        vis_search = img.copy()
        x1, y1, x2, y2 = info["search_xyxy"]
        cv2.rectangle(vis_search, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "00_search_roi.png"), vis_search)

    if info["edge"] is None:
        return

    cv2.imwrite(str(out_dir / "01_edges.png"), info["edge"])

    vis_lines = cv2.cvtColor(info["edge"], cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in info["lines"]:
        cv2.line(
            vis_lines,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "02_raw_lines.png"), vis_lines)

    scale = info["scale"]

    if info.get("used_local_roi", False) and info.get("search_xyxy") is not None:
        sx1, sy1, sx2, sy2 = info["search_xyxy"]
        src = img[sy1:sy2, sx1:sx2]
    else:
        src = img

    work, _ = resize_max(src, max_side=int(round(max(src.shape[:2]) * scale)))
    vis_merged = work.copy()

    for item in info["merged"]:
        x1, y1, x2, y2 = item["segment"]
        cv2.line(
            vis_merged,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "03_merged_lines.png"), vis_merged)
