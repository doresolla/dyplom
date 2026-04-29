import numpy as np
import csv
import os
import cv2
from typing import List, Tuple, Optional

from refine_bbox import (clamp_xyxy,
                         order_quad_pts,
                         bbox_xywh_to_quad,
                         draw_box, xywh_to_xyxy)
from pathlib import Path



QUAD_FIELDS = [
    "file_name",
    "image_path",
    "video_path",
    "video_name",
    "frame_idx",
    "timestamp_sec",
    "image_width",
    "image_height",
    "source_mode",
    "model_x",
    "model_y",
    "model_w",
    "model_h",
    "search_x1",
    "search_y1",
    "search_x2",
    "search_y2",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "quad_x1", "quad_y1",
    "quad_x2", "quad_y2",
    "quad_x3", "quad_y3",
    "quad_x4", "quad_y4",
]


class QuadEditor:
    def __init__(
        self,
        frame_bgr: np.ndarray,
        video_name: str,
        frame_idx: int,
        timestamp_sec: float,
        rough_xywh,
        search_xyxy,
        auto_quad: np.ndarray,
    ):
        self.frame = frame_bgr
        self.video_name = video_name
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec

        self.rough_xywh = [float(v) for v in rough_xywh]
        self.search_xyxy = [int(v) for v in search_xyxy]

        self.auto_quad = order_quad_pts(np.asarray(auto_quad, dtype=np.float32))
        self.bbox_quad = bbox_xywh_to_quad(rough_xywh)

        self.quad = self.auto_quad.copy()
        self.manual_changed = False

        self.window_name = "semi_auto_labeler"
        self.drag_idx = None
        self.drag_radius = 24

        self.h, self.w = self.frame.shape[:2]

    def _clip_point(self, x: int, y: int) -> Tuple[float, float]:
        x = float(np.clip(x, 0, self.w - 1))
        y = float(np.clip(y, 0, self.h - 1))
        return x, y

    def _nearest_point(self, x: int, y: int) -> Optional[int]:
        pts = self.quad
        d = np.sqrt(((pts[:, 0] - x) ** 2) + ((pts[:, 1] - y) ** 2))
        idx = int(np.argmin(d))
        if d[idx] <= self.drag_radius:
            return idx
        return None

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._nearest_point(x, y)
            if idx is not None:
                self.drag_idx = idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_idx is not None:
                nx, ny = self._clip_point(x, y)
                self.quad[self.drag_idx] = [nx, ny]
                self.manual_changed = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = None

    def _render(self) -> np.ndarray:
        vis = self.frame.copy()

        vis = draw_box(vis, xywh_to_xyxy(self.rough_xywh), color=(0, 255, 255), thickness=2, label="model")
        vis = draw_box(vis, self.search_xyxy, color=(255, 255, 0), thickness=2, label="search")
        vis = draw_quad(vis, self.quad, color=(0, 0, 255), thickness=2)

        status = "manual" if self.manual_changed else "auto"
        lines = [
            f"video: {self.video_name}",
            f"frame: {self.frame_idx}   time: {self.timestamp_sec:.2f}s   mode: {status}",
            "LMB: drag nearest point",
            "Enter/Y: save   S: skip   Q/Esc: quit",
            "R: reset auto   B: reset to model bbox",
        ]
        vis = draw_help_text(vis, lines)
        return vis

    def edit(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        while True:
            vis = self._render()
            cv2.imshow(self.window_name, vis)

            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord("y")):
                mode = "manual" if self.manual_changed else "auto"
                cv2.destroyWindow(self.window_name)
                return self.quad.copy(), mode

            if key in (ord("s"), ord("n")):
                cv2.destroyWindow(self.window_name)
                return None, "skip"

            if key in (27, ord("q")):
                cv2.destroyWindow(self.window_name)
                return None, "quit"

            if key == ord("r"):
                self.quad = self.auto_quad.copy()
                self.manual_changed = False

            if key == ord("b"):
                self.quad = self.bbox_quad.copy()
                self.manual_changed = True


def draw_help_text(img: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img.copy()
    y = 25
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return out


def quad_to_xyxy(quad: np.ndarray, width: int, height: int) -> List[int]:
    quad = np.asarray(quad, dtype=np.float32)
    x1 = float(np.min(quad[:, 0]))
    y1 = float(np.min(quad[:, 1]))
    x2 = float(np.max(quad[:, 0]))
    y2 = float(np.max(quad[:, 1]))
    return clamp_xyxy([x1, y1, x2, y2], width, height)

def shift_quad(quad: np.ndarray, dx: float, dy: float) -> np.ndarray:
    q = np.asarray(quad, dtype=np.float32).copy()
    q[:, 0] += dx
    q[:, 1] += dy
    return q

def draw_quad(img, quad, color=(0, 0, 255), thickness=2):
    out = img.copy()
    if quad is None:
        return out

    q = np.round(quad).astype(int)
    cv2.polylines(out, [q.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(q):
        cv2.circle(out, (x, y), 5, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out

def append_quad_csv(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    # ensure_parent()
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUAD_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)