from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class InstancePred:
    """Single predicted or labeled instance with optional mask and YOLO-seg polygon."""

    class_id: int
    conf: float
    bbox_xyxy: np.ndarray  # shape (4,), float32 pixel coords [x1, y1, x2, y2]
    polygon: np.ndarray | None = None  # shape (N, 2), float32 normalized [0..1]
    source: str = "yolo11"
    second_best_cls: int | None = None
    second_best_conf: float = 0.0
    mask_bbox_iou: float = 1.0
    uncertainty_reasons: list[str] = field(default_factory=list)

    @property
    def margin(self) -> float:
        if self.second_best_cls is None:
            return 1.0
        return max(0.0, float(self.conf) - float(self.second_best_conf))


def box_area_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Compute areas for (N, 4) xyxy boxes."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return w * h


def pairwise_iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute (M, N) IoU matrix between two (M, 4) and (N, 4) xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]  # (M, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]  # (1, N, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return inter / union


def containment_ratio_xyxy(inner_box: np.ndarray, outer_box: np.ndarray) -> float:
    """Return intersection(inner, outer) / area(inner) for two [x1,y1,x2,y2] boxes."""
    ix1 = max(float(inner_box[0]), float(outer_box[0]))
    iy1 = max(float(inner_box[1]), float(outer_box[1]))
    ix2 = min(float(inner_box[2]), float(outer_box[2]))
    iy2 = min(float(inner_box[3]), float(outer_box[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_inner = max(0.0, float(inner_box[2]) - float(inner_box[0])) * max(
        0.0, float(inner_box[3]) - float(inner_box[1])
    )
    if area_inner <= 1e-6:
        return 0.0
    return inter / area_inner


def _bridge_contours_mask(mask_u8: np.ndarray, contours: list[np.ndarray]) -> np.ndarray:
    """Connect multiple disconnected contours along closest vertices so one valid YOLO-seg polygon forms."""
    if len(contours) <= 1:
        return mask_u8
    bridged = mask_u8.copy()
    # Sort descending by area; connect each significant sub-contour to the main contour
    areas = [cv2.contourArea(c) for c in contours]
    order = np.argsort(areas)[::-1]
    main = contours[order[0]].reshape(-1, 2)
    main_area = areas[order[0]]

    for idx in order[1:]:
        if areas[idx] < 0.15 * main_area:
            continue
        other = contours[idx].reshape(-1, 2)
        # Subsample points if contours are dense for fast pairwise distance
        m_sub = main[:: max(1, len(main) // 64)]
        o_sub = other[:: max(1, len(other) // 64)]
        dists = np.sum((m_sub[:, None, :] - o_sub[None, :, :]) ** 2, axis=-1)
        mi, oi = np.unravel_index(np.argmin(dists), dists.shape)
        pt1 = tuple(int(v) for v in m_sub[mi])
        pt2 = tuple(int(v) for v in o_sub[oi])
        cv2.line(bridged, pt1, pt2, color=255, thickness=2)
    return bridged


def mask_to_polygon(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    epsilon_ratio: float = 0.002,
    min_area_px: int = 24,
) -> np.ndarray | None:
    """Convert a 2D binary mask into a normalized (N, 2) YOLO-seg polygon."""
    if mask is None or mask.size == 0:
        return None
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not valid:
        return None

    if len(valid) > 1:
        bridged = _bridge_contours_mask(mask_u8, valid)
        contours2, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid2 = [c for c in contours2 if cv2.contourArea(c) >= min_area_px]
        if valid2:
            valid = valid2

    largest = max(valid, key=cv2.contourArea)
    peri = cv2.arcLength(largest, closed=True)
    eps = max(0.5, float(epsilon_ratio) * peri)
    approx = cv2.approxPolyDP(largest, epsilon=eps, closed=True)
    pts = approx.reshape(-1, 2).astype(np.float32)

    if len(pts) < 3:
        # Retry without aggressive simplification
        pts = largest.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None

    pts[:, 0] = np.clip(pts[:, 0] / float(img_w), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(img_h), 0.0, 1.0)
    return pts


def polygon_to_bbox_xyxy(poly_norm: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Compute tight pixel [x1, y1, x2, y2] from normalized polygon (N, 2)."""
    xs = poly_norm[:, 0] * float(img_w)
    ys = poly_norm[:, 1] * float(img_h)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def polygon_area_px(poly_norm: np.ndarray, img_w: int, img_h: int) -> float:
    """Compute polygon area in pixels."""
    pts = np.column_stack([poly_norm[:, 0] * img_w, poly_norm[:, 1] * img_h]).astype(np.float32)
    return float(cv2.contourArea(pts))


def mask_bbox_iou(mask: np.ndarray, bbox_xyxy: np.ndarray) -> float:
    """Compute IoU between a binary mask and its prompting xyxy bounding box."""
    if mask is None or mask.size == 0:
        return 0.0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    m_box = np.array([[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]], dtype=np.float32)
    p_box = np.asarray(bbox_xyxy, dtype=np.float32)[None, :]
    return float(pairwise_iou_xyxy(m_box, p_box)[0, 0])


def box_xyxy_to_polygon_rect(bbox_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Fallback 4-point rectangle polygon in normalized coords when only a box exists."""
    x1 = float(np.clip(bbox_xyxy[0] / img_w, 0.0, 1.0))
    y1 = float(np.clip(bbox_xyxy[1] / img_h, 0.0, 1.0))
    x2 = float(np.clip(bbox_xyxy[2] / img_w, 0.0, 1.0))
    y2 = float(np.clip(bbox_xyxy[3] / img_h, 0.0, 1.0))
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def read_legacy_bbox_file(txt_path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """Read legacy YOLO detection format (cls xc yc w h) into list of (cls_id, xyxy_pixels)."""
    if not txt_path.exists():
        return []
    out: list[tuple[int, np.ndarray]] = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        x1 = max(0.0, (xc - bw / 2.0) * img_w)
        y1 = max(0.0, (yc - bh / 2.0) * img_h)
        x2 = min(float(img_w), (xc + bw / 2.0) * img_w)
        y2 = min(float(img_h), (yc + bh / 2.0) * img_h)
        if x2 > x1 and y2 > y1:
            out.append((cls_id, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return out


def read_yolo_seg_file(txt_path: Path, img_w: int, img_h: int) -> list[InstancePred]:
    """Read YOLO-seg polygon .txt file into InstancePred objects."""
    if not txt_path.exists():
        return []
    preds: list[InstancePred] = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        cls_id = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(coords[:, 0], 0.0, 1.0)
        coords[:, 1] = np.clip(coords[:, 1], 0.0, 1.0)
        bbox = polygon_to_bbox_xyxy(coords, img_w, img_h)
        preds.append(
            InstancePred(
                class_id=cls_id,
                conf=1.0,
                bbox_xyxy=bbox,
                polygon=coords,
                source="yolo_seg_file",
            )
        )
    return preds


def write_yolo_seg_file(
    txt_path: Path,
    instances: Iterable[InstancePred],
    img_w: int,
    img_h: int,
    allow_rect_fallback: bool = False,
) -> int:
    """Write instances in standard YOLO-seg format: `cls x1 y1 x2 y2 ...`."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for inst in instances:
        poly = inst.polygon
        if poly is None or len(poly) < 3:
            if not allow_rect_fallback:
                continue
            poly = box_xyxy_to_polygon_rect(inst.bbox_xyxy, img_w, img_h)
        flat = " ".join(f"{float(np.clip(x, 0.0, 1.0)):.6f} {float(np.clip(y, 0.0, 1.0)):.6f}" for x, y in poly)
        lines.append(f"{int(inst.class_id)} {flat}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
