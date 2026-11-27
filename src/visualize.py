from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .active_learning import ImageALResult
from .config import Taxonomy
from .geometry import InstancePred


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_instances_overlay(
    image_path: Path,
    instances: list[InstancePred],
    taxonomy: Taxonomy,
    out_path: Path,
    al_result: ImageALResult | None = None,
) -> Path:
    """Render semi-transparent YOLO-seg polygons, boxes, class labels, and AL status badge."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    overlay = img.copy()

    for inst in instances:
        cdef = taxonomy.classes.get(inst.class_id)
        color = _hex_to_bgr(cdef.color) if cdef else (200, 200, 200)
        label_name = cdef.name if cdef else f"cls_{inst.class_id}"

        if inst.polygon is not None and len(inst.polygon) >= 3:
            pts = np.column_stack([inst.polygon[:, 0] * w, inst.polygon[:, 1] * h]).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        x1, y1, x2, y2 = [int(round(v)) for v in inst.bbox_xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        tag = f"{inst.class_id}:{label_name} {inst.conf:.2f}"
        if inst.second_best_cls is not None and inst.margin < 0.20:
            alt = taxonomy.classes[inst.second_best_cls].name
            tag += f" (?{alt}:{inst.second_best_conf:.2f})"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(img, (x1, ty - th - 4), (min(w, x1 + tw + 6), ty + 2), color, -1)
        cv2.putText(img, tag, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    if al_result is not None:
        is_auto = al_result.bucket == "auto"
        badge_color = (46, 160, 67) if is_auto else (38, 110, 235)
        header = (
            f"[{al_result.bucket.upper()}] objs={al_result.num_objects} "
            f"min_conf={al_result.min_conf:.2f} al_score={al_result.al_score:.2f}"
        )
        if al_result.reasons:
            header += " | " + ", ".join(al_result.reasons[:3])
        cv2.rectangle(blended, (0, 0), (w, 28), (20, 20, 20), -1)
        cv2.rectangle(blended, (0, 0), (8, 28), badge_color, -1)
        cv2.putText(blended, header[:120], (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)
    return out_path
