from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, Taxonomy
from .geometry import InstancePred, box_area_xyxy, pairwise_iou_xyxy, read_legacy_bbox_file


@dataclass
class ImageALResult:
    """Active Learning scoring & routing decision for a single image."""

    stem: str
    image_path: Path
    bucket: str  # "auto" (easy, >0.85 & clean) or "human" (uncertain -> CVAT)
    al_score: float
    min_conf: float
    mean_conf: float
    num_objects: int
    reasons: list[str] = field(default_factory=list)
    instances: list[InstancePred] = field(default_factory=list)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "image": self.image_path.name,
            "bucket": self.bucket,
            "al_score": round(float(self.al_score), 4),
            "min_conf": round(float(self.min_conf), 4),
            "mean_conf": round(float(self.mean_conf), 4),
            "num_objects": self.num_objects,
            "reasons": self.reasons,
        }


def annotate_pair_confusion(
    raw_instances: list[InstancePred],
    nms_iou: float,
    pair_confusion_iou: float,
    pair_confusion_margin: float,
    taxonomy: Taxonomy,
) -> list[InstancePred]:
    """Run class-agnostic NMS on raw multi-class predictions while capturing top-2 class confusion."""
    if not raw_instances:
        return []

    order = sorted(range(len(raw_instances)), key=lambda i: raw_instances[i].conf, reverse=True)
    boxes = np.stack([raw_instances[i].bbox_xyxy for i in order], axis=0)
    ious = pairwise_iou_xyxy(boxes, boxes)

    suppressed = [False] * len(order)
    kept: list[InstancePred] = []

    for pos_i, orig_i in enumerate(order):
        if suppressed[pos_i]:
            continue
        winner = raw_instances[orig_i]
        best_alt_cls: int | None = None
        best_alt_conf = 0.0

        for pos_j in range(pos_i + 1, len(order)):
            if suppressed[pos_j]:
                continue
            iou = float(ious[pos_i, pos_j])
            other = raw_instances[order[pos_j]]
            if iou >= pair_confusion_iou and other.class_id != winner.class_id:
                if other.conf > best_alt_conf:
                    best_alt_conf = float(other.conf)
                    best_alt_cls = int(other.class_id)
            if iou >= nms_iou:
                suppressed[pos_j] = True

        if best_alt_cls is not None:
            winner.second_best_cls = best_alt_cls
            winner.second_best_conf = best_alt_conf
            margin = winner.conf - best_alt_conf
            if margin < pair_confusion_margin or taxonomy.is_confusion_pair(winner.class_id, best_alt_cls):
                if margin < pair_confusion_margin:
                    c1 = taxonomy.classes[winner.class_id].name
                    c2 = taxonomy.classes[best_alt_cls].name
                    winner.uncertainty_reasons.append(f"pair_confusion:{c1}_vs_{c2}(margin={margin:.2f})")

        kept.append(winner)

    return kept


def evaluate_image_active_learning(
    stem: str,
    image_path: Path,
    img_w: int,
    img_h: int,
    instances: list[InstancePred],
    cfg: PipelineConfig,
    taxonomy: Taxonomy,
    legacy_txt_path: Path | None = None,
) -> ImageALResult:
    """Score uncertainty for one image and route to 'auto' (easy) vs 'human' (CVAT)."""
    al_cfg = cfg.active_learning
    auto_thresh = float(al_cfg.get("auto_label_conf_thresh", 0.85))
    uncertain_thresh = float(al_cfg.get("uncertain_conf_thresh", 0.60))
    min_mask_iou = float(al_cfg.get("min_mask_bbox_iou", 0.40))
    large_truck_area = float(al_cfg.get("large_truck_area_thresh", 0.14))
    legacy_match_iou = float(al_cfg.get("legacy_match_iou", 0.30))

    reasons: list[str] = []
    al_score = 0.0

    # Check legacy 4-class priors to detect missed objects / truck-trailer ambiguities
    legacy_boxes = read_legacy_bbox_file(legacy_txt_path, img_w, img_h) if legacy_txt_path else []
    img_area = float(max(1, img_w * img_h))

    if not instances:
        if legacy_boxes:
            reasons.append(f"missed_legacy_objects({len(legacy_boxes)})")
            al_score += 0.95
        else:
            reasons.append("no_detections")
            al_score += 0.35
        return ImageALResult(
            stem=stem,
            image_path=image_path,
            bucket="human",
            al_score=al_score,
            min_conf=0.0,
            mean_conf=0.0,
            num_objects=0,
            reasons=reasons,
            instances=[],
        )

    confs = [float(inst.conf) for inst in instances]
    min_conf = min(confs)
    mean_conf = float(np.mean(confs))

    low_count = 0
    mid_count = 0
    pair_count = 0
    mask_issue_count = 0
    domain_flags: set[str] = set()

    for inst in instances:
        # Collect object-level uncertainty reasons produced during fusion/NMS
        for r in inst.uncertainty_reasons:
            domain_flags.add(r)
            if r.startswith("pair_confusion:"):
                pair_count += 1

        if inst.conf < uncertain_thresh:
            low_count += 1
        elif inst.conf < auto_thresh:
            mid_count += 1

        if inst.polygon is None or len(inst.polygon) < 3:
            mask_issue_count += 1
        elif inst.mask_bbox_iou < min_mask_iou:
            mask_issue_count += 1

        # Domain rule check: large Truck (2) box might still include attached trailer (Other vehicle = 3)
        if inst.class_id == 2:
            area_frac = float(box_area_xyxy(inst.bbox_xyxy[None, :])[0]) / img_area
            if area_frac >= large_truck_area:
                domain_flags.add("truck_trailer_split_check")

    if low_count > 0:
        reasons.append(f"low_conf(<{uncertain_thresh:.2f}):{low_count}")
        al_score += 0.65 + 0.15 * min(low_count, 4) + (uncertain_thresh - min_conf)

    if mid_count > 0:
        reasons.append(f"mid_conf({uncertain_thresh:.2f}-{auto_thresh:.2f}):{mid_count}")
        al_score += 0.25 + 0.08 * min(mid_count, 5) + max(0.0, auto_thresh - min_conf) * 0.5

    if pair_count > 0:
        al_score += 0.35 + 0.10 * min(pair_count, 3)

    if mask_issue_count > 0:
        reasons.append(f"mask_quality_low:{mask_issue_count}")
        al_score += 0.25 + 0.08 * min(mask_issue_count, 4)

    # Check if any legacy box had no overlap with predicted 11-class instances
    if legacy_boxes:
        pred_arr = np.stack([inst.bbox_xyxy for inst in instances], axis=0)
        leg_arr = np.stack([b for _, b in legacy_boxes], axis=0)
        ious = pairwise_iou_xyxy(leg_arr, pred_arr)
        unmatched = int(np.sum(ious.max(axis=1) < legacy_match_iou))
        if unmatched > 0:
            reasons.append(f"legacy_unmatched:{unmatched}")
            al_score += 0.20 + 0.05 * min(unmatched, 4)

    for df in sorted(domain_flags):
        if df not in reasons:
            reasons.append(df)
            if not df.startswith("pair_confusion:"):
                al_score += 0.20

    # Strict Easy Auto-Label Gate:
    # Only auto-accept if ALL objects > 0.85, zero domain/pair/legacy/mask flags.
    is_easy = (
        low_count == 0
        and mid_count == 0
        and pair_count == 0
        and mask_issue_count == 0
        and len(reasons) == 0
        and min_conf >= auto_thresh
    )
    bucket = "auto" if is_easy else "human"
    return ImageALResult(
        stem=stem,
        image_path=image_path,
        bucket=bucket,
        al_score=al_score,
        min_conf=min_conf,
        mean_conf=mean_conf,
        num_objects=len(instances),
        reasons=reasons,
        instances=instances,
    )
