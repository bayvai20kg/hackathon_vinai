from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .active_learning import annotate_pair_confusion
from .config import PipelineConfig, Taxonomy
from .geometry import InstancePred, containment_ratio_xyxy, pairwise_iou_xyxy, read_legacy_bbox_file
from .segmenter import Segmenter

# Virtual internal class IDs used prior to rider-vehicle spatial merging
_VCLS_MOTORCYCLE = 104
_VCLS_BICYCLE = 106
_VCLS_PERSON = 108

# Prompt-free YOLOE specialist classes (fills COCO gaps: Sign=9, Other vehicle=3, plus Traffic light=10)
_PF_SPECIALIST_TO_VCLS: dict[str, int] = {
    # 3 = Other vehicle
    "trailer": 3,
    "shipping container": 3,
    "tricycle": 3,
    "rickshaw": 3,
    "excavator": 3,
    "forklift": 3,
    "tractor": 3,
    "crane": 3,
    # 9 = Sign (road/traffic signs only, avoid shop banners)
    "traffic sign": 9,
    "road sign": 9,
    "street sign": 9,
    "stop sign": 9,
    "speed limit sign": 9,
    "warning sign": 9,
    "parking sign": 9,
    # 10 = Traffic light
    "traffic light": 10,
}

# Standard COCO 80-class mapping -> target/virtual class
_COCO_CLASS_MAP: dict[int, int] = {
    0: _VCLS_PERSON,
    1: _VCLS_BICYCLE,
    2: 0,  # car
    3: _VCLS_MOTORCYCLE,
    5: 1,  # bus
    7: 2,  # truck
    9: 10,  # traffic light
    11: 9,  # stop sign
}


@dataclass
class _RawDet:
    vcls: int
    conf: float
    bbox_xyxy: np.ndarray
    source: str


def _resolve_weight_path(weights_dir: Path, filename_or_path: str) -> str:
    p = Path(filename_or_path)
    if p.exists():
        return str(p)
    local_w = weights_dir / p.name
    if local_w.exists():
        return str(local_w)
    return filename_or_path


def _move_downloaded_weight(weights_dir: Path, name: str) -> None:
    src = Path.cwd() / Path(name).name
    dst = weights_dir / Path(name).name
    if src.exists() and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)


def _merge_riders_and_two_wheelers(
    dets: list[_RawDet],
    overlap_thresh: float = 0.25,
    borderline_low: float = 0.08,
) -> list[InstancePred]:
    """Apply domain rules: merge overlapping person + motorcycle/bicycle into Motorcyclist (5) / Cyclist (7)."""
    vehicles = [d for d in dets if d.vcls in (_VCLS_MOTORCYCLE, _VCLS_BICYCLE)]
    persons = [d for d in dets if d.vcls == _VCLS_PERSON]
    explicit_riders = [d for d in dets if d.vcls in (5, 7)]
    others = [d for d in dets if d.vcls not in (_VCLS_MOTORCYCLE, _VCLS_BICYCLE, _VCLS_PERSON, 5, 7)]

    used_persons: set[int] = set()
    used_vehicles: set[int] = set()
    borderline_persons: set[int] = set()
    borderline_vehicles: set[int] = set()
    out: list[InstancePred] = []

    for er in explicit_riders:
        all_boxes = [er.bbox_xyxy]
        for vi, veh in enumerate(vehicles):
            if vi in used_vehicles:
                continue
            if containment_ratio_xyxy(veh.bbox_xyxy, er.bbox_xyxy) >= 0.45 or float(
                pairwise_iou_xyxy(veh.bbox_xyxy[None, :], er.bbox_xyxy[None, :])[0, 0]
            ) >= 0.40:
                all_boxes.append(veh.bbox_xyxy)
                used_vehicles.add(vi)
        for pi, per in enumerate(persons):
            if pi in used_persons:
                continue
            if containment_ratio_xyxy(per.bbox_xyxy, er.bbox_xyxy) >= 0.45 or float(
                pairwise_iou_xyxy(per.bbox_xyxy[None, :], er.bbox_xyxy[None, :])[0, 0]
            ) >= 0.40:
                all_boxes.append(per.bbox_xyxy)
                used_persons.add(pi)
        merged_box = np.array(
            [
                min(b[0] for b in all_boxes),
                min(b[1] for b in all_boxes),
                max(b[2] for b in all_boxes),
                max(b[3] for b in all_boxes),
            ],
            dtype=np.float32,
        )
        out.append(InstancePred(class_id=int(er.vcls), conf=er.conf, bbox_xyxy=merged_box, source=er.source))

    for vi, veh in enumerate(vehicles):
        if vi in used_vehicles:
            continue
        matched_p_indices: list[int] = []
        vx1, vy1, vx2, vy2 = veh.bbox_xyxy
        vw = max(1.0, vx2 - vx1)
        vh = max(1.0, vy2 - vy1)

        for pi, per in enumerate(persons):
            if pi in used_persons:
                continue
            px1, py1, px2, py2 = per.bbox_xyxy
            pw = max(1.0, px2 - px1)
            x_inter = max(0.0, min(vx2, px2) - max(vx1, px1))
            x_overlap = x_inter / max(1.0, min(vw, pw))

            y_inter = max(0.0, min(vy2, py2) - max(vy1, py1))
            p_contain = containment_ratio_xyxy(per.bbox_xyxy, veh.bbox_xyxy)
            v_contain = containment_ratio_xyxy(veh.bbox_xyxy, per.bbox_xyxy)
            iou = float(pairwise_iou_xyxy(veh.bbox_xyxy[None, :], per.bbox_xyxy[None, :])[0, 0])

            vertical_contact = (y_inter > 0) and (py2 <= vy2 + 0.18 * vh)
            score = max(iou, p_contain, v_contain, 0.7 * x_overlap if vertical_contact else 0.0)

            if score >= overlap_thresh and x_overlap >= 0.25 and vertical_contact:
                matched_p_indices.append(pi)
                used_persons.add(pi)
            elif score >= borderline_low:
                borderline_persons.add(pi)
                borderline_vehicles.add(vi)

        if matched_p_indices:
            all_boxes = [veh.bbox_xyxy] + [persons[pi].bbox_xyxy for pi in matched_p_indices]
            merged_box = np.array(
                [
                    min(b[0] for b in all_boxes),
                    min(b[1] for b in all_boxes),
                    max(b[2] for b in all_boxes),
                    max(b[3] for b in all_boxes),
                ],
                dtype=np.float32,
            )
            rider_confs = [veh.conf] + [persons[pi].conf for pi in matched_p_indices]
            merged_conf = float(np.mean(rider_confs))
            target_cls = 5 if veh.vcls == _VCLS_MOTORCYCLE else 7
            reasons = ["rider_ambiguity"] if vi in borderline_vehicles else []
            out.append(
                InstancePred(
                    class_id=target_cls,
                    conf=merged_conf,
                    bbox_xyxy=merged_box,
                    source=f"{veh.source}+rider_merge",
                    uncertainty_reasons=reasons,
                )
            )
        else:
            target_cls = 4 if veh.vcls == _VCLS_MOTORCYCLE else 6
            reasons = ["rider_ambiguity"] if vi in borderline_vehicles else []
            out.append(
                InstancePred(
                    class_id=target_cls,
                    conf=veh.conf,
                    bbox_xyxy=veh.bbox_xyxy,
                    source=veh.source,
                    uncertainty_reasons=reasons,
                )
            )

    for pi, per in enumerate(persons):
        if pi in used_persons:
            continue
        reasons = ["rider_ambiguity"] if pi in borderline_persons else []
        out.append(
            InstancePred(
                class_id=8,  # Pedestrian
                conf=per.conf,
                bbox_xyxy=per.bbox_xyxy,
                source=per.source,
                uncertainty_reasons=reasons,
            )
        )

    for oth in others:
        out.append(
            InstancePred(
                class_id=int(oth.vcls),
                conf=oth.conf,
                bbox_xyxy=oth.bbox_xyxy,
                source=oth.source,
            )
        )

    return out


class ColdStartBootstrapper:
    """Hybrid COCO-YOLO11 + YOLOE-PF specialist + legacy 4-class fusion bootstrapper."""

    def __init__(self, cfg: PipelineConfig, taxonomy: Taxonomy) -> None:
        self.cfg = cfg
        self.taxonomy = taxonomy
        cfg.configure_ultralytics_env()

        det_cfg = cfg.detector
        self.device = (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if det_cfg.get("device", "auto") == "auto"
            else det_cfg["device"]
        )
        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.coco_conf = max(0.22, float(det_cfg.get("predict_conf", 0.25)))
        self.pf_conf = 0.32

        from ultralytics import YOLO

        coco_name = det_cfg.get("bootstrap_coco_model", "yolo11n-seg.pt")
        coco_arg = _resolve_weight_path(cfg.dirs.weights_dir, coco_name)
        self.coco_model = YOLO(coco_arg)
        _move_downloaded_weight(cfg.dirs.weights_dir, coco_name)
        self.coco_classes = sorted(_COCO_CLASS_MAP.keys())

        # Optional prompt-free YOLOE specialist for Sign (9), Other vehicle (3), Traffic light (10)
        self.pf_model = None
        self.pf_map: dict[int, int] = {}
        self.pf_classes: list[int] = []
        yoloe_name = det_cfg.get("bootstrap_yoloe_model", "yoloe-11s-seg-pf.pt")
        pf_path = cfg.dirs.weights_dir / Path(yoloe_name).name
        if pf_path.exists() or "-pf" in Path(yoloe_name).stem:
            try:
                pf_arg = _resolve_weight_path(cfg.dirs.weights_dir, yoloe_name)
                pf_m = YOLO(pf_arg)
                _move_downloaded_weight(cfg.dirs.weights_dir, yoloe_name)
                names = pf_m.names if isinstance(pf_m.names, dict) else dict(enumerate(pf_m.names))
                self.pf_map = {
                    int(cid): _PF_SPECIALIST_TO_VCLS[str(cname).strip().lower()]
                    for cid, cname in names.items()
                    if str(cname).strip().lower() in _PF_SPECIALIST_TO_VCLS
                }
                self.pf_classes = sorted(self.pf_map.keys())
                self.pf_model = pf_m
            except Exception as exc:
                print(f"[Bootstrap] Specialist YOLOE-PF skipped ({exc})")

        self.backend_name = (
            f"YOLO11-COCO+YOLOE-PF({Path(coco_name).name}+{Path(yoloe_name).name})"
            if self.pf_model is not None
            else f"YOLO11-COCO({Path(coco_name).name})"
        )
        print(f"[Bootstrap] Detector ready: {self.backend_name} on {self.device}")

    def _detect_raw(self, img_path: Path) -> list[_RawDet]:
        dets: list[_RawDet] = []

        # 1. Calibrated COCO detections (Car, Bus, Truck, Motorcycle, Bicycle, Person, Traffic light)
        res_coco = self.coco_model.predict(
            source=str(img_path),
            conf=self.coco_conf,
            iou=0.55,
            imgsz=self.imgsz,
            classes=self.coco_classes,
            device=self.device,
            agnostic_nms=False,
            verbose=False,
        )
        if res_coco and res_coco[0].boxes is not None and len(res_coco[0].boxes) > 0:
            b = res_coco[0].boxes
            for box, conf, cid in zip(
                b.xyxy.detach().cpu().numpy(),
                b.conf.detach().cpu().numpy(),
                b.cls.detach().cpu().numpy().astype(int),
            ):
                vcls = _COCO_CLASS_MAP.get(int(cid))
                if vcls is not None:
                    dets.append(
                        _RawDet(
                            vcls=vcls,
                            conf=float(conf),
                            bbox_xyxy=np.asarray(box, dtype=np.float32),
                            source="yolo11_coco",
                        )
                    )

        # 2. Specialist YOLOE-PF detections (Road Sign=9, Trailers/Tricycles/Machinery=3, Traffic light=10)
        if self.pf_model is not None and self.pf_classes:
            res_pf = self.pf_model.predict(
                source=str(img_path),
                conf=self.pf_conf,
                iou=0.55,
                imgsz=self.imgsz,
                classes=self.pf_classes,
                device=self.device,
                agnostic_nms=False,
                verbose=False,
            )
            if res_pf and res_pf[0].boxes is not None and len(res_pf[0].boxes) > 0:
                b = res_pf[0].boxes
                for box, conf, cid in zip(
                    b.xyxy.detach().cpu().numpy(),
                    b.conf.detach().cpu().numpy(),
                    b.cls.detach().cpu().numpy().astype(int),
                ):
                    vcls = self.pf_map.get(int(cid))
                    if vcls is not None:
                        dets.append(
                            _RawDet(
                                vcls=vcls,
                                conf=float(conf),
                                bbox_xyxy=np.asarray(box, dtype=np.float32),
                                source="yoloe_pf",
                            )
                        )

        return dets

    def _fuse_with_legacy(
        self,
        instances: list[InstancePred],
        legacy_txt_path: Path,
        img_w: int,
        img_h: int,
    ) -> list[InstancePred]:
        legacy_boxes = read_legacy_bbox_file(legacy_txt_path, img_w, img_h)
        if not legacy_boxes:
            return instances

        legacy_map = self.taxonomy.legacy_4class_mapping
        match_iou = float(self.cfg.active_learning.get("legacy_match_iou", 0.30))

        pred_boxes = (
            np.stack([inst.bbox_xyxy for inst in instances], axis=0)
            if instances
            else np.zeros((0, 4), dtype=np.float32)
        )

        for leg_cls, leg_box in legacy_boxes:
            meta = legacy_map.get(leg_cls)
            if meta is None:
                continue
            if len(pred_boxes) > 0:
                ious = pairwise_iou_xyxy(leg_box[None, :], pred_boxes)[0]
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])
            else:
                best_idx, best_iou = -1, 0.0

            if best_iou >= match_iou and best_idx >= 0:
                matched = instances[best_idx]
                if matched.class_id in meta["candidate_target_classes"]:
                    if leg_cls in (1, 2):
                        matched.conf = min(0.92, max(matched.conf, float(meta["prior_conf"])))
                elif leg_cls == 0 and matched.class_id == 8 and best_iou >= 0.40:
                    # COCO detected only the person on a legacy 2-wheeler box -> upgrade to Motorcyclist (5) and flag for review
                    matched.class_id = 5
                    matched.bbox_xyxy = np.array(
                        [
                            min(matched.bbox_xyxy[0], leg_box[0]),
                            min(matched.bbox_xyxy[1], leg_box[1]),
                            max(matched.bbox_xyxy[2], leg_box[2]),
                            max(matched.bbox_xyxy[3], leg_box[3]),
                        ],
                        dtype=np.float32,
                    )
                    matched.conf = min(matched.conf, 0.65)
                    matched.uncertainty_reasons.append("legacy_upgraded_rider")
                else:
                    matched.uncertainty_reasons.append(f"legacy_class_conflict(old={leg_cls},new={matched.class_id})")
            else:
                prior_conf = float(meta["prior_conf"])
                target_cls = int(meta["default_target_class"])
                instances.append(
                    InstancePred(
                        class_id=target_cls,
                        conf=prior_conf,
                        bbox_xyxy=leg_box,
                        source="legacy_prior",
                        uncertainty_reasons=[f"legacy_prior_only(old={leg_cls})"],
                    )
                )
        return instances

    def predict_images(
        self,
        image_paths: list[Path],
        segmenter: Segmenter | None = None,
    ) -> dict[str, list[InstancePred]]:
        """Run hybrid detection + rider/vehicle fusion + legacy priors, then optionally refine via SAM."""
        al_cfg = self.cfg.active_learning
        overlap_thresh = float(al_cfg.get("rider_overlap_thresh", 0.25))
        borderline_low = float(al_cfg.get("rider_borderline_low", 0.08))
        nms_iou = float(self.cfg.detector.get("nms_iou", 0.55))
        pair_iou = float(al_cfg.get("pair_confusion_iou", 0.65))
        pair_margin = float(al_cfg.get("pair_confusion_margin", 0.20))

        preds_by_stem: dict[str, list[InstancePred]] = {}
        for img_path in image_paths:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            raw_dets = self._detect_raw(img_path)
            fused = _merge_riders_and_two_wheelers(
                raw_dets,
                overlap_thresh=overlap_thresh,
                borderline_low=borderline_low,
            )
            deduped = annotate_pair_confusion(
                fused,
                nms_iou=nms_iou,
                pair_confusion_iou=pair_iou,
                pair_confusion_margin=pair_margin,
                taxonomy=self.taxonomy,
            )
            legacy_txt = self.cfg.legacy_labels_dir / f"{img_path.stem}.txt"
            deduped = self._fuse_with_legacy(deduped, legacy_txt, w, h)
            preds_by_stem[img_path.stem] = deduped

        if segmenter is not None:
            for img_path in image_paths:
                stem = img_path.stem
                if preds_by_stem.get(stem):
                    segmenter.refine_instances(img_path, preds_by_stem[stem])

        return preds_by_stem

    def close(self) -> None:
        self.coco_model = None
        self.pf_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
