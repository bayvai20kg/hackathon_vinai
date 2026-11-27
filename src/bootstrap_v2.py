"""Pipeline V2: bootstrap + SAM tối ưu cho đủ 11 lớp.

Giữ nguyên V1 (src/bootstrap.py, src/segmenter.py). V2 kế thừa và thay phần yếu:
  1. Detector lớn (yolo11x-seg, yoloe-26l-pf) + YOLOE-26l text prompt cho lớp hiếm 3/6/7/9/10.
  2. Suy luận cả ảnh ở 1280 + chia ô 2x2 phóng to, gộp bằng NMS + loại hộp bị cắt ở mép ô.
  3. Ngưỡng conf theo từng lớp.
  4. Người đang dắt xe -> Pedestrian (8), xe -> không người (4/6) thay vì bị ghép thành 5/7.
  5. Lọc biển/đèn: quá to (quảng cáo) hoặc nằm trong thân xe (đèn hậu, chữ trên xe).
  6. Tách đầu kéo (2) khỏi rơ-moóc (3) khi phát hiện được rơ-moóc chồng lên hộp xe tải.
  7. SAM: cắt mask biển/đèn trong hộp, cắt cột biển báo, polygon sát biên hơn.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .active_learning import annotate_pair_confusion
from .bootstrap import (
    _COCO_CLASS_MAP,
    _PF_SPECIALIST_TO_VCLS,
    _VCLS_BICYCLE,
    _VCLS_MOTORCYCLE,
    _VCLS_PERSON,
    ColdStartBootstrapper,
    _merge_riders_and_two_wheelers,
    _move_downloaded_weight,
    _RawDet,
    _resolve_weight_path,
)
from .config import PipelineConfig, Taxonomy
from .geometry import InstancePred, containment_ratio_xyxy, mask_bbox_iou, mask_to_polygon, pairwise_iou_xyxy
from .segmenter import Segmenter

# Người đứng cạnh xe (đang dắt) — id tạm, không bị luật ghép người+xe nuốt mất, đổi về 8 sau khi ghép
_VCLS_WALKER = 118

_VEHICLE_CLASSES = (0, 1, 2, 3)

_DEFAULT_V2 = {
    "full_imgsz": 1280,
    "tile_grid": [2, 2],
    "tile_overlap": 0.20,
    "tile_imgsz": 960,
    "merge_iou": 0.55,
    "merge_containment": 0.80,
    "class_conf": {},
    "default_conf": 0.30,
    "pf_conf": 0.25,
    "text_prompts": {},
    "signal_max_area_frac": 0.02,
    "signal_min_side_px": 6,
    "signal_in_vehicle_contain": 0.60,
    "walker_feet_tol": 0.10,
    "walker_lateral_offset": 0.30,
    "trailer_overlap_contain": 0.50,
    "truck_min_keep_frac": 0.20,
}


def load_v2_options(cfg: PipelineConfig, config_path: Path | None = None) -> dict:
    path = config_path or (cfg.project_root / "configs" / "pipeline_v2.yaml")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    opts = dict(_DEFAULT_V2)
    opts.update(raw.get("bootstrap_v2", {}) or {})
    opts["class_conf"] = {int(k): float(v) for k, v in (opts.get("class_conf") or {}).items()}
    opts["text_prompts"] = {str(k): int(v) for k, v in (opts.get("text_prompts") or {}).items()}
    return opts


# ------------------------------------------------------------------------------------------ tiện ích hình học
def _tiles(w: int, h: int, grid: list[int], overlap: float) -> list[tuple[int, int, int, int]]:
    cols, rows = int(grid[0]), int(grid[1])
    if cols <= 1 and rows <= 1:
        return []
    tw = int(np.ceil(w / (cols - (cols - 1) * overlap))) if cols > 1 else w
    th = int(np.ceil(h / (rows - (rows - 1) * overlap))) if rows > 1 else h
    xs = [0] if cols <= 1 else [int(round(i * (w - tw) / (cols - 1))) for i in range(cols)]
    ys = [0] if rows <= 1 else [int(round(j * (h - th) / (rows - 1))) for j in range(rows)]
    return [(x, y, min(w, x + tw), min(h, y + th)) for y in ys for x in xs]


def _merge_dets(dets: list[_RawDet], iou_thr: float, contain_thr: float) -> list[_RawDet]:
    """NMS theo lớp + bỏ hộp cùng lớp nằm gần trọn trong hộp conf cao hơn (vật bị cắt ở mép ô)."""
    if not dets:
        return []
    from torchvision.ops import batched_nms

    boxes = torch.tensor(np.stack([d.bbox_xyxy for d in dets]), dtype=torch.float32)
    scores = torch.tensor([d.conf for d in dets], dtype=torch.float32)
    labels = torch.tensor([d.vcls for d in dets], dtype=torch.int64)
    keep = batched_nms(boxes, scores, labels, iou_thr).tolist()
    kept = [dets[i] for i in keep]  # đã sắp theo conf giảm dần

    out: list[_RawDet] = []
    for d in kept:
        if any(o.vcls == d.vcls and containment_ratio_xyxy(d.bbox_xyxy, o.bbox_xyxy) >= contain_thr for o in out):
            continue
        out.append(d)
    return out


def _union_box(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])], dtype=np.float32)


# ------------------------------------------------------------------------------------------------ bootstrap V2
class ColdStartBootstrapperV2(ColdStartBootstrapper):
    """COCO-x + YOLOE-PF-l + YOLOE-text-l, chạy cả ảnh + chia ô, kèm luật hậu xử lý cho 11 lớp."""

    def __init__(self, cfg: PipelineConfig, taxonomy: Taxonomy) -> None:
        self.cfg = cfg
        self.taxonomy = taxonomy
        self.opts = load_v2_options(cfg)
        cfg.configure_ultralytics_env()

        det_cfg = cfg.detector
        self.device = (
            ("cuda:0" if torch.cuda.is_available() else "cpu")
            if det_cfg.get("device", "auto") == "auto"
            else det_cfg["device"]
        )
        self.imgsz = int(self.opts["full_imgsz"])
        conf_values = list(self.opts["class_conf"].values()) + [float(self.opts["default_conf"])]
        self.min_conf = max(0.05, min(conf_values))

        from ultralytics import YOLO, YOLOE

        wdir = cfg.dirs.weights_dir
        coco_name = det_cfg.get("bootstrap_coco_model", "yolo11x-seg.pt")
        self.coco_model = YOLO(_resolve_weight_path(wdir, coco_name))
        _move_downloaded_weight(wdir, coco_name)
        self.coco_classes = sorted(_COCO_CLASS_MAP.keys())

        # Prompt-free YOLOE (4585 lớp) cho biển/đèn/phương tiện khác
        self.pf_model, self.pf_map, self.pf_classes = None, {}, []
        pf_name = det_cfg.get("bootstrap_yoloe_model", "yoloe-26l-seg-pf.pt")
        try:
            pf_m = YOLO(_resolve_weight_path(wdir, pf_name))
            _move_downloaded_weight(wdir, pf_name)
            names = pf_m.names if isinstance(pf_m.names, dict) else dict(enumerate(pf_m.names))
            self.pf_map = {
                int(cid): _PF_SPECIALIST_TO_VCLS[str(n).strip().lower()]
                for cid, n in names.items()
                if str(n).strip().lower() in _PF_SPECIALIST_TO_VCLS
            }
            self.pf_classes = sorted(self.pf_map)
            self.pf_model = pf_m
        except Exception as exc:  # noqa: BLE001
            print(f"[BootstrapV2] YOLOE-PF bỏ qua ({exc})")

        # YOLOE text prompt cho lớp hiếm
        self.text_model, self.text_map = None, {}
        prompts = self.opts["text_prompts"]
        text_name = det_cfg.get("bootstrap_text_model", "yoloe-26l-seg.pt")
        if prompts:
            try:
                tm = YOLOE(_resolve_weight_path(wdir, text_name))
                _move_downloaded_weight(wdir, text_name)
                names = list(prompts.keys())
                tm.set_classes(names, tm.get_text_pe(names))
                self.text_map = {i: prompts[n] for i, n in enumerate(names)}
                self.text_model = tm
            except Exception as exc:  # noqa: BLE001
                print(f"[BootstrapV2] YOLOE text prompt bỏ qua ({exc})")

        parts = [Path(coco_name).name]
        if self.pf_model is not None:
            parts.append(Path(pf_name).name)
        if self.text_model is not None:
            parts.append(f"{Path(text_name).name}[{len(self.text_map)} prompts]")
        self.backend_name = "V2-tiled(" + "+".join(parts) + ")"
        print(f"[BootstrapV2] Detector ready: {self.backend_name} on {self.device}")

    # -------------------------------------------------------------------------------------- phát hiện
    def _conf_for(self, vcls: int) -> float:
        return float(self.opts["class_conf"].get(vcls, self.opts["default_conf"]))

    def _run_model(self, model, img: np.ndarray, mapping, classes, source: str, conf: float) -> list[_RawDet]:
        h, w = img.shape[:2]
        views = [(0, 0, w, h, self.imgsz)] + [
            (x1, y1, x2, y2, int(self.opts["tile_imgsz"]))
            for x1, y1, x2, y2 in _tiles(w, h, self.opts["tile_grid"], float(self.opts["tile_overlap"]))
        ]
        out: list[_RawDet] = []
        for x1, y1, x2, y2, sz in views:
            crop = img[y1:y2, x1:x2]
            res = model.predict(
                source=crop, conf=conf, iou=0.55, imgsz=sz, classes=classes,
                device=self.device, verbose=False, retina_masks=False,
            )
            if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                continue
            b = res[0].boxes
            for box, c, cid in zip(
                b.xyxy.detach().cpu().numpy(), b.conf.detach().cpu().numpy(), b.cls.detach().cpu().numpy().astype(int)
            ):
                vcls = mapping(int(cid))
                if vcls is None or float(c) < self._conf_for(vcls):
                    continue
                gbox = np.asarray(box, dtype=np.float32) + np.array([x1, y1, x1, y1], dtype=np.float32)
                tag = source if (x2 - x1, y2 - y1) == (w, h) else f"{source}_tile"
                out.append(_RawDet(vcls=vcls, conf=float(c), bbox_xyxy=gbox, source=tag))
        return out

    def _detect_raw(self, img_path: Path) -> list[_RawDet]:
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        dets = self._run_model(
            self.coco_model, img, _COCO_CLASS_MAP.get, self.coco_classes, "yolo11x_coco", self.min_conf
        )
        if self.pf_model is not None and self.pf_classes:
            dets += self._run_model(
                self.pf_model, img, self.pf_map.get, self.pf_classes, "yoloe_pf",
                max(self.min_conf, float(self.opts["pf_conf"])),
            )
        if self.text_model is not None:
            dets += self._run_model(self.text_model, img, self.text_map.get, None, "yoloe_text", self.min_conf)
        merged = _merge_dets(dets, float(self.opts["merge_iou"]), float(self.opts["merge_containment"]))
        return self._suppress_confusions(merged, img.shape[0])

    def _suppress_confusions(self, dets: list[_RawDet], img_h: int) -> list[_RawDet]:
        """Lọc nhầm lẫn hay gặp của model open-vocab trên ảnh giao thông VN.

        - Other vehicle (3) chỉ là xe máy/người mặc áo mưa: hộp 3 phủ một xe máy mà diện tích không lớn hơn
          xe máy bao nhiêu -> bỏ (ba gác, xe lam thật thì to hơn hẳn một xe máy).
        - Other vehicle (3) treo trên cao (đáy hộp ở dải trên cùng ảnh) -> bảng quảng cáo, bỏ.
        - Xe đạp (106) trùng hộp xe máy (104) -> model nhầm xe máy thành xe đạp, bỏ.
        """
        ratio = float(self.opts.get("other_vs_moto_area_ratio", 1.8))
        top_frac = float(self.opts.get("other_min_bottom_frac", 0.20))
        two_wheel_iou = float(self.opts.get("bike_vs_moto_iou", 0.30))
        motos = [d for d in dets if d.vcls in (_VCLS_MOTORCYCLE, 5)]
        riders = motos + [d for d in dets if d.vcls == _VCLS_PERSON]

        def area(b):
            return max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))

        out = []
        for d in dets:
            if d.vcls == 3:
                if d.bbox_xyxy[3] < top_frac * img_h:
                    continue
                if any(
                    containment_ratio_xyxy(m.bbox_xyxy, d.bbox_xyxy) >= 0.6 and area(d.bbox_xyxy) < ratio * area(m.bbox_xyxy)
                    for m in riders
                ):
                    continue
            if d.vcls == _VCLS_BICYCLE and any(
                float(pairwise_iou_xyxy(d.bbox_xyxy[None], m.bbox_xyxy[None])[0, 0]) >= two_wheel_iou
                or containment_ratio_xyxy(d.bbox_xyxy, m.bbox_xyxy) >= 0.7
                for m in motos
            ):
                continue
            out.append(d)
        return out

    # ------------------------------------------------------------------------------------- luật V2
    def _tag_walkers(self, dets: list[_RawDet]) -> None:
        """Người đứng cạnh xe máy/xe đạp, chân chạm đất ngang đáy xe -> người dắt xe (Pedestrian)."""
        tol = float(self.opts["walker_feet_tol"])
        lat = float(self.opts["walker_lateral_offset"])
        vehicles = [d for d in dets if d.vcls in (_VCLS_MOTORCYCLE, _VCLS_BICYCLE)]
        for p in dets:
            if p.vcls != _VCLS_PERSON:
                continue
            px1, py1, px2, py2 = p.bbox_xyxy
            pcx = 0.5 * (px1 + px2)
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v.bbox_xyxy
                vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
                if float(pairwise_iou_xyxy(p.bbox_xyxy[None], v.bbox_xyxy[None])[0, 0]) <= 0.0:
                    continue
                feet_on_ground = abs(py2 - vy2) <= tol * vh
                beside = abs(pcx - 0.5 * (vx1 + vx2)) >= lat * vw
                if feet_on_ground and beside:
                    p.vcls = _VCLS_WALKER
                    break

    def _filter_signals(self, insts: list[InstancePred], w: int, h: int) -> list[InstancePred]:
        max_area = float(self.opts["signal_max_area_frac"]) * w * h
        min_side = float(self.opts["signal_min_side_px"])
        contain = float(self.opts["signal_in_vehicle_contain"])
        vehicles = [i.bbox_xyxy for i in insts if i.class_id in _VEHICLE_CLASSES + (4, 5, 6, 7)]
        out = []
        for inst in insts:
            if inst.class_id in (9, 10):
                bw, bh = inst.bbox_xyxy[2] - inst.bbox_xyxy[0], inst.bbox_xyxy[3] - inst.bbox_xyxy[1]
                if bw * bh > max_area or min(bw, bh) < min_side:
                    continue
                if any(containment_ratio_xyxy(inst.bbox_xyxy, vb) >= contain for vb in vehicles):
                    continue
            out.append(inst)
        return out

    def _split_truck_trailer(self, insts: list[InstancePred]) -> None:
        """Hộp xe tải chứa rơ-moóc (lớp 3) -> thu hộp xe tải về phần cabin không trùng rơ-moóc."""
        thr = float(self.opts["trailer_overlap_contain"])
        keep_frac = float(self.opts["truck_min_keep_frac"])
        trailers = [i for i in insts if i.class_id == 3]
        for t in (i for i in insts if i.class_id == 2):
            tb = t.bbox_xyxy
            area = max(1.0, (tb[2] - tb[0]) * (tb[3] - tb[1]))
            for r in trailers:
                rb = r.bbox_xyxy
                if containment_ratio_xyxy(rb, tb) < thr:
                    continue
                cands = [
                    np.array([tb[0], tb[1], rb[0], tb[3]], dtype=np.float32),  # cabin bên trái
                    np.array([rb[2], tb[1], tb[2], tb[3]], dtype=np.float32),  # cabin bên phải
                    np.array([tb[0], tb[1], tb[2], rb[1]], dtype=np.float32),  # cabin phía trên
                    np.array([tb[0], rb[3], tb[2], tb[3]], dtype=np.float32),  # cabin phía dưới
                ]
                cands = [c for c in cands if c[2] > c[0] and c[3] > c[1]]
                if not cands:
                    continue
                best = max(cands, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
                if (best[2] - best[0]) * (best[3] - best[1]) >= keep_frac * area:
                    t.bbox_xyxy = best
                    t.uncertainty_reasons.append("truck_trailer_split_auto")
                    break

    # ---------------------------------------------------------------------------------- toàn bộ
    def predict_images(self, image_paths: list[Path], segmenter: Segmenter | None = None) -> dict[str, list[InstancePred]]:
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
            raw = self._detect_raw(img_path)
            self._tag_walkers(raw)
            fused = _merge_riders_and_two_wheelers(raw, overlap_thresh=overlap_thresh, borderline_low=borderline_low)
            for inst in fused:
                if inst.class_id == _VCLS_WALKER:
                    inst.class_id = 8
                    inst.uncertainty_reasons.append("walking_beside_vehicle")
            deduped = annotate_pair_confusion(
                fused, nms_iou=nms_iou, pair_confusion_iou=pair_iou, pair_confusion_margin=pair_margin, taxonomy=self.taxonomy
            )
            deduped = self._fuse_with_legacy(deduped, self.cfg.legacy_labels_dir / f"{img_path.stem}.txt", w, h)
            deduped = self._filter_signals(deduped, w, h)
            self._split_truck_trailer(deduped)
            preds_by_stem[img_path.stem] = deduped

        if segmenter is not None:
            for img_path in image_paths:
                if preds_by_stem.get(img_path.stem):
                    segmenter.refine_instances(img_path, preds_by_stem[img_path.stem])
        return preds_by_stem

    def close(self) -> None:
        self.text_model = None
        super().close()


# ------------------------------------------------------------------------------------------------ SAM V2
def _trim_pole(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Giữ đoạn hàng (rồi đoạn cột) liền nhau quanh chỗ mask rộng nhất; bỏ phần hẹp như cột/cần treo."""
    out = mask.copy()
    for axis in (1, 0):  # axis=1: độ rộng theo hàng (cột đứng); axis=0: chiều cao theo cột (cần ngang)
        prof = out.sum(axis=axis)
        if prof.max() <= 0:
            return mask
        ok = prof >= ratio * prof.max()
        peak = int(np.argmax(prof))
        lo, hi = peak, peak
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(ok) - 1 and ok[hi + 1]:
            hi += 1
        if axis == 1:
            out[:lo] = False
            out[hi + 1 :] = False
        else:
            out[:, :lo] = False
            out[:, hi + 1 :] = False
    return out if out.sum() >= 0.3 * mask.sum() else mask


class SegmenterV2(Segmenter):
    """SAM như V1, thêm: cắt mask trong hộp cho biển/đèn, cắt cột biển báo."""

    def __init__(self, cfg: PipelineConfig, **kw) -> None:
        super().__init__(cfg, **kw)
        seg = cfg.segmenter
        self.clip_classes = set(int(c) for c in seg.get("clip_to_box_classes", [9, 10]))
        self.clip_margin = float(seg.get("clip_box_margin_frac", 0.04))
        self.pole_classes = set(int(c) for c in seg.get("pole_trim_classes", [9]))
        self.pole_ratio = float(seg.get("pole_trim_ratio", 0.35))
        self.backend_name = f"{self.backend_name}+V2post"

    def _post(self, cls: int, mask: np.ndarray, box: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        if cls in self.clip_classes:
            bw, bh = box[2] - box[0], box[3] - box[1]
            mx, my = self.clip_margin * bw + 2, self.clip_margin * bh + 2
            x1, y1 = max(0, int(box[0] - mx)), max(0, int(box[1] - my))
            x2, y2 = min(w, int(np.ceil(box[2] + mx))), min(h, int(np.ceil(box[3] + my)))
            clipped = np.zeros_like(mask)
            clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            mask = clipped
        if cls in self.pole_classes and mask.any():
            mask = _trim_pole(mask, self.pole_ratio)
        return mask

    def refine_instances(self, img_path: Path, instances: list[InstancePred], keep_existing_poly_if_sam_fails: bool = False):
        if not instances:
            return instances
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        h, w = img.shape[:2]
        boxes = np.stack([i.bbox_xyxy for i in instances], axis=0)
        masks, scores = self.segment_boxes(img, boxes)
        for inst, mask, s in zip(instances, masks, scores):
            if mask is None or s < self.sam_conf:
                if not keep_existing_poly_if_sam_fails:
                    inst.polygon, inst.mask_bbox_iou = None, 0.0
                continue
            mask = self._post(int(inst.class_id), mask.astype(bool), inst.bbox_xyxy)
            inst.mask_bbox_iou = mask_bbox_iou(mask, inst.bbox_xyxy)
            poly = mask_to_polygon(mask, img_w=w, img_h=h, epsilon_ratio=self.epsilon_ratio, min_area_px=self.min_area_px)
            if poly is not None:
                inst.polygon = poly
                inst.source = f"{inst.source}+{self.backend_name}"
            elif not keep_existing_poly_if_sam_fails:
                inst.polygon = None
        return instances
