from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

from .config import PipelineConfig
from .geometry import InstancePred, mask_bbox_iou, mask_to_polygon


def _hardware_profile() -> tuple[float, float]:
    """Return (gpu_vram_gb, system_ram_gb)."""
    vram_gb = 0.0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = float(props.total_memory) / (1024**3)
    ram_gb = float(psutil.virtual_memory().total) / (1024**3)
    return vram_gb, ram_gb


class Segmenter:
    """Unified SAM 3 / SAM 2.1 box-prompted segmenter with automatic hardware-safe fallback."""

    def __init__(
        self,
        cfg: PipelineConfig,
        force_sam3: bool = False,
        force_fallback: bool = False,
        device: str = "auto",
    ) -> None:
        self.cfg = cfg
        cfg.configure_ultralytics_env()

        seg_cfg = cfg.segmenter
        self.sam_conf = float(seg_cfg.get("sam_conf", 0.25))
        self.epsilon_ratio = float(seg_cfg.get("polygon_epsilon_ratio", 0.002))
        self.min_area_px = int(seg_cfg.get("min_mask_area_px", 24))
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        sam3_path = (cfg.project_root / seg_cfg.get("sam3_weights", "weights/sam3.pt")).resolve()
        fb_path = (cfg.project_root / seg_cfg.get("fallback_weights", "weights/sam2.1_t.pt")).resolve()
        fb_name = seg_cfg.get("fallback_auto_name", "sam2.1_t.pt")

        vram_gb, ram_gb = _hardware_profile()
        min_vram = float(seg_cfg.get("min_vram_gb_for_sam3", 7.5))
        min_ram = float(seg_cfg.get("min_ram_gb_for_sam3", 11.0))

        use_sam3 = False
        reason = ""
        if force_fallback:
            reason = "force_fallback=True"
        elif sam3_path.exists() and (seg_cfg.get("prefer_sam3", True) or force_sam3):
            if force_sam3 or (vram_gb >= min_vram and ram_gb >= min_ram):
                use_sam3 = True
            else:
                reason = (
                    f"sam3.pt found at {sam3_path}, but hardware ({vram_gb:.1f}GB VRAM / {ram_gb:.1f}GB RAM) "
                    f"is below safe SAM3 threshold ({min_vram}GB VRAM / {min_ram}GB RAM); using lightweight SAM2.1"
                )
        else:
            reason = f"sam3.pt not found at {sam3_path}; using lightweight SAM2.1 fallback"

        self.backend_name: str
        self.weights_used: Path
        if use_sam3:
            self.backend_name = "SAM3"
            self.weights_used = sam3_path
            self.imgsz = int(seg_cfg.get("sam3_imgsz", 1008))
            from ultralytics.models.sam import SAM3Predictor

            self.predictor = SAM3Predictor(
                overrides={
                    "model": str(sam3_path),
                    "conf": 0.05,
                    "imgsz": self.imgsz,
                    "device": self.device,
                    "verbose": False,
                }
            )
        else:
            self.backend_name = "SAM2.1-fallback"
            self.imgsz = int(seg_cfg.get("sam_imgsz", 1024))
            print(f"[Segmenter] {reason}")
            if fb_path.exists():
                weight_arg = str(fb_path)
                self.weights_used = fb_path
            else:
                # Download/cache inside weights_dir cleanly
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                weight_arg = str(fb_path) if (fb_path.parent / fb_name).exists() else fb_name
                self.weights_used = fb_path
            from ultralytics.models.sam import SAM2Predictor

            self.predictor = SAM2Predictor(
                overrides={
                    "model": weight_arg,
                    "conf": 0.05,
                    "imgsz": self.imgsz,
                    "device": self.device,
                    "verbose": False,
                }
            )
            # Trigger model setup once now and copy cached weights to weights/ if downloaded to cwd
            self.predictor.setup_model(verbose=False)
            cwd_weight = Path.cwd() / fb_name
            if cwd_weight.exists() and not fb_path.exists():
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                cwd_weight.replace(fb_path)

        if self.predictor.model is None:
            self.predictor.setup_model(verbose=False)
        print(f"[Segmenter] Ready: backend={self.backend_name}, imgsz={self.imgsz}, device={self.device}")

    def segment_boxes(
        self,
        img_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        chunk_size: int = 12,
    ) -> tuple[list[np.ndarray | None], list[float]]:
        """Generate binary masks and mask scores aligned 1-to-1 with input (N, 4) xyxy boxes."""
        n = len(boxes_xyxy)
        if n == 0:
            return [], []

        masks_out: list[np.ndarray | None] = [None] * n
        scores_out: list[float] = [0.0] * n

        self.predictor.set_image(img_bgr)
        try:
            src_shape = img_bgr.shape[:2]
            dst_shape = (self.imgsz, self.imgsz)
            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                chunk_boxes = np.asarray(boxes_xyxy[start:end], dtype=np.float32)
                pred_masks, pred_boxes = self.predictor.inference_features(
                    self.predictor.features,
                    src_shape=src_shape,
                    dst_shape=dst_shape,
                    bboxes=chunk_boxes,
                    multimask_output=False,
                )
                if pred_masks is None or len(pred_masks) == 0:
                    continue
                masks_np = pred_masks.detach().cpu().numpy()
                boxes_np = pred_boxes.detach().cpu().numpy()
                for local_idx in range(min(len(masks_np), end - start)):
                    prompt_slot = start + int(boxes_np[local_idx, 5]) if boxes_np.shape[1] >= 6 else start + local_idx
                    if 0 <= prompt_slot < n:
                        masks_out[prompt_slot] = masks_np[local_idx].astype(bool)
                        scores_out[prompt_slot] = float(boxes_np[local_idx, 4])
        finally:
            self.predictor.reset_image()

        return masks_out, scores_out

    def refine_instances(
        self,
        img_path: Path,
        instances: list[InstancePred],
        keep_existing_poly_if_sam_fails: bool = False,
    ) -> list[InstancePred]:
        """Run SAM on each instance's bbox prompt and populate clean YOLO-seg polygons."""
        if not instances:
            return instances
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        h, w = img_bgr.shape[:2]

        boxes = np.stack([inst.bbox_xyxy for inst in instances], axis=0)
        masks, sam_scores = self.segment_boxes(img_bgr, boxes)

        for inst, mask, s_score in zip(instances, masks, sam_scores):
            if mask is None or s_score < self.sam_conf:
                if not keep_existing_poly_if_sam_fails:
                    inst.polygon = None
                    inst.mask_bbox_iou = 0.0
                continue
            iou = mask_bbox_iou(mask, inst.bbox_xyxy)
            poly = mask_to_polygon(
                mask,
                img_w=w,
                img_h=h,
                epsilon_ratio=self.epsilon_ratio,
                min_area_px=self.min_area_px,
            )
            inst.mask_bbox_iou = iou
            if poly is not None:
                inst.polygon = poly
                inst.source = f"{inst.source}+{self.backend_name}"
            elif not keep_existing_poly_if_sam_fails:
                inst.polygon = None

        return instances

    def close(self) -> None:
        """Release model weights and CUDA cache so training/other stages fit in 4GB VRAM / 8GB RAM."""
        if hasattr(self, "predictor") and self.predictor is not None:
            self.predictor.reset_image()
            self.predictor.model = None
            self.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
