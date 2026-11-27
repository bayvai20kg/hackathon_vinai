from __future__ import annotations

import gc
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from .active_learning import annotate_pair_confusion
from .config import PipelineConfig, Taxonomy
from .geometry import InstancePred


def _camera_group_key(stem: str) -> str:
    """Group frames by camera prefix (e.g., day_cam_01) for balanced splitting."""
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else stem


def _find_matching_image(folder: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_yolo_seg_split(
    samples: dict[str, tuple[Path, Path, str]],
    out_ds_dir: Path,
    taxonomy: Taxonomy,
    val_ratio: float = 0.15,
    seed: int = 42,
    camera_grouped: bool = True,
    min_images_for_split: int = 5,
) -> dict[str, Any]:
    """Create train/val YOLO-seg directory + data.yaml from {stem: (img_path, lbl_path, kind)}."""
    if out_ds_dir.exists():
        shutil.rmtree(out_ds_dir)
    for split in ("train", "val"):
        (out_ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if not samples:
        raise ValueError("No labeled samples provided to build YOLO-seg dataset.")

    stems = sorted(samples.keys())
    rng = random.Random(seed)
    val_stems: set[str] = set()

    if len(stems) >= min_images_for_split:
        if camera_grouped:
            by_cam: dict[str, list[str]] = {}
            for s in stems:
                by_cam.setdefault(_camera_group_key(s), []).append(s)
            for cam_stems in by_cam.values():
                if len(cam_stems) >= 3:
                    k = max(1, int(round(len(cam_stems) * val_ratio)))
                    k = min(k, len(cam_stems) - 1)
                    val_stems.update(rng.sample(cam_stems, k))
            if not val_stems:
                k = max(1, int(round(len(stems) * val_ratio)))
                val_stems.update(rng.sample(stems, min(k, len(stems) - 1)))
        else:
            k = max(1, int(round(len(stems) * val_ratio)))
            val_stems.update(rng.sample(stems, min(k, len(stems) - 1)))
    else:
        val_stems.add(stems[-1])

    counts = {"train": 0, "val": 0, "human": 0, "auto": 0, "other": 0}
    for stem in stems:
        img_src, lbl_src, src_kind = samples[stem]
        split = "val" if stem in val_stems else "train"
        shutil.copy2(img_src, out_ds_dir / "images" / split / img_src.name)
        shutil.copy2(lbl_src, out_ds_dir / "labels" / split / f"{stem}.txt")
        counts[split] += 1
        counts[src_kind if src_kind in ("human", "auto") else "other"] += 1

    data_yaml_path = out_ds_dir / "data.yaml"
    data_yaml = {
        "path": str(out_ds_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": taxonomy.num_classes,
        "names": taxonomy.id_to_name,
    }
    data_yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    return {
        "data_yaml": data_yaml_path,
        "total_images": len(stems),
        "train_images": counts["train"],
        "val_images": counts["val"],
        "human_verified_images": counts["human"],
        "auto_labeled_images": counts["auto"],
    }


def _index_images(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {p.stem: p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in exts}


def prepare_yolo_seg_dataset(
    cfg: PipelineConfig,
    taxonomy: Taxonomy,
    out_ds_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge human_verified (priority) + auto_labeled into a clean YOLO-seg train/val dataset."""
    src_index = _index_images(cfg.images_dir)
    samples: dict[str, tuple[Path, Path, str]] = {}
    for lbl in sorted(cfg.dirs.auto_labeled_labels.glob("*.txt")):
        stem = lbl.stem
        img = _find_matching_image(cfg.dirs.auto_labeled_images, stem) or src_index.get(stem)
        if img is not None:
            samples[stem] = (img, lbl, "auto")

    for lbl in sorted(cfg.dirs.human_verified_labels.glob("*.txt")):
        stem = lbl.stem
        img = _find_matching_image(cfg.dirs.human_verified_images, stem) or src_index.get(stem)
        if img is not None:
            samples[stem] = (img, lbl, "human")

    if not samples:
        raise ValueError("No labeled samples found in human_verified/ or auto_labeled/ to train YOLO11-seg.")

    return build_yolo_seg_split(
        samples=samples,
        out_ds_dir=out_ds_dir or cfg.dirs.yolo_dataset_dir,
        taxonomy=taxonomy,
        val_ratio=float(cfg.dataset_split.get("val_ratio", 0.15)),
        seed=int(cfg.dataset_split.get("seed", 42)),
        camera_grouped=bool(cfg.dataset_split.get("camera_grouped", True)),
    )


def train_yolo11_seg(
    cfg: PipelineConfig,
    taxonomy: Taxonomy,
    run_name: str = "round_01",
    weights_init: str | Path | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    custom_samples: dict[str, tuple[Path, Path, str]] | None = None,
    custom_ds_dir: Path | None = None,
    data_yaml: str | Path | None = None,
) -> dict[str, Any]:
    """Train or retrain YOLO11-seg on merged production data, custom sample dict, or an existing data.yaml."""
    cfg.configure_ultralytics_env()

    if data_yaml is not None:
        data_yaml_path = Path(data_yaml).resolve()
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
        ds_info = {"data_yaml": str(data_yaml_path), "custom_data_yaml": True}
    elif custom_samples is not None:
        ds_info = build_yolo_seg_split(
            samples=custom_samples,
            out_ds_dir=custom_ds_dir or (cfg.dirs.work_dir / "custom_yolo_ds"),
            taxonomy=taxonomy,
            val_ratio=float(cfg.dataset_split.get("val_ratio", 0.15)),
            seed=int(cfg.dataset_split.get("seed", 42)),
            camera_grouped=bool(cfg.dataset_split.get("camera_grouped", True)),
        )
    else:
        ds_info = prepare_yolo_seg_dataset(cfg, taxonomy, out_ds_dir=custom_ds_dir)

    det_cfg = cfg.detector
    base_model = str(weights_init or det_cfg.get("base_model", "yolo11n-seg.pt"))
    if not Path(base_model).exists():
        candidate = cfg.dirs.weights_dir / Path(base_model).name
        if candidate.exists():
            base_model = str(candidate)

    device = ("0" if torch.cuda.is_available() else "cpu") if det_cfg.get("device", "auto") == "auto" else det_cfg["device"]
    epochs_val = int(epochs if epochs is not None else det_cfg.get("epochs_warmup", 30))
    batch_val = int(batch if batch is not None else det_cfg.get("batch", 4))
    imgsz_val = int(imgsz if imgsz is not None else det_cfg.get("imgsz", 640))
    workers_val = int(det_cfg.get("workers", 0))
    amp_val = bool(det_cfg.get("amp", True))

    from ultralytics import YOLO

    model = YOLO(base_model)
    cwd_base = Path.cwd() / Path(base_model).name
    dst_base = cfg.dirs.weights_dir / Path(base_model).name
    if cwd_base.exists() and not dst_base.exists() and cwd_base.resolve() != dst_base.resolve():
        dst_base.parent.mkdir(parents=True, exist_ok=True)
        cwd_base.replace(dst_base)

    epoch_bar = tqdm(total=epochs_val, desc=f"Train {Path(base_model).stem} ({run_name})", unit="epoch", leave=True)

    def _on_fit_epoch_end(trainer) -> None:
        epoch_bar.n = min(epochs_val, int(trainer.epoch) + 1)
        m = getattr(trainer, "metrics", {}) or {}
        postfix = {}
        for k, short in (
            ("metrics/mAP50(B)", "box_mAP50"),
            ("metrics/mAP50(M)", "mask_mAP50"),
        ):
            if k in m:
                postfix[short] = f"{float(m[k]):.3f}"
        if postfix:
            epoch_bar.set_postfix(postfix)
        epoch_bar.refresh()

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    try:
        results = model.train(
            data=str(ds_info["data_yaml"]),
            epochs=epochs_val,
            imgsz=imgsz_val,
            batch=batch_val,
            workers=workers_val,
            amp=amp_val,
            device=device,
            project=str(cfg.dirs.runs_dir),
            name=run_name,
            exist_ok=True,
            verbose=True,
            plots=True,
        )
    finally:
        epoch_bar.close()

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    chosen_pt = best_pt if best_pt.exists() else last_pt

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics_dict: dict[str, Any] = {}
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            try:
                metrics_dict[k] = round(float(v), 4)
            except Exception:
                pass

    return {
        "dataset": ds_info,
        "run_dir": str(save_dir),
        "best_weights": str(chosen_pt),
        "metrics": metrics_dict,
    }


def predict_yolo11_with_uncertainty(
    model_weights: Path,
    image_paths: list[Path],
    cfg: PipelineConfig,
    taxonomy: Taxonomy,
) -> dict[str, list[InstancePred]]:
    """Run trained YOLO11-seg inference and capture raw class-confusion pairs + confidence scores."""
    cfg.configure_ultralytics_env()
    det_cfg = cfg.detector
    device = ("0" if torch.cuda.is_available() else "cpu") if det_cfg.get("device", "auto") == "auto" else det_cfg["device"]
    imgsz = int(det_cfg.get("imgsz", 640))
    raw_conf = float(det_cfg.get("raw_probe_conf", 0.15))
    predict_conf = float(det_cfg.get("predict_conf", 0.25))
    nms_iou = float(det_cfg.get("nms_iou", 0.55))

    al_cfg = cfg.active_learning
    pair_iou = float(al_cfg.get("pair_confusion_iou", 0.65))
    pair_margin = float(al_cfg.get("pair_confusion_margin", 0.20))

    from ultralytics import YOLO

    model = YOLO(str(model_weights))
    preds_by_stem: dict[str, list[InstancePred]] = {}

    try:
        for img_path in tqdm(image_paths, desc="YOLO11 AL Inference", unit="img", mininterval=1.0):
            res_list = model.predict(
                source=str(img_path),
                conf=raw_conf,
                iou=0.60,
                imgsz=imgsz,
                device=device,
                agnostic_nms=False,
                retina_masks=True,
                verbose=False,
            )
            if not res_list or res_list[0].boxes is None or len(res_list[0].boxes) == 0:
                preds_by_stem[img_path.stem] = []
                continue

            r = res_list[0]
            boxes = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            clss = r.boxes.cls.detach().cpu().numpy().astype(int)
            polys = r.masks.xyn if r.masks is not None else [None] * len(boxes)

            raw_insts: list[InstancePred] = []
            for box, conf, cid, poly in zip(boxes, confs, clss, polys):
                poly_norm = (
                    np.asarray(poly, dtype=np.float32)
                    if poly is not None and len(poly) >= 3
                    else None
                )
                raw_insts.append(
                    InstancePred(
                        class_id=int(cid),
                        conf=float(conf),
                        bbox_xyxy=np.asarray(box, dtype=np.float32),
                        polygon=poly_norm,
                        source="yolo11",
                    )
                )

            deduped = annotate_pair_confusion(
                raw_insts,
                nms_iou=nms_iou,
                pair_confusion_iou=pair_iou,
                pair_confusion_margin=pair_margin,
                taxonomy=taxonomy,
            )
            preds_by_stem[img_path.stem] = [inst for inst in deduped if inst.conf >= predict_conf]
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return preds_by_stem
