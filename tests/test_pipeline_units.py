from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from src.active_learning import annotate_pair_confusion, evaluate_image_active_learning
from src.bootstrap import _RawDet, _VCLS_BICYCLE, _VCLS_MOTORCYCLE, _VCLS_PERSON, _merge_riders_and_two_wheelers
from src.config import load_pipeline_config, load_taxonomy
from src.cvat_io import export_cvat_package, import_cvat_annotations
from src.geometry import InstancePred, mask_to_polygon, read_yolo_seg_file, write_yolo_seg_file


def test_mask_to_polygon_roundtrip(tmp_path: Path) -> None:
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (40, 50), (160, 150), 1, thickness=-1)
    poly = mask_to_polygon(mask, img_w=300, img_h=200)
    assert poly is not None
    assert poly.shape[1] == 2 and len(poly) >= 4
    assert np.all((poly >= 0.0) & (poly <= 1.0))

    txt = tmp_path / "sample.txt"
    inst = InstancePred(
        class_id=0,
        conf=0.92,
        bbox_xyxy=np.array([40, 50, 160, 150], dtype=np.float32),
        polygon=poly,
    )
    count = write_yolo_seg_file(txt, [inst], img_w=300, img_h=200)
    assert count == 1
    loaded = read_yolo_seg_file(txt, img_w=300, img_h=200)
    assert len(loaded) == 1
    assert loaded[0].class_id == 0
    assert loaded[0].polygon is not None and len(loaded[0].polygon) == len(poly)


def test_rider_vehicle_merge_and_pedestrian() -> None:
    dets = [
        _RawDet(vcls=_VCLS_MOTORCYCLE, conf=0.80, bbox_xyxy=np.array([100, 150, 180, 240], dtype=np.float32), source="t"),
        _RawDet(vcls=_VCLS_PERSON, conf=0.88, bbox_xyxy=np.array([110, 90, 170, 210], dtype=np.float32), source="t"),
        _RawDet(vcls=_VCLS_MOTORCYCLE, conf=0.82, bbox_xyxy=np.array([300, 160, 360, 230], dtype=np.float32), source="t"),
        _RawDet(vcls=_VCLS_BICYCLE, conf=0.78, bbox_xyxy=np.array([420, 170, 490, 240], dtype=np.float32), source="t"),
        _RawDet(vcls=_VCLS_PERSON, conf=0.90, bbox_xyxy=np.array([560, 100, 600, 230], dtype=np.float32), source="t"),
    ]
    merged = _merge_riders_and_two_wheelers(dets)
    classes = sorted(inst.class_id for inst in merged)
    # 1 Motorcyclist (5), 1 parked Motorcycle (4), 1 parked Bicycle (6), 1 Pedestrian (8)
    assert classes == [4, 5, 6, 8]


def test_active_learning_easy_vs_hard_and_pair_confusion(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_pipeline_config(root)
    taxonomy = load_taxonomy(root / "configs" / "classes_11.yaml")

    poly = np.array([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]], dtype=np.float32)

    # Case 1: High confidence > 0.85, valid polygon, no flags -> AUTO
    easy_inst = InstancePred(
        class_id=0,
        conf=0.91,
        bbox_xyxy=np.array([100, 100, 200, 200], dtype=np.float32),
        polygon=poly,
        mask_bbox_iou=0.85,
    )
    res_easy = evaluate_image_active_learning("img_easy", tmp_path / "img_easy.jpg", 1280, 720, [easy_inst], cfg, taxonomy)
    assert res_easy.bucket == "auto"

    # Case 2: Pair confusion between Motorcycle (4) and Motorcyclist (5) -> HUMAN
    raw = [
        InstancePred(class_id=5, conf=0.68, bbox_xyxy=np.array([100, 100, 200, 240], dtype=np.float32), polygon=poly),
        InstancePred(class_id=4, conf=0.59, bbox_xyxy=np.array([102, 105, 198, 238], dtype=np.float32), polygon=poly),
    ]
    deduped = annotate_pair_confusion(raw, nms_iou=0.55, pair_confusion_iou=0.65, pair_confusion_margin=0.20, taxonomy=taxonomy)
    assert len(deduped) == 1
    assert deduped[0].second_best_cls == 4
    res_hard = evaluate_image_active_learning("img_hard", tmp_path / "img_hard.jpg", 1280, 720, deduped, cfg, taxonomy)
    assert res_hard.bucket == "human"
    assert any("pair_confusion:Motorcyclist_vs_Motorcycle" in r for r in res_hard.reasons)


def test_cvat_export_and_import_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_pipeline_config(root)
    taxonomy = load_taxonomy(root / "configs" / "classes_11.yaml")

    # Use one real image from workspace for authentic roundtrip without polluting human_verified permanently
    sample_img = next((root / "images").glob("*.jpg"))
    poly = np.array([[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]], dtype=np.float32)
    inst = InstancePred(class_id=9, conf=0.77, bbox_xyxy=np.array([256, 144, 512, 288], dtype=np.float32), polygon=poly)

    pkg = export_cvat_package(
        image_paths=[sample_img],
        instances_by_stem={sample_img.stem: [inst]},
        al_records={sample_img.stem: {"al_score": 0.5, "min_conf": 0.77, "reasons": ["mid_conf"]}},
        out_dir=tmp_path / "cvat_out",
        taxonomy=taxonomy,
        package_name="unit_pkg",
    )
    assert pkg["coco_zip"].exists()
    assert pkg["yolo_zip"].exists()
    assert pkg["labels_json"].exists()

    # Temporarily redirect human_verified into tmp_path to avoid side effects
    cfg.dirs.human_verified_images = tmp_path / "hv" / "images"
    cfg.dirs.human_verified_labels = tmp_path / "hv" / "labels"
    cfg.dirs.human_verified_images.mkdir(parents=True, exist_ok=True)
    cfg.dirs.human_verified_labels.mkdir(parents=True, exist_ok=True)

    res = import_cvat_annotations(pkg["coco_zip"], cfg, taxonomy)
    assert res["imported_images"] == 1
    assert res["total_instances"] == 1
    imported_lbl = cfg.dirs.human_verified_labels / f"{sample_img.stem}.txt"
    assert imported_lbl.exists()
    loaded = read_yolo_seg_file(imported_lbl, 1280, 720)
    assert len(loaded) == 1 and loaded[0].class_id == 9


def test_custom_dataset_overrides(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    custom_imgs = tmp_path / "other_imgs"
    custom_imgs.mkdir()
    cfg = load_pipeline_config(
        root,
        images_dir=custom_imgs,
        labels_dir="",
        work_dir=tmp_path / "other_work",
    )
    assert cfg.images_dir == custom_imgs.resolve()
    assert not cfg.legacy_labels_dir.exists()
    assert cfg.dirs.work_dir == (tmp_path / "other_work").resolve()

