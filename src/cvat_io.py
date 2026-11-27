from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .config import PipelineConfig, Taxonomy
from .geometry import (
    InstancePred,
    box_xyxy_to_polygon_rect,
    polygon_area_px,
    polygon_to_bbox_xyxy,
    read_yolo_seg_file,
    write_yolo_seg_file,
)


def _get_image_hw(img_path: Path) -> tuple[int, int]:
    im = cv2.imread(str(img_path))
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return im.shape[0], im.shape[1]


def export_cvat_package(
    image_paths: list[Path],
    instances_by_stem: dict[str, list[InstancePred]],
    al_records: dict[str, dict[str, Any]],
    out_dir: Path,
    taxonomy: Taxonomy,
    package_name: str = "cvat_batch",
) -> dict[str, Path]:
    """Export both CVAT COCO 1.0 ZIP and Ultralytics YOLO-seg ZIP + label schema + AL manifest."""
    pkg_dir = out_dir / package_name
    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    yolo_dir = pkg_dir / "yolo_seg"
    coco_dir = pkg_dir / "coco"
    (yolo_dir / "images").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels").mkdir(parents=True, exist_ok=True)
    (coco_dir / "images").mkdir(parents=True, exist_ok=True)
    (coco_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # 1. CVAT label JSON config
    labels_json_path = taxonomy.export_cvat_labels_file(pkg_dir / "cvat_labels.json")

    # 2. Build YOLO-seg data.yaml & train.txt + COCO JSON structures
    data_yaml = {
        "path": ".",
        "train": "train.txt",
        "names": taxonomy.id_to_name,
    }
    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    coco_categories = [
        {"id": cid + 1, "name": cdef.name, "supercategory": "traffic"}
        for cid, cdef in sorted(taxonomy.classes.items())
    ]
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    train_lines: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    ann_id = 1
    for img_idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem
        h, w = _get_image_hw(img_path)
        insts = instances_by_stem.get(stem, [])

        # Copy image into both staging dirs
        dst_yolo_img = yolo_dir / "images" / img_path.name
        dst_coco_img = coco_dir / "images" / img_path.name
        shutil.copy2(img_path, dst_yolo_img)
        shutil.copy2(img_path, dst_coco_img)
        train_lines.append(f"images/{img_path.name}")

        # Write YOLO-seg txt (fallback rect polygon if an uncertain box had no mask so human sees it in CVAT)
        write_yolo_seg_file(
            yolo_dir / "labels" / f"{stem}.txt",
            insts,
            img_w=w,
            img_h=h,
            allow_rect_fallback=True,
        )

        # COCO entries
        coco_images.append(
            {
                "id": img_idx,
                "file_name": img_path.name,
                "width": int(w),
                "height": int(h),
            }
        )
        for inst in insts:
            poly = inst.polygon
            if poly is None or len(poly) < 3:
                poly = box_xyxy_to_polygon_rect(inst.bbox_xyxy, w, h)
            px_poly = np.column_stack([poly[:, 0] * w, poly[:, 1] * h])
            seg_flat = [round(float(v), 2) for v in px_poly.reshape(-1)]
            x1, y1, x2, y2 = polygon_to_bbox_xyxy(poly, w, h).tolist()
            area = max(1.0, polygon_area_px(poly, w, h))
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": int(inst.class_id) + 1,
                    "segmentation": [seg_flat],
                    "bbox": [round(x1, 2), round(y1, 2), round(max(1.0, x2 - x1), 2), round(max(1.0, y2 - y1), 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        rec = al_records.get(stem, {})
        manifest_rows.append(
            {
                "rank": img_idx,
                "image": img_path.name,
                "al_score": round(float(rec.get("al_score", 0.0)), 4),
                "min_conf": round(float(rec.get("min_conf", 1.0)), 4),
                "num_objects": len(insts),
                "reasons": "; ".join(rec.get("reasons", [])),
            }
        )

    (yolo_dir / "train.txt").write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    coco_json = {
        "info": {"description": f"CVAT 11-class Pre-label Export ({package_name})"},
        "licenses": [],
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (coco_dir / "annotations" / "instances_default.json").write_text(
        json.dumps(coco_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 3. Write AL priority manifest
    manifest_csv = pkg_dir / "review_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "image", "al_score", "min_conf", "num_objects", "reasons"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 4. Zip both formats cleanly
    yolo_zip = pkg_dir / f"{package_name}_yolo_seg.zip"
    coco_zip = pkg_dir / f"{package_name}_coco.zip"
    _zip_directory(yolo_dir, yolo_zip)
    _zip_directory(coco_dir, coco_zip)

    return {
        "package_dir": pkg_dir,
        "yolo_zip": yolo_zip,
        "coco_zip": coco_zip,
        "labels_json": labels_json_path,
        "manifest_csv": manifest_csv,
    }


def _zip_directory(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(src_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(src_dir))


def _index_workspace_images(images_dir: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}


def import_cvat_annotations(
    import_source: Path,
    cfg: PipelineConfig,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Import human-reviewed annotations from CVAT (COCO JSON/ZIP or YOLO-seg dir/ZIP) into human_verified/."""
    import_source = Path(import_source).resolve()
    if not import_source.exists():
        raise FileNotFoundError(f"CVAT import source not found: {import_source}")

    scratch_unpack = cfg.dirs.work_dir / "_tmp_cvat_unpack"
    if scratch_unpack.exists():
        shutil.rmtree(scratch_unpack)

    root_dir = import_source
    if import_source.is_file() and import_source.suffix.lower() == ".zip":
        scratch_unpack.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(import_source, "r") as zf:
            zf.extractall(scratch_unpack)
        root_dir = scratch_unpack

    src_images_index = _index_workspace_images(cfg.images_dir)
    imported_stems: list[str] = []
    skipped_items: list[str] = []
    total_instances = 0

    try:
        # Case 1: Single COCO .json file or extracted dir containing instances_*.json
        coco_jsons = (
            [root_dir]
            if root_dir.is_file() and root_dir.suffix.lower() == ".json"
            else sorted(root_dir.rglob("*.json"))
        )
        coco_json = next((p for p in coco_jsons if _looks_like_coco_json(p)), None)

        if coco_json is not None:
            imported_stems, total_instances, skipped_items = _import_coco_json(
                coco_json, taxonomy, src_images_index, cfg
            )
        else:
            # Case 2: YOLO-seg .txt files (labels/*.txt or obj_train_data/*.txt)
            txt_files = sorted(
                p for p in root_dir.rglob("*.txt") if p.name.lower() not in {"train.txt", "val.txt", "classes.txt"}
            )
            if not txt_files:
                raise ValueError(f"No COCO JSON or YOLO-seg .txt label files found in {import_source}")
            imported_stems, total_instances, skipped_items = _import_yolo_seg_txts(
                txt_files, taxonomy, src_images_index, cfg
            )
    finally:
        if scratch_unpack.exists():
            shutil.rmtree(scratch_unpack, ignore_errors=True)

    return {
        "imported_images": len(imported_stems),
        "total_instances": total_instances,
        "skipped": skipped_items,
        "human_verified_dir": str(cfg.dirs.human_verified_labels.parent),
    }


def _looks_like_coco_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and "images" in data and "annotations" in data and "categories" in data
    except Exception:
        return False


def _import_coco_json(
    coco_path: Path,
    taxonomy: Taxonomy,
    src_images_index: dict[str, Path],
    cfg: PipelineConfig,
) -> tuple[list[str], int, list[str]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    name_to_id = {name.lower(): cid for name, cid in taxonomy.name_to_id.items()}
    cat_map: dict[int, int] = {}
    for cat in data["categories"]:
        cname = str(cat["name"]).strip().lower()
        if cname in name_to_id:
            cat_map[int(cat["id"])] = name_to_id[cname]

    img_by_id = {int(im["id"]): im for im in data["images"]}
    anns_by_img: dict[int, list[dict[str, Any]]] = {iid: [] for iid in img_by_id}
    for ann in data["annotations"]:
        iid = int(ann["image_id"])
        if iid in anns_by_img:
            anns_by_img[iid].append(ann)

    imported: list[str] = []
    skipped: list[str] = []
    total_instances = 0

    for iid, im_meta in img_by_id.items():
        stem = Path(im_meta["file_name"]).stem
        src_img = src_images_index.get(stem)
        if src_img is None:
            skipped.append(f"missing_source_image:{im_meta['file_name']}")
            continue

        w = float(im_meta.get("width") or 0)
        h = float(im_meta.get("height") or 0)
        if w <= 0 or h <= 0:
            ih, iw = _get_image_hw(src_img)
            w, h = float(iw), float(ih)

        insts: list[InstancePred] = []
        for ann in anns_by_img.get(iid, []):
            cid = cat_map.get(int(ann["category_id"]))
            if cid is None:
                continue
            segs = ann.get("segmentation") or []
            if not isinstance(segs, list) or not segs:
                continue
            # Take largest polygon part if COCO entry has multiple parts
            best_poly = max(segs, key=len)
            if len(best_poly) < 6:
                continue
            pts = np.array(best_poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
            insts.append(
                InstancePred(
                    class_id=cid,
                    conf=1.0,
                    bbox_xyxy=polygon_to_bbox_xyxy(pts, int(w), int(h)),
                    polygon=pts,
                    source="cvat_human",
                )
            )

        shutil.copy2(src_img, cfg.dirs.human_verified_images / src_img.name)
        total_instances += write_yolo_seg_file(
            cfg.dirs.human_verified_labels / f"{stem}.txt",
            insts,
            img_w=int(w),
            img_h=int(h),
            allow_rect_fallback=False,
        )
        imported.append(stem)

    return imported, total_instances, skipped


def _import_yolo_seg_txts(
    txt_files: list[Path],
    taxonomy: Taxonomy,
    src_images_index: dict[str, Path],
    cfg: PipelineConfig,
) -> tuple[list[str], int, list[str]]:
    imported: list[str] = []
    skipped: list[str] = []
    total_instances = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        src_img = src_images_index.get(stem)
        if src_img is None:
            skipped.append(f"missing_source_image:{stem}")
            continue
        ih, iw = _get_image_hw(src_img)
        insts = [inst for inst in read_yolo_seg_file(txt_path, iw, ih) if 0 <= inst.class_id < taxonomy.num_classes]
        shutil.copy2(src_img, cfg.dirs.human_verified_images / src_img.name)
        total_instances += write_yolo_seg_file(
            cfg.dirs.human_verified_labels / f"{stem}.txt",
            insts,
            img_w=iw,
            img_h=ih,
            allow_rect_fallback=False,
        )
        imported.append(stem)

    return imported, total_instances, skipped
