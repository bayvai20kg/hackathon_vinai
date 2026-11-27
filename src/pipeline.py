from __future__ import annotations

import collections
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
from tqdm.auto import tqdm

from .active_learning import ImageALResult, evaluate_image_active_learning
from .bootstrap import ColdStartBootstrapper
from .config import PipelineConfig, Taxonomy, load_pipeline_config, load_taxonomy
from .cvat_io import export_cvat_package, import_cvat_annotations
from .geometry import InstancePred, write_yolo_seg_file
from .segmenter import Segmenter
from .visualize import draw_instances_overlay
from .yolo_trainer import predict_yolo11_with_uncertainty, train_yolo11_seg


def _list_all_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _existing_labeled_stems(cfg: PipelineConfig) -> set[str]:
    stems: set[str] = set()
    for d in (cfg.dirs.auto_labeled_labels, cfg.dirs.human_verified_labels):
        for p in d.glob("*.txt"):
            stems.add(p.stem)
    return stems


def select_diverse_images(
    images_dir: Path,
    count: int,
    exclude_stems: set[str] | None = None,
    seed: int = 42,
) -> list[Path]:
    """Sample images evenly across day/night cameras (prefix = first 3 underscore tokens)."""
    exclude = exclude_stems or set()
    candidates = [p for p in _list_all_images(images_dir) if p.stem not in exclude]
    if count <= 0 or count >= len(candidates):
        return candidates

    by_cam: dict[str, list[Path]] = {}
    for p in candidates:
        parts = p.stem.split("_")
        cam = "_".join(parts[:3]) if len(parts) >= 3 else p.stem
        by_cam.setdefault(cam, []).append(p)

    rng = random.Random(seed)
    cam_keys = sorted(by_cam.keys())
    for k in cam_keys:
        rng.shuffle(by_cam[k])

    selected: list[Path] = []
    idx = 0
    while len(selected) < count:
        added_any = False
        for k in cam_keys:
            if idx < len(by_cam[k]):
                selected.append(by_cam[k][idx])
                added_any = True
                if len(selected) >= count:
                    break
        if not added_any:
            break
        idx += 1

    return sorted(selected, key=lambda p: p.name)


def _score_predictions(
    image_paths: list[Path],
    preds_by_stem: dict[str, list[InstancePred]],
    cfg: PipelineConfig,
    taxonomy: Taxonomy,
) -> list[ImageALResult]:
    results: list[ImageALResult] = []
    for img_path in image_paths:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        stem = img_path.stem
        legacy_txt = cfg.legacy_labels_dir / f"{stem}.txt"
        al_res = evaluate_image_active_learning(
            stem=stem,
            image_path=img_path,
            img_w=w,
            img_h=h,
            instances=preds_by_stem.get(stem, []),
            cfg=cfg,
            taxonomy=taxonomy,
            legacy_txt_path=legacy_txt if legacy_txt.exists() else None,
        )
        results.append(al_res)
    return results


def _write_al_reports(
    al_results: list[ImageALResult],
    taxonomy: Taxonomy,
    report_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    class_counts = collections.Counter()
    reason_counts = collections.Counter()
    auto_count = 0
    human_count = 0

    rows: list[dict[str, Any]] = []
    for res in sorted(al_results, key=lambda r: r.al_score, reverse=True):
        if res.bucket == "auto":
            auto_count += 1
        else:
            human_count += 1
        for inst in res.instances:
            cname = taxonomy.classes[inst.class_id].name if inst.class_id in taxonomy.classes else str(inst.class_id)
            class_counts[cname] += 1
        for reason in res.reasons:
            reason_counts[reason.split("(")[0].split(":")[0]] += 1
        rows.append(res.to_manifest_dict())

    csv_path = report_dir / f"{prefix}_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "stem", "image", "bucket", "al_score", "min_conf", "mean_conf", "num_objects", "reasons"],
        )
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({**row, "rank": idx, "reasons": "; ".join(row["reasons"])})

    json_path = report_dir / f"{prefix}_summary.json"
    summary = {
        "total_images": len(al_results),
        "auto_labeled_easy": auto_count,
        "human_review_hard": human_count,
        "total_instances": sum(class_counts.values()),
        "class_distribution": dict(sorted(class_counts.items())),
        "uncertainty_reason_counts": dict(reason_counts.most_common()),
        "images": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


def cmd_warmup_export(
    project_root: Path,
    num_images: int = 200,
    batch_name: str = "warmup_batch",
    force_sam3: bool = False,
    force_fallback: bool = False,
    config_path: Path | None = None,
    images_dir: Path | None = None,
    labels_dir: Path | str | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Step 1: Sample diverse images, generate bootstrap SAM pre-labels, and export CVAT ZIP."""
    cfg = load_pipeline_config(
        project_root,
        config_path=config_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        work_dir=work_dir,
    )
    taxonomy = load_taxonomy(project_root / "configs" / "classes_11.yaml")

    selected = select_diverse_images(cfg.images_dir, num_images, exclude_stems=_existing_labeled_stems(cfg))
    print(f"[Warmup-Export] Selected {len(selected)} diverse images across cameras.")

    bootstrapper = ColdStartBootstrapper(cfg, taxonomy)
    preds = bootstrapper.predict_images(selected, segmenter=None)
    bootstrapper.close()

    segmenter = Segmenter(cfg, force_sam3=force_sam3, force_fallback=force_fallback)
    seg_backend = segmenter.backend_name
    for img_path in tqdm(selected, desc=f"{seg_backend} Segment", unit="img", mininterval=1.0):
        if preds.get(img_path.stem):
            segmenter.refine_instances(img_path, preds[img_path.stem])
    segmenter.close()

    al_results = _score_predictions(selected, preds, cfg, taxonomy)
    al_records = {r.stem: r.to_manifest_dict() for r in al_results}
    pkg = export_cvat_package(
        image_paths=selected,
        instances_by_stem=preds,
        al_records=al_records,
        out_dir=cfg.dirs.cvat_export_dir,
        taxonomy=taxonomy,
        package_name=batch_name,
    )
    reports = _write_al_reports(al_results, taxonomy, cfg.dirs.reports_dir, batch_name)

    return {
        "selected_images": len(selected),
        "segmenter_backend": seg_backend,
        "cvat_export": {k: str(v) for k, v in pkg.items()},
        "reports": {k: str(v) for k, v in reports.items()},
    }


def cmd_import_cvat(
    project_root: Path,
    import_path: Path,
    config_path: Path | None = None,
    images_dir: Path | None = None,
    labels_dir: Path | str | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Import human-corrected annotations from CVAT ZIP/JSON/folder into human_verified/."""
    cfg = load_pipeline_config(
        project_root,
        config_path=config_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        work_dir=work_dir,
    )
    taxonomy = load_taxonomy(project_root / "configs" / "classes_11.yaml")
    return import_cvat_annotations(import_path, cfg, taxonomy)


def cmd_train(
    project_root: Path,
    run_name: str = "round_01",
    weights_init: str | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    config_path: Path | None = None,
    images_dir: Path | None = None,
    labels_dir: Path | str | None = None,
    work_dir: Path | None = None,
    data_yaml: Path | None = None,
) -> dict[str, Any]:
    """Step 1/5: Train or retrain YOLO11-seg on merged human_verified + auto_labeled data or custom data.yaml."""
    cfg = load_pipeline_config(
        project_root,
        config_path=config_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        work_dir=work_dir,
    )
    taxonomy = load_taxonomy(project_root / "configs" / "classes_11.yaml")
    return train_yolo11_seg(
        cfg=cfg,
        taxonomy=taxonomy,
        run_name=run_name,
        weights_init=weights_init,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        data_yaml=data_yaml,
    )


def cmd_al_step(
    project_root: Path,
    model_weights: Path,
    batch_name: str = "al_round_01",
    max_unlabeled: int = 0,
    max_cvat_export: int | None = None,
    force_sam3: bool = False,
    force_fallback: bool = False,
    config_path: Path | None = None,
    images_dir: Path | None = None,
    labels_dir: Path | str | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Steps 2-4: YOLO11 inference -> uncertainty routing -> SAM3/2.1 auto-label (>0.85) + CVAT export (<0.60/uncertain)."""
    cfg = load_pipeline_config(
        project_root,
        config_path=config_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        work_dir=work_dir,
    )
    taxonomy = load_taxonomy(project_root / "configs" / "classes_11.yaml")

    labeled = _existing_labeled_stems(cfg)
    unlabeled = [p for p in _list_all_images(cfg.images_dir) if p.stem not in labeled]
    if max_unlabeled > 0:
        unlabeled = unlabeled[:max_unlabeled]
    if not unlabeled:
        return {"message": "All images are already labeled!", "unlabeled_remaining": 0}

    print(f"[AL-Step] Running YOLO11-seg inference on {len(unlabeled)} unlabeled images...")
    preds = predict_yolo11_with_uncertainty(model_weights, unlabeled, cfg, taxonomy)

    auto_thresh = float(cfg.active_learning.get("auto_label_conf_thresh", 0.85))

    segmenter = Segmenter(cfg, force_sam3=force_sam3, force_fallback=force_fallback)
    seg_backend = segmenter.backend_name
    for img_path in tqdm(unlabeled, desc=f"{seg_backend} Segment", unit="img", mininterval=1.0):
        insts = preds.get(img_path.stem, [])
        if insts:
            segmenter.refine_instances(img_path, insts, keep_existing_poly_if_sam_fails=False)
    segmenter.close()

    # Final AL routing after SAM mask quality verification
    final_al = _score_predictions(unlabeled, preds, cfg, taxonomy)

    auto_saved = 0
    hard_results: list[ImageALResult] = []
    for res in final_al:
        if res.bucket == "auto":
            im = cv2.imread(str(res.image_path))
            ih, iw = im.shape[:2]
            shutil.copy2(res.image_path, cfg.dirs.auto_labeled_images / res.image_path.name)
            write_yolo_seg_file(
                cfg.dirs.auto_labeled_labels / f"{res.stem}.txt",
                res.instances,
                img_w=iw,
                img_h=ih,
                allow_rect_fallback=False,
            )
            auto_saved += 1
        else:
            hard_results.append(res)

    hard_results.sort(key=lambda r: r.al_score, reverse=True)
    cvat_limit = int(max_cvat_export or cfg.active_learning.get("cvat_batch_size", 200))
    hard_batch = hard_results[:cvat_limit]

    cvat_pkg: dict[str, Path] = {}
    if hard_batch:
        cvat_pkg = export_cvat_package(
            image_paths=[r.image_path for r in hard_batch],
            instances_by_stem={r.stem: r.instances for r in hard_batch},
            al_records={r.stem: r.to_manifest_dict() for r in hard_batch},
            out_dir=cfg.dirs.cvat_export_dir,
            taxonomy=taxonomy,
            package_name=batch_name,
        )

    reports = _write_al_reports(final_al, taxonomy, cfg.dirs.reports_dir, batch_name)

    return {
        "unlabeled_evaluated": len(unlabeled),
        "segmenter_backend": seg_backend,
        "auto_labeled_saved": auto_saved,
        "hard_images_total": len(hard_results),
        "cvat_batch_exported": len(hard_batch),
        "auto_conf_threshold": auto_thresh,
        "cvat_export": {k: str(v) for k, v in cvat_pkg.items()},
        "reports": {k: str(v) for k, v in reports.items()},
    }


def cmd_demo_50(
    project_root: Path,
    num_images: int = 50,
    train_epochs: int = 5,
    force_sam3: bool = False,
    force_fallback: bool = False,
    config_path: Path | None = None,
    images_dir: Path | None = None,
    labels_dir: Path | str | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full end-to-end pipeline test on 50 diverse images without modifying production verified labels."""
    cfg = load_pipeline_config(
        project_root,
        config_path=config_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        work_dir=work_dir,
    )
    taxonomy = load_taxonomy(project_root / "configs" / "classes_11.yaml")

    demo_dir = cfg.dirs.work_dir / "demo_50"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    boot_lbl_dir = demo_dir / "bootstrap_labels"
    auto_lbl_dir = demo_dir / "auto_labeled_round1" / "labels"
    auto_img_dir = demo_dir / "auto_labeled_round1" / "images"
    vis_boot_dir = demo_dir / "visuals_bootstrap"
    vis_al_dir = demo_dir / "visuals_yolo11_al"
    for d in (boot_lbl_dir, auto_lbl_dir, auto_img_dir, vis_boot_dir, vis_al_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Select 50 diverse images across day & night cameras
    selected = select_diverse_images(cfg.images_dir, num_images, seed=42)
    print(f"\n[Demo-50] Selected {len(selected)} diverse images across cameras.")

    # 2. Bootstrap Detector (YOLOE / COCO + legacy 4-class fusion + rider rules)
    bootstrapper = ColdStartBootstrapper(cfg, taxonomy)
    det_backend = bootstrapper.backend_name
    boot_preds = bootstrapper.predict_images(selected, segmenter=None)
    bootstrapper.close()

    # 3. SAM3 / SAM2.1 mask generation on bootstrap boxes
    segmenter = Segmenter(cfg, force_sam3=force_sam3, force_fallback=force_fallback)
    seg_backend = segmenter.backend_name
    for img_path in tqdm(selected, desc=f"{seg_backend} Bootstrap Segment", unit="img", mininterval=1.0):
        if boot_preds.get(img_path.stem):
            segmenter.refine_instances(img_path, boot_preds[img_path.stem], keep_existing_poly_if_sam_fails=False)
    segmenter.close()

    boot_al = _score_predictions(selected, boot_preds, cfg, taxonomy)
    boot_reports = _write_al_reports(boot_al, taxonomy, demo_dir / "reports", "bootstrap_50")

    # Save bootstrap YOLO-seg labels & export CVAT warm-up package
    smoke_train_samples: dict[str, tuple[Path, Path, str]] = {}
    for res in boot_al:
        im = cv2.imread(str(res.image_path))
        ih, iw = im.shape[:2]
        lbl_path = boot_lbl_dir / f"{res.stem}.txt"
        n_written = write_yolo_seg_file(lbl_path, res.instances, img_w=iw, img_h=ih, allow_rect_fallback=False)
        if n_written > 0:
            smoke_train_samples[res.stem] = (res.image_path, lbl_path, "human" if res.bucket == "auto" else "auto")

    for res in boot_al[:12]:
        draw_instances_overlay(res.image_path, res.instances, taxonomy, vis_boot_dir / res.image_path.name, al_result=res)

    boot_hard = [r for r in sorted(boot_al, key=lambda x: x.al_score, reverse=True) if r.bucket == "human"]
    boot_cvat_pkg = export_cvat_package(
        image_paths=[r.image_path for r in (boot_hard or boot_al)],
        instances_by_stem={r.stem: r.instances for r in (boot_hard or boot_al)},
        al_records={r.stem: r.to_manifest_dict() for r in (boot_hard or boot_al)},
        out_dir=cfg.dirs.cvat_export_dir,
        taxonomy=taxonomy,
        package_name="demo_50_warmup_cvat",
    )

    # 4. Smoke-train YOLO11n-seg on the 50 SAM-refined bootstrap labels
    print(f"\n[Demo-50] Training YOLO11-seg smoke test ({train_epochs} epochs on {len(smoke_train_samples)} images)...")
    train_res = train_yolo11_seg(
        cfg=cfg,
        taxonomy=taxonomy,
        run_name="demo_50_yolo11_seg",
        epochs=train_epochs,
        custom_samples=smoke_train_samples,
        custom_ds_dir=demo_dir / "yolo_dataset",
    )
    best_weights = Path(train_res["best_weights"])

    # 5. Active Learning Inference -> SAM3/2.1 Auto-labeling (>0.85) + CVAT Export (<0.60/uncertain)
    print("\n[Demo-50] Running trained YOLO11-seg inference + Active Learning + SAM refinement...")
    al_preds = predict_yolo11_with_uncertainty(best_weights, selected, cfg, taxonomy)

    segmenter = Segmenter(cfg, force_sam3=force_sam3, force_fallback=force_fallback)
    seg_backend_al = segmenter.backend_name
    for img_path in tqdm(selected, desc=f"{seg_backend_al} AL Segment", unit="img", mininterval=1.0):
        insts = al_preds.get(img_path.stem, [])
        if insts:
            segmenter.refine_instances(img_path, insts, keep_existing_poly_if_sam_fails=False)
    segmenter.close()

    final_al = _score_predictions(selected, al_preds, cfg, taxonomy)
    al_reports = _write_al_reports(final_al, taxonomy, demo_dir / "reports", "yolo11_al_50")

    auto_saved = 0
    hard_final: list[ImageALResult] = []
    for res in final_al:
        if res.bucket == "auto":
            im = cv2.imread(str(res.image_path))
            ih, iw = im.shape[:2]
            shutil.copy2(res.image_path, auto_img_dir / res.image_path.name)
            write_yolo_seg_file(auto_lbl_dir / f"{res.stem}.txt", res.instances, img_w=iw, img_h=ih)
            auto_saved += 1
        else:
            hard_final.append(res)

    hard_final.sort(key=lambda r: r.al_score, reverse=True)
    al_cvat_pkg: dict[str, Path] = {}
    if hard_final:
        al_cvat_pkg = export_cvat_package(
            image_paths=[r.image_path for r in hard_final],
            instances_by_stem={r.stem: r.instances for r in hard_final},
            al_records={r.stem: r.to_manifest_dict() for r in hard_final},
            out_dir=cfg.dirs.cvat_export_dir,
            taxonomy=taxonomy,
            package_name="demo_50_al_review_cvat",
        )

    # Render overlays for top-6 hardest + up to 6 easiest (or top 12)
    preview_list = [r for r in final_al if r.bucket == "auto"][:6] + hard_final[:6]
    if len(preview_list) < 12:
        seen = {r.stem for r in preview_list}
        for r in final_al:
            if r.stem not in seen:
                preview_list.append(r)
                seen.add(r.stem)
            if len(preview_list) >= 12:
                break
    for res in preview_list:
        draw_instances_overlay(res.image_path, res.instances, taxonomy, vis_al_dir / res.image_path.name, al_result=res)

    boot_summary = json.loads(boot_reports["json"].read_text(encoding="utf-8"))
    al_summary = json.loads(al_reports["json"].read_text(encoding="utf-8"))

    summary = {
        "num_images_tested": len(selected),
        "bootstrap_detector": det_backend,
        "segmenter_backend": seg_backend,
        "bootstrap_phase": {
            "auto_easy_count": boot_summary["auto_labeled_easy"],
            "hard_cvat_count": boot_summary["human_review_hard"],
            "total_instances": boot_summary["total_instances"],
            "class_distribution": boot_summary["class_distribution"],
            "uncertainty_reasons": boot_summary["uncertainty_reason_counts"],
            "cvat_warmup_zip": str(boot_cvat_pkg["coco_zip"]),
            "cvat_warmup_yolo_zip": str(boot_cvat_pkg["yolo_zip"]),
        },
        "yolo11_smoke_train": {
            "train_images": train_res["dataset"]["train_images"],
            "val_images": train_res["dataset"]["val_images"],
            "epochs": train_epochs,
            "best_weights": str(best_weights),
            "metrics": train_res["metrics"],
        },
        "active_learning_phase": {
            "auto_easy_count": auto_saved,
            "hard_cvat_count": len(hard_final),
            "total_instances": al_summary["total_instances"],
            "class_distribution": al_summary["class_distribution"],
            "uncertainty_reasons": al_summary["uncertainty_reason_counts"],
            "cvat_al_coco_zip": str(al_cvat_pkg.get("coco_zip", "")),
            "cvat_al_yolo_zip": str(al_cvat_pkg.get("yolo_zip", "")),
        },
        "artifacts": {
            "demo_dir": str(demo_dir),
            "bootstrap_visuals": str(vis_boot_dir),
            "al_visuals": str(vis_al_dir),
            "bootstrap_report": str(boot_reports["json"]),
            "al_report": str(al_reports["json"]),
        },
    }
    (demo_dir / "demo_50_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
