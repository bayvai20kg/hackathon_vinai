from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import cmd_al_step, cmd_demo_50, cmd_import_cvat, cmd_train, cmd_warmup_export


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=None, help="Optional YAML config path (default: configs/pipeline.yaml)")
    common.add_argument("--images-dir", type=str, default=None, help="Override dataset images directory")
    common.add_argument("--labels-dir", type=str, default=None, help="Override legacy labels directory (pass '' for none)")
    common.add_argument("--work-dir", type=str, default=None, help="Override pipeline output working directory")

    parser = argparse.ArgumentParser(
        description="SAM 3 + YOLO11 + Active Learning 11-Class Pre-Labeling Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_warm = sub.add_parser("warmup-export", parents=[common], help="Sample diverse images, pre-label via bootstrap+SAM, export CVAT ZIP")
    p_warm.add_argument("--num-images", type=int, default=200, help="0 = all images in images-dir")
    p_warm.add_argument("--batch-name", type=str, default="warmup_batch")
    p_warm.add_argument("--force-sam3", action="store_true")
    p_warm.add_argument("--force-fallback", action="store_true")

    p_imp = sub.add_parser("import-cvat", parents=[common], help="Import human-corrected CVAT ZIP/JSON/folder into human_verified/")
    p_imp.add_argument("--source", type=str, required=True, help="Path to CVAT exported ZIP, COCO JSON, or YOLO-seg dir")

    p_train = sub.add_parser("train", parents=[common], help="Train/retrain YOLO11-seg on merged data or custom data.yaml")
    p_train.add_argument("--run-name", type=str, default="round_01")
    p_train.add_argument("--weights", type=str, default=None, help="Initial weights (e.g. yolo11m-seg.pt or previous best.pt)")
    p_train.add_argument("--data", type=str, default=None, help="Optional custom Ultralytics data.yaml to train directly")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch", type=int, default=None)
    p_train.add_argument("--imgsz", type=int, default=None)

    p_al = sub.add_parser("al-step", parents=[common], help="Run Active Learning inference + SAM auto-label (>0.85) + CVAT hard export")
    p_al.add_argument("--weights", type=str, required=True, help="Trained YOLO11-seg best.pt path")
    p_al.add_argument("--batch-name", type=str, default="al_round_01")
    p_al.add_argument("--max-unlabeled", type=int, default=0, help="0 = process all remaining unlabeled images")
    p_al.add_argument("--max-cvat-export", type=int, default=None)
    p_al.add_argument("--force-sam3", action="store_true")
    p_al.add_argument("--force-fallback", action="store_true")

    p_demo = sub.add_parser("run-demo-50", parents=[common], help="Run full end-to-end smoke test on N diverse images")
    p_demo.add_argument("--num-images", type=int, default=50)
    p_demo.add_argument("--epochs", type=int, default=5)
    p_demo.add_argument("--force-sam3", action="store_true")
    p_demo.add_argument("--force-fallback", action="store_true")

    return parser


def main() -> None:
    root = Path(__file__).resolve().parent
    args = build_parser().parse_args()
    cfg_path = Path(args.config) if args.config else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    labels_dir = args.labels_dir
    work_dir = Path(args.work_dir) if args.work_dir else None

    if args.command == "warmup-export":
        out = cmd_warmup_export(
            root,
            num_images=args.num_images,
            batch_name=args.batch_name,
            force_sam3=args.force_sam3,
            force_fallback=args.force_fallback,
            config_path=cfg_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            work_dir=work_dir,
        )
    elif args.command == "import-cvat":
        out = cmd_import_cvat(
            root,
            Path(args.source),
            config_path=cfg_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            work_dir=work_dir,
        )
    elif args.command == "train":
        out = cmd_train(
            root,
            run_name=args.run_name,
            weights_init=args.weights,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            config_path=cfg_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            work_dir=work_dir,
            data_yaml=Path(args.data) if args.data else None,
        )
    elif args.command == "al-step":
        out = cmd_al_step(
            root,
            model_weights=Path(args.weights),
            batch_name=args.batch_name,
            max_unlabeled=args.max_unlabeled,
            max_cvat_export=args.max_cvat_export,
            force_sam3=args.force_sam3,
            force_fallback=args.force_fallback,
            config_path=cfg_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            work_dir=work_dir,
        )
    elif args.command == "run-demo-50":
        out = cmd_demo_50(
            root,
            num_images=args.num_images,
            train_epochs=args.epochs,
            force_sam3=args.force_sam3,
            force_fallback=args.force_fallback,
            config_path=cfg_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            work_dir=work_dir,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print("\n=== PIPELINE RESULT ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
