from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClassDef:
    """Single class definition in the fixed 11-class segmentation taxonomy."""

    id: int
    name: str
    name_vi: str
    color: str
    description_vi: str
    text_prompts: list[str]
    rule_notes: str


@dataclass
class Taxonomy:
    """Complete 11-class schema, confusion pairs, and legacy 4-class priors."""

    num_classes: int
    classes: dict[int, ClassDef]
    legacy_4class_mapping: dict[int, dict[str, Any]]
    confusion_pairs: set[tuple[int, int]]

    @property
    def id_to_name(self) -> dict[int, str]:
        return {cid: c.name for cid, c in sorted(self.classes.items())}

    @property
    def name_to_id(self) -> dict[str, int]:
        return {c.name: cid for cid, c in self.classes.items()}

    def is_confusion_pair(self, cls_a: int, cls_b: int) -> bool:
        pair = (min(cls_a, cls_b), max(cls_a, cls_b))
        return pair in self.confusion_pairs

    def build_cvat_labels_config(self) -> list[dict[str, Any]]:
        """Return CVAT project label JSON schema (pasteable into CVAT 'Raw' label config)."""
        return [
            {
                "name": c.name,
                "color": c.color,
                "type": "polygon",
                "attributes": [],
            }
            for _, c in sorted(self.classes.items())
        ]

    def export_cvat_labels_file(self, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.build_cvat_labels_config(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path


@dataclass
class WorkspaceDirs:
    """Standard directory layout inside pipeline_work/."""

    root: Path
    work_dir: Path
    weights_dir: Path
    ultralytics_config_dir: Path
    auto_labeled_labels: Path
    auto_labeled_images: Path
    human_verified_labels: Path
    human_verified_images: Path
    cvat_export_dir: Path
    cvat_import_dir: Path
    yolo_dataset_dir: Path
    runs_dir: Path
    reports_dir: Path

    def ensure_all(self) -> None:
        for attr in (
            self.work_dir,
            self.weights_dir,
            self.ultralytics_config_dir,
            self.auto_labeled_labels,
            self.auto_labeled_images,
            self.human_verified_labels,
            self.human_verified_images,
            self.cvat_export_dir,
            self.cvat_import_dir,
            self.yolo_dataset_dir,
            self.runs_dir,
            self.reports_dir,
        ):
            attr.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Loaded pipeline configuration with resolved absolute paths."""

    project_root: Path
    images_dir: Path
    legacy_labels_dir: Path
    dirs: WorkspaceDirs
    detector: dict[str, Any] = field(default_factory=dict)
    segmenter: dict[str, Any] = field(default_factory=dict)
    active_learning: dict[str, Any] = field(default_factory=dict)
    dataset_split: dict[str, Any] = field(default_factory=dict)

    def configure_ultralytics_env(self) -> None:
        """Ensure Ultralytics stores config/settings inside workspace rather than restricted AppData."""
        self.dirs.ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(self.dirs.ultralytics_config_dir)
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def load_taxonomy(path: str | Path) -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    classes: dict[int, ClassDef] = {}
    for cid_raw, entry in raw["classes"].items():
        cid = int(cid_raw)
        classes[cid] = ClassDef(
            id=cid,
            name=entry["name"],
            name_vi=entry["name_vi"],
            color=entry["color"],
            description_vi=entry["description_vi"],
            text_prompts=list(entry.get("text_prompts", [])),
            rule_notes=entry.get("rule_notes", ""),
        )
    legacy = {int(k): v for k, v in raw.get("legacy_4class_mapping", {}).items()}
    pairs = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in raw.get("confusion_pairs", [])}
    return Taxonomy(
        num_classes=int(raw["num_classes"]),
        classes=classes,
        legacy_4class_mapping=legacy,
        confusion_pairs=pairs,
    )


def load_pipeline_config(project_root: str | Path, config_path: str | Path | None = None) -> PipelineConfig:
    root = Path(project_root).resolve()
    cfg_file = Path(config_path) if config_path else root / "configs" / "pipeline.yaml"
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    paths = raw["paths"]

    work_dir = (root / paths["work_dir"]).resolve()
    dirs = WorkspaceDirs(
        root=root,
        work_dir=work_dir,
        weights_dir=(root / paths["weights_dir"]).resolve(),
        ultralytics_config_dir=(root / paths["ultralytics_config_dir"]).resolve(),
        auto_labeled_labels=work_dir / "auto_labeled" / "labels",
        auto_labeled_images=work_dir / "auto_labeled" / "images",
        human_verified_labels=work_dir / "human_verified" / "labels",
        human_verified_images=work_dir / "human_verified" / "images",
        cvat_export_dir=work_dir / "cvat_export",
        cvat_import_dir=work_dir / "cvat_import",
        yolo_dataset_dir=work_dir / "dataset_yolo",
        runs_dir=work_dir / "runs",
        reports_dir=work_dir / "reports",
    )
    dirs.ensure_all()

    cfg = PipelineConfig(
        project_root=root,
        images_dir=(root / paths["images_dir"]).resolve(),
        legacy_labels_dir=(root / paths["legacy_labels_dir"]).resolve(),
        dirs=dirs,
        detector=dict(raw["detector"]),
        segmenter=dict(raw["segmenter"]),
        active_learning=dict(raw["active_learning"]),
        dataset_split=dict(raw["dataset_split"]),
    )
    cfg.configure_ultralytics_env()
    return cfg
