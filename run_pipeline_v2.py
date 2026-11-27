"""Pipeline V2 — cùng lệnh với run_pipeline.py nhưng dùng configs/pipeline_v2.yaml,
bootstrap V2 (model lớn + chia ô + text prompt + luật 11 lớp) và SAM V2 (cắt cột biển).
Kết quả ghi vào pipeline_work_v2/, không đụng pipeline_work/ của V1.

    python run_pipeline_v2.py run-demo-50 --num-images 10 --epochs 3
    python run_pipeline_v2.py warmup-export --num-images 200 --batch-name warmup_v2
"""

from __future__ import annotations

from pathlib import Path

import src.pipeline as P
from src.bootstrap_v2 import ColdStartBootstrapperV2, SegmenterV2
from src.config import load_pipeline_config as _load_v1

V2_CONFIG = Path(__file__).resolve().parent / "configs" / "pipeline_v2.yaml"


def _load_v2(project_root, config_path=None):
    return _load_v1(project_root, config_path or V2_CONFIG)


# Thay các thành phần mà src/pipeline.py dùng (code V1 không bị sửa)
P.load_pipeline_config = _load_v2
P.ColdStartBootstrapper = ColdStartBootstrapperV2
P.Segmenter = SegmenterV2

if __name__ == "__main__":
    from run_pipeline import main

    main()
