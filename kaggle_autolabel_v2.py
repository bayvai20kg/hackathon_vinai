"""Gán nhãn nháp 11 lớp bằng pipeline V2, ghi thẳng ra thư mục YOLO-seg dùng cho notebook train.
Kèm bước sửa lỗi hệ thống của nhãn (gộp mảnh xe lớn...), dùng được riêng bằng --fix-only.

Prelabel (mặc định có gộp mảnh):
    python kaggle_autolabel_v2.py --images-dir <ảnh> --legacy-dir <nhãn bbox cũ> --out <thư mục ra> --min-conf 0.30

Chỉ sửa một thư mục nhãn YOLO-seg có sẵn (không chạy model):
    python kaggle_autolabel_v2.py --fix-only --images-dir <ảnh> --labels-dir <nhãn> --out <thư mục ra>

Đầu ra:
    <out>/images/*.jpg   (copy ảnh gốc)
    <out>/labels/*.txt   (polygon YOLO-seg, 11 lớp; ảnh không có đối tượng -> file rỗng)
    <out>/preview/*.jpg  (vài ảnh vẽ mask để kiểm tra nhanh; chỉ khi prelabel)
    <out>/autolabel_report.json hoặc <out>/fix_report.json

Bước sửa lỗi (fix):
  1. Xe lớn (Car/Bus/Truck/Other) vắt qua mép ô chia ảnh -> nhiều instance cùng lớp chồng nhau (1 xe buýt = 4 object)
     -> gộp (union mask) các instance cùng lớp xe lớn chồng nhau >= 50% (tính trên instance nhỏ hơn).
  2. Instance xe lớn KHÁC lớp nằm >= 70% trong instance lớn hơn -> bỏ instance nhỏ.
     (Cabin + rơ-moóc nằm cạnh nhau, chồng ít -> giữ cả hai.)
  3. Polygon có đường zig-zag bên trong (lỗ của mask) -> vẽ lại chỉ theo viền ngoài.
  4. Other vehicle quá nhỏ (đa số nhận nhầm) -> bỏ nếu diện tích < 2000 px.

LƯU Ý: đây là NHÃN NHÁP (chưa người duyệt). Dùng được cho vòng train đầu; muốn điểm cao cần sửa trên CVAT.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = ["Car", "Bus", "Truck", "Other vehicle", "Motorcycle", "Motorcyclist",
         "Bicycle", "Cyclist", "Pedestrian", "Sign", "Traffic light"]
BIG = {0, 1, 2, 3}


# ------------------------------------------------------------------------------------------- sửa lỗi nhãn (fix)
class FixArgs:
    merge_overlap = 0.5     # gộp 2 instance cùng lớp xe lớn khi chồng >= tỉ lệ này
    cross_overlap = 0.7     # bỏ instance xe lớn khác lớp khi nằm trong instance lớn hơn >= tỉ lệ này
    min_other_area = 2000   # px ở độ phân giải gốc
    eps = 0.0015            # độ đơn giản hóa polygon (theo chu vi)


def _to_mask(pts_norm, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [(pts_norm * [w, h]).round().astype(np.int32)], 1)
    return m


def _outer_polygon(mask, w, h, eps_ratio):
    """Viền ngoài của thành phần lớn nhất (bỏ lỗ bên trong)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    c = cv2.approxPolyDP(c, eps_ratio * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return np.clip(c / [w, h], 0, 1)


def fix_one(lines, w, h, a=FixArgs, stats: Counter | None = None) -> list[str]:
    """Sửa nhãn YOLO-seg của 1 ảnh (danh sách dòng) -> danh sách dòng mới."""
    stats = stats if stats is not None else Counter()
    inst = []
    for ln in lines:
        v = ln.split()
        if not v:
            continue
        m = _to_mask(np.array(v[1:], float).reshape(-1, 2), w, h)
        if m.sum():
            inst.append([int(v[0]), m])

    merged = True
    while merged:  # 1. gộp mảnh cùng lớp xe lớn
        merged = False
        for i in range(len(inst)):
            for j in range(i + 1, len(inst)):
                (ci, mi), (cj, mj) = inst[i], inst[j]
                if ci != cj or ci not in BIG:
                    continue
                inter = int((mi & mj).sum())
                if inter and inter / min(mi.sum(), mj.sum()) >= a.merge_overlap:
                    inst[i][1] = mi | mj
                    del inst[j]
                    stats["gop_manh_cung_lop"] += 1
                    merged = True
                    break
            if merged:
                break

    keep = [True] * len(inst)  # 2. bỏ mảnh xe lớn khác lớp nằm gần trọn trong xe lớn hơn
    for i, (ci, mi) in enumerate(inst):
        for j, (cj, mj) in enumerate(inst):
            if i == j or not keep[j] or ci not in BIG or cj not in BIG or ci == cj:
                continue
            if mi.sum() <= mj.sum() and (mi & mj).sum() / mi.sum() >= a.cross_overlap:
                keep[i] = False
                stats["bo_manh_khac_lop"] += 1
                break
    inst = [x for x, k in zip(inst, keep) if k]

    out = []
    for c, m in inst:
        if c == 3 and m.sum() < a.min_other_area:  # 4. bỏ Other vehicle quá nhỏ
            stats["bo_other_nho"] += 1
            continue
        poly = _outer_polygon(m, w, h, a.eps)  # 3. chỉ giữ viền ngoài
        if poly is None:
            continue
        out.append(f"{c} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
        stats[NAMES[c]] += 1
    return out


def run_fix_only(images_dir: Path, labels_dir: Path, out: Path) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    stats, before = Counter(), Counter()
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        txt = labels_dir / f"{img.stem}.txt"
        if not txt.exists():
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        lines = txt.read_text().splitlines()
        before.update(NAMES[int(l.split()[0])] for l in lines if l.strip())
        fixed = fix_one(lines, w, h, FixArgs, stats)
        (out / "labels" / txt.name).write_text("\n".join(fixed) + ("\n" if fixed else ""))
        if not (out / "images" / img.name).exists():
            shutil.copy2(img, out / "images" / img.name)
    rep = {"truoc": {n: before[n] for n in NAMES}, "sau": {n: stats[n] for n in NAMES},
           **{k: stats[k] for k in ("gop_manh_cung_lop", "bo_manh_khac_lop", "bo_other_nho")}}
    (out / "fix_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------- prelabel V2
def run_prelabel(a) -> None:
    from src.bootstrap_v2 import ColdStartBootstrapperV2, SegmenterV2
    from src.config import load_pipeline_config, load_taxonomy
    from src.geometry import write_yolo_seg_file
    from src.visualize import draw_instances_overlay

    cfg = load_pipeline_config(ROOT, ROOT / "configs" / "pipeline_v2.yaml")
    if a.images_dir:
        cfg.images_dir = Path(a.images_dir).resolve()
    if a.legacy_dir:
        cfg.legacy_labels_dir = Path(a.legacy_dir).resolve()
    do_fix = not a.no_fix
    print("Gộp mảnh xe lớn:", "BẬT" if do_fix else "TẮT")
    taxonomy = load_taxonomy(ROOT / "configs" / "classes_11.yaml")
    out = Path(a.out)
    for sub in ("images", "labels", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in cfg.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    todo = [p for p in imgs if not (out / "labels" / f"{p.stem}.txt").exists()]  # chạy tiếp được nếu bị ngắt
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(imgs)} ảnh, cần gán {len(todo)}")

    boot = ColdStartBootstrapperV2(cfg, taxonomy)
    seg = SegmenterV2(cfg)
    counts: Counter = Counter()
    n_prev = len(list((out / "preview").glob("*.jpg")))
    t0 = time.time()
    for i in range(0, len(todo), a.chunk):
        batch = todo[i : i + a.chunk]
        preds = boot.predict_images(batch, segmenter=seg)
        for p in batch:
            insts = [x for x in preds.get(p.stem, []) if x.conf >= a.min_conf and x.polygon is not None]
            h, w = cv2.imread(str(p)).shape[:2]
            shutil.copy2(p, out / "images" / p.name)
            lbl = out / "labels" / f"{p.stem}.txt"
            write_yolo_seg_file(lbl, insts, img_w=w, img_h=h)
            if do_fix:
                fixed = fix_one(lbl.read_text().splitlines(), w, h)
                lbl.write_text("\n".join(fixed) + ("\n" if fixed else ""))
                counts.update(NAMES[int(ln.split()[0])] for ln in fixed)
            else:
                counts.update(NAMES[x.class_id] for x in insts)
            if n_prev < a.preview:
                draw_instances_overlay(p, insts, taxonomy, out / "preview" / p.name)
                n_prev += 1
        done = min(i + a.chunk, len(todo))
        rate = (time.time() - t0) / done
        print(f"{done}/{len(todo)}  {rate:.1f}s/ảnh  còn ~{rate * (len(todo) - done) / 60:.0f} phút", flush=True)

    boot.close()
    seg.close()
    report = {
        "images": len(list((out / "labels").glob("*.txt"))),
        "instances_this_run": dict(sorted(counts.items())),
        "min_conf": a.min_conf,
        "fix": do_fix,
        "detector": boot.backend_name,
        "segmenter": seg.backend_name,
        "note": "nhãn nháp, chưa người duyệt",
    }
    (out / "autolabel_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/kaggle/working/du_lieu")
    ap.add_argument("--images-dir", help="thư mục ảnh (mặc định images/ của repo)")
    ap.add_argument("--legacy-dir", help="prelabel: thư mục nhãn bbox 4 lớp cũ tương ứng (gợi ý cho v2)")
    ap.add_argument("--min-conf", type=float, default=0.30, help="prelabel: bỏ instance có conf thấp hơn")
    ap.add_argument("--limit", type=int, default=0, help="prelabel: 0 = tất cả ảnh")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--preview", type=int, default=24)
    ap.add_argument("--no-fix", action="store_true", help="prelabel: tắt bước gộp mảnh xe lớn")
    ap.add_argument("--fix-only", action="store_true", help="chỉ sửa nhãn có sẵn trong --labels-dir, không chạy model")
    ap.add_argument("--labels-dir", help="--fix-only: thư mục nhãn YOLO-seg cần sửa")
    a = ap.parse_args()

    if a.fix_only:
        if not (a.images_dir and a.labels_dir):
            ap.error("--fix-only cần --images-dir và --labels-dir")
        run_fix_only(Path(a.images_dir), Path(a.labels_dir), Path(a.out))
    else:
        run_prelabel(a)


if __name__ == "__main__":
    main()
