#!/usr/bin/env python
"""
Build VOC replay buffer from step-1 training data using GT pixel-area ranking.

Usage:
    python tools/build_voc_replay_buffer.py \
        --config-file configs/voc/semantic-segmentation/15-5.yaml \
        --step1-output ./output/ss/.../step1 \
        [--buffer-capacity 100]

The script:
  1. Registers step-1 datasets.
  2. For each foreground class (1..BASE_CLS-1), ranks step-1 training images
     by GT pixel count (most pixels = most representative).
  3. Selects top-K per class (K = buffer_capacity / num_fg_classes).
  4. Saves buffer to <step1_output>/replay_buffer.json.
No model inference needed — fast and deterministic.
"""

import argparse
import os
import sys
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import default_argument_parser
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from mask2former.data.datasets.register_pascal_voc_21_semantic import (
    register_current_voc_semantic,
)
from continual import add_continual_config
from continual.replay_buffer import build_voc_replay_buffer


def _resolve_voc_incremental_split(filename):
    project_root = Path(__file__).resolve().parent.parent
    candidates = [Path(filename), project_root / "incremental_split" / filename]
    split_root = os.getenv("VOC_INCREMENTAL_SPLIT_ROOT")
    if split_root:
        candidates.append(Path(split_root) / filename)
    dataset_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    candidates.extend([
        dataset_root / "pascal_voc_d2" / "incremental_split" / filename,
        dataset_root / "voc" / "incremental_split" / filename,
    ])
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError(f"Cannot find VOC incremental split {filename}")


def main():
    parser = argparse.ArgumentParser(description="Build VOC replay buffer")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--step1-output", required=True)
    parser.add_argument("--buffer-capacity", type=int, default=100)
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    # ---- config ----
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_continual_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # ---- register step-1 datasets ----
    voc_root = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "pascal_voc_d2")
    voc_split_name = f"{cfg.CONT.BASE_CLS - 1}-{cfg.CONT.INC_CLS}"
    register_current_voc_semantic({
        "current_pascal21_sem_seg_train1": (
            os.path.join(voc_root, "images/training"),
            os.path.join(voc_root, "annotations_pascal21/training"),
            _resolve_voc_incremental_split(f"train_{voc_split_name}_step_1.txt"),
        ),
    })

    # ---- collect step-1 images from complete training set ----
    complete_data = DatasetCatalog.get("openvocab_pascal21_sem_seg_train")
    step1_files = set(
        img["file_name"] for img in DatasetCatalog.get("current_pascal21_sem_seg_train1")
    )
    step1_data = [d for d in complete_data if d["file_name"] in step1_files]
    print(f"[ReplayBuilder] Step-1 has {len(step1_data)} training images")

    # ---- foreground classes only (exclude background = 0) ----
    num_classes = cfg.CONT.BASE_CLS  # 16 for 15-5, 20 for 19-1
    all_class_ids = list(range(1, num_classes))

    # ---- build buffer ----
    output_path = os.path.join(args.step1_output, "replay_buffer.json")
    build_voc_replay_buffer(step1_data, all_class_ids, args.buffer_capacity, output_path)


if __name__ == "__main__":
    main()
