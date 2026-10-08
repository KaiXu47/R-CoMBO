import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _read_names(path):
    with open(path) as f:
        return [Path(line.strip().split()[0]).stem for line in f if line.strip()]


def _has_any_class(ann_path, target_classes):
    annotation = np.array(Image.open(ann_path))
    return np.isin(annotation, np.array(target_classes, dtype=annotation.dtype)).any()


def _filter_names(names, ann_dir, target_classes, desc):
    selected = []
    for name in tqdm(names, desc=desc):
        ann_path = ann_dir / f"{name}.png"
        if ann_path.exists() and _has_any_class(ann_path, target_classes):
            selected.append(name)
    return selected


def _save_split(names, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    print(f"Saved {len(names)} images to {output_file}")


def _task_ranges(base_cls, inc_cls, total_cls=20):
    num_tasks = 1 + (total_cls - base_cls) // inc_cls
    # VOC21 keeps background as label 0. The protocol names (15-1, 19-1)
    # count foreground classes only, so split selection uses foreground ids 1..20.
    ranges = [(1, base_cls + 1)]
    for idx in range(num_tasks - 1):
        start = base_cls + idx * inc_cls + 1
        end = base_cls + (idx + 1) * inc_cls + 1
        ranges.append((start, end))
    return ranges


def prepare_split(base_cls, inc_cls, total_cls, train_names, val_names, train_ann_dir, val_ann_dir, output_dir):
    ranges = _task_ranges(base_cls, inc_cls, total_cls)
    task_name = f"{base_cls}-{inc_cls}"
    for task_idx, (start, end) in enumerate(ranges, start=1):
        train_classes = list(range(start, end))
        val_classes = list(range(1, end))

        train_selected = _filter_names(
            train_names,
            train_ann_dir,
            train_classes,
            f"{task_name} train step {task_idx}: {train_classes}",
        )
        val_selected = _filter_names(
            val_names,
            val_ann_dir,
            val_classes,
            f"{task_name} val step {task_idx}: 0-{end - 1}",
        )

        _save_split(train_selected, output_dir / f"train_{task_name}_step_{task_idx}.txt")
        _save_split(val_selected, output_dir / f"val_{task_name}_step_{task_idx}.txt")


if __name__ == "__main__":
    dataset_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    voc_root = Path(os.getenv("VOC_ROOT", dataset_root / "VOCdevkit/VOC2012"))
    train_list = Path(os.getenv("VOC_TRAIN_LIST", voc_root / "list/train_aug.txt"))
    val_list = Path(os.getenv("VOC_VAL_LIST", voc_root / "ImageSets/Segmentation/val.txt"))
    converted_root = dataset_root / "pascal_voc_d2" / "annotations_pascal21"
    train_ann_dir = converted_root / "training"
    val_ann_dir = converted_root / "validation"
    output_dir = Path("incremental_split")

    train_names = _read_names(train_list)
    val_names = _read_names(val_list)

    for base_cls, inc_cls in [(15, 1), (15, 5), (19, 1)]:
        prepare_split(
            base_cls=base_cls,
            inc_cls=inc_cls,
            total_cls=20,
            train_names=train_names,
            val_names=val_names,
            train_ann_dir=train_ann_dir,
            val_ann_dir=val_ann_dir,
            output_dir=output_dir,
        )
