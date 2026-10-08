#!/usr/bin/env python
import argparse
import copy
import json
from pathlib import Path


CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}


def current_ids(base_cls, inc_cls, task, total_cls=19, accumulated=False):
    if task == 1:
        end = base_cls
        return list(range(0, end))
    start = base_cls + (task - 2) * inc_cls
    end = min(base_cls + (task - 1) * inc_cls, total_cls)
    if accumulated:
        return list(range(0, end))
    return list(range(start, end))


def convert_categories(categories):
    out = []
    for cat in categories:
        if cat["id"] not in CITYSCAPES_ID_TO_TRAIN_ID:
            continue
        new_cat = copy.deepcopy(cat)
        new_cat["id"] = CITYSCAPES_ID_TO_TRAIN_ID[cat["id"]]
        out.append(new_cat)
    return sorted(out, key=lambda x: x["id"])


def filter_panoptic_json(src_json, dst_json, keep_train_ids):
    keep_train_ids = set(keep_train_ids)
    with open(src_json, "r") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data.get("images", [])}
    filtered_images = []
    filtered_annotations = []

    for ann in data["annotations"]:
        new_segments = []
        for seg in ann["segments_info"]:
            train_id = CITYSCAPES_ID_TO_TRAIN_ID.get(seg["category_id"])
            if train_id is None or train_id not in keep_train_ids:
                continue
            new_seg = copy.deepcopy(seg)
            new_seg["category_id"] = train_id
            new_segments.append(new_seg)
        if not new_segments:
            continue
        new_ann = copy.deepcopy(ann)
        new_ann["segments_info"] = new_segments
        filtered_annotations.append(new_ann)
        if ann["image_id"] in images_by_id:
            filtered_images.append(images_by_id[ann["image_id"]])

    out = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": convert_categories(data["categories"]),
    }
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(out, f)
    print(f"{dst_json}: images={len(filtered_images)} annotations={len(filtered_annotations)} keep={sorted(keep_train_ids)}")


def write_semantic_split(panoptic_json, dst_txt):
    with open(panoptic_json, "r") as f:
        data = json.load(f)
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_txt, "w") as f:
        for img in data["images"]:
            # Cityscapes ids contain city/name without extension in the panoptic json.
            f.write(f"{img['id']}\n")
    print(f"{dst_txt}: images={len(data['images'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/cityscapes")
    parser.add_argument("--base-cls", type=int, required=True)
    parser.add_argument("--inc-cls", type=int, required=True)
    parser.add_argument("--total-cls", type=int, default=19)
    parser.add_argument("--out-dir", default="json/cityscapes_pan")
    parser.add_argument("--split-dir", default="incremental_split/cityscapes")
    args = parser.parse_args()

    num_tasks = 1 + (args.total_cls - args.base_cls + args.inc_cls - 1) // args.inc_cls
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    split_dir = Path(args.split_dir)
    protocol = f"{args.base_cls}-{args.inc_cls}"

    for task in range(1, num_tasks + 1):
        train_keep = current_ids(args.base_cls, args.inc_cls, task, args.total_cls, accumulated=False)
        val_keep = current_ids(args.base_cls, args.inc_cls, task, args.total_cls, accumulated=True)
        for split, keep in [("train", train_keep), ("val", val_keep)]:
            src = root / "gtFine" / f"cityscapes_panoptic_{split}.json"
            dst = out_dir / f"{split}_{protocol}_step{task}_pan.json"
            filter_panoptic_json(src, dst, keep)
            write_semantic_split(dst, split_dir / f"{split}_{protocol}_step_{task}.txt")


if __name__ == "__main__":
    main()
