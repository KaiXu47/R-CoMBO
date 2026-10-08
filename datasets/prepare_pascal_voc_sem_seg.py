
"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/NVlabs/ODISE/blob/main/datasets/prepare_pascal_voc_sem_seg.py
"""

import os
from pathlib import Path
import shutil

import numpy as np
import tqdm
from PIL import Image


def convert_pas21(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    # do nothing
    Image.fromarray(img).save(output)

def convert_pas20(input, output):
    img = np.array(Image.open(input))
    img[img == 0] = 255
    img = img - 1
    img[img == 254] = 255
    assert img.dtype == np.uint8
    # do nothing
    Image.fromarray(img).save(output)


def parse_image_name(line):
    first_field = line.strip().split()[0]
    return Path(first_field).stem


if __name__ == "__main__":
    output_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets"))
    dataset_dir = output_root / "pascal_voc_d2"
    voc_root = Path(os.getenv("VOC_ROOT", output_root / "VOCdevkit/VOC2012"))
    train_list = Path(os.getenv("VOC_TRAIN_LIST", voc_root / "list/train_aug.txt"))
    val_list = Path(os.getenv("VOC_VAL_LIST", voc_root / "ImageSets/Segmentation/val.txt"))

    for split in ["training", "validation"]:
        if split == "training":
            img_name_path = train_list
        else:
            img_name_path = val_list
        img_dir = voc_root / "JPEGImages"
        ann_dir = voc_root / "SegmentationClassAug"

        output_img_dir = dataset_dir / "images" / split
        output_ann_dir_21 = dataset_dir / "annotations_pascal21" / split
        output_ann_dir_20 = dataset_dir / "annotations_pascal20" / split

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_ann_dir_21.mkdir(parents=True, exist_ok=True)
        output_ann_dir_20.mkdir(parents=True, exist_ok=True)

        print(f"Preparing {split} from {img_name_path}")
        with open(img_name_path) as f:
            for line in tqdm.tqdm(f.readlines()):
                img_name = parse_image_name(line)
                img_path = img_dir / f"{img_name}.jpg"
                ann_path = ann_dir / f"{img_name}.png"

                # print(f'copy2 {output_img_dir}')
                shutil.copy2(img_path, output_img_dir)
                # print(f"convert {ann_dir} to {output_ann_dir / f'{img_name}.png'}")
                convert_pas21(ann_path, output_ann_dir_21 / f"{img_name}.png")
                convert_pas20(ann_path, output_ann_dir_20 / f"{img_name}.png")
