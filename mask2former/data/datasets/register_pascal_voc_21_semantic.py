"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import os

import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

from . import openseg_classes

PASCAL_VOC_21_CATEGORIES = openseg_classes.get_pascal_21_categories_with_prompt_eng()
PASCAL_VOC_21_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PASCAL_VOC_21_COLORS = [k["color"] for k in PASCAL_VOC_21_CATEGORIES]

MetadataCatalog.get("openvocab_pascal21_sem_seg_train").set(
    stuff_colors=PASCAL_VOC_21_COLORS[:],
)

MetadataCatalog.get("openvocab_pascal21_sem_seg_val").set(
    stuff_colors=PASCAL_VOC_21_COLORS[:],
)


def _get_pascal21_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.
    stuff_ids = [k["id"] for k in PASCAL_VOC_21_CATEGORIES]
    assert len(stuff_ids) == 21, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = PASCAL_VOC_21_CLASS_NAMES[:]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_pascal21(root):
    root = os.path.join(root, "pascal_voc_d2")
    meta = _get_pascal21_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_pascal21", dirname)
        name = f"openvocab_pascal21_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            thing_dataset_id_to_contiguous_id={},  # to make Mask2Former happy
            stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            gt_ext="png",
        )



def load_voc_semantic_txt(txt_file, image_root, semantic_root):
    dataset_dicts = []

    with open(txt_file, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]

    for file_name in file_names:
        image_file = os.path.join(image_root, f"{file_name}.jpg")
        sem_seg_file = os.path.join(semantic_root, f"{file_name}.png")

        dataset_dicts.append({
            "file_name": image_file,
            "sem_seg_file_name": sem_seg_file
        })
    return dataset_dicts

def register_voc_current_semantic(
    name, metadata, image_root, semantic_root, txt_file
):
    """
    Register a "standard" version of VOC semantic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "voc_semantic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        semantic_root (str): directory which contains semantic segmentation annotation images
        txt_file (str): path to the txt file containing image file names
    """
    semantic_name = name
    DatasetCatalog.register(
        semantic_name,
        lambda: load_voc_semantic_txt(
            txt_file, image_root, semantic_root
        ),
    )
    MetadataCatalog.get(semantic_name).set(
        stuff_classes=metadata["stuff_classes"][:],
        thing_dataset_id_to_contiguous_id={},  # to make Mask2Former happy
        stuff_dataset_id_to_contiguous_id=metadata["stuff_dataset_id_to_contiguous_id"],
        image_root=image_root,
        sem_seg_root=semantic_root,
        evaluator_type="sem_seg",
        ignore_label=255,
        gt_ext="png",
    )

def register_current_voc_semantic(predefined_split):
    metadata = _get_pascal21_meta()
    for (
            prefix,
            (image_root, semantic_root, continual_txt),
    ) in predefined_split.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_voc_current_semantic(
            prefix,
            metadata,
            image_root,
            semantic_root,
            continual_txt
        )
        
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal21(_root)
