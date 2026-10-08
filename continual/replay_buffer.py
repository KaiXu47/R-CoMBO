"""
VOC Replay Buffer for Class Incremental Learning.

Strategy:
  - Fixed-capacity buffer (default 50 images total).
  - After each step, the buffer contains representative images from ALL classes
    seen so far, evenly distributed.
  - Selection: for each class, pick images with the LARGEST pixel area of that
    class (most representative).
  - When new classes arrive, per-class budget shrinks proportionally; excess
    images from old classes are dropped.
  - Buffer is built AFTER step N finishes, BEFORE step N+1 sees its data.
"""

import json
import os
import numpy as np
from collections import defaultdict
from PIL import Image


def build_voc_replay_buffer(dataset_dicts, all_class_ids, buffer_capacity, output_path):
    """
    Build a replay buffer from a list of Detectron2-style dataset dicts.

    Args:
        dataset_dicts: list of dicts, each with 'file_name' and 'sem_seg_file_name'
        all_class_ids: list of int class IDs seen so far (including current step)
        buffer_capacity: total number of images to keep in the buffer
        output_path: path to save the buffer JSON

    Returns:
        list of selected dicts with 'file_name' and 'sem_seg_file_name'
    """
    num_classes = len(all_class_ids)
    base_per_class = max(1, buffer_capacity // num_classes)
    remainder = buffer_capacity % num_classes

    # ---- tally per-class pixel area for every candidate image ----
    class_candidates = defaultdict(list)  # class_id -> [(pixel_count, file_name, sem_seg_file_name)]

    for d in dataset_dicts:
        sem_file = d.get("sem_seg_file_name")
        if not sem_file or not os.path.exists(sem_file):
            continue
        gt = np.array(Image.open(sem_file)).astype(int)
        for cls_id in all_class_ids:
            cnt = int((gt == cls_id).sum())
            if cnt > 0:
                class_candidates[cls_id].append((cnt, d["file_name"], sem_file))

    # ---- greedy selection: per class, pick top pixel-count images ----
    # Distribute remainder: first N classes get one extra image
    selected_files = set()
    selected = []
    per_class_selected = {}

    for idx, cls_id in enumerate(all_class_ids):
        budget = base_per_class + (1 if idx < remainder else 0)
        candidates = sorted(class_candidates.get(cls_id, []), key=lambda x: x[0], reverse=True)
        taken = 0
        for cnt, fname, sem_file in candidates:
            if fname in selected_files:
                continue
            selected_files.add(fname)
            selected.append({"file_name": fname, "sem_seg_file_name": sem_file})
            taken += 1
            if taken >= budget:
                break
        per_class_selected[cls_id] = taken

    # ---- persist ----
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    buffer_info = {
        "all_class_ids": all_class_ids,
        "base_per_class": base_per_class,
        "remainder": remainder,
        "total_selected": len(selected),
        "per_class_selected": per_class_selected,
        "images": selected,
    }
    with open(output_path, "w") as f:
        json.dump(buffer_info, f, indent=2)

    print(
        f"[ReplayBuffer] Built buffer: {len(selected)} images, "
        f"{num_classes} classes, {base_per_class}+{remainder} → {output_path}"
    )
    return selected


def load_voc_replay_buffer(buffer_path):
    """Load a replay buffer from JSON.  Returns list of image dicts or []."""
    if not os.path.exists(buffer_path):
        print(f"[ReplayBuffer] No buffer found at {buffer_path}")
        return []
    with open(buffer_path, "r") as f:
        info = json.load(f)
    images = []
    for img in info.get("images", []):
        images.append({
            "file_name": img["file_name"],
            "sem_seg_file_name": img.get("sem_seg_file_name", ""),
            "voc_replay": True,
        })
    print(f"[ReplayBuffer] Loaded {len(images)} images from {buffer_path}")
    return images
