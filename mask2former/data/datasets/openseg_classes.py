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

import copy

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]

PASCAL_VOC_21_CATEGORIES = [
    {"color": [0, 0, 0], "id": 0, "isthing": 1, "name": "background"},
    {"color": [128, 0, 0], "id": 1, "isthing": 1, "name": "aeroplane"},
    {"color": [0, 128, 0], "id": 2, "isthing": 1, "name": "bicycle"},
    {"color": [128, 128, 0], "id": 3, "isthing": 1, "name": "bird"},
    {"color": [0, 0, 128], "id": 4, "isthing": 1, "name": "boat"},
    {"color": [128, 0, 128], "id": 5, "isthing": 1, "name": "bottle"},
    {"color": [0, 128, 128], "id": 6, "isthing": 1, "name": "bus"},
    {"color": [128, 128, 128], "id": 7, "isthing": 1, "name": "car"},
    {"color": [64, 0, 0], "id": 8, "isthing": 1, "name": "cat"},
    {"color": [192, 0, 0], "id": 9, "isthing": 1, "name": "chair"},
    {"color": [64, 128, 0], "id": 10, "isthing": 1, "name": "cow"},
    {"color": [192, 128, 0], "id": 11, "isthing": 1, "name": "diningtable"},
    {"color": [64, 0, 128], "id": 12, "isthing": 1, "name": "dog"},
    {"color": [192, 0, 128], "id": 13, "isthing": 1, "name": "horse"},
    {"color": [64, 128, 128], "id": 14, "isthing": 1, "name": "motorbike"},
    {"color": [192, 128, 128], "id": 15, "isthing": 1, "name": "person"},
    {"color": [0, 64, 0], "id": 16, "isthing": 1, "name": "pottedplant"},
    {"color": [128, 64, 0], "id": 17, "isthing": 1, "name": "sheep"},
    {"color": [0, 192, 0], "id": 18, "isthing": 1, "name": "sofa"},
    {"color": [128, 192, 0], "id": 19, "isthing": 1, "name": "train"},
    {"color": [0, 64, 128], "id": 20, "isthing": 1, "name": "tvmonitor"},
]

def get_cityscapes_categories_with_prompt_eng():
    CITYSCAPES_CATEGORIES_ = copy.deepcopy(CITYSCAPES_CATEGORIES)
    cityscapes_id_names = open('./mask2former/data/datasets/cityscapes_with_prompt_eng.txt').read().splitlines()
    cityscapes_idx = 0
    for line in cityscapes_id_names:
        idx, name = line.split(':')
        idx = int(idx)
        if name == "invalid_class_id":
            continue
        #print(CITYSCAPES_CATEGORIES_[cityscapes_idx]["name"], '->', name)
        assert CITYSCAPES_CATEGORIES_[cityscapes_idx]["trainId"] == idx
        CITYSCAPES_CATEGORIES_[cityscapes_idx]["name"] = name
        cityscapes_idx += 1
    return CITYSCAPES_CATEGORIES_

def get_pascal_21_categories_with_prompt_eng():
    PASCAL_VOC_21_CATEGORIES_ = copy.deepcopy(PASCAL_VOC_21_CATEGORIES)
    pascal_21_id_names = open('./mask2former/data/datasets/pascal_voc_21_with_prompt_eng.txt').read().splitlines()
    pas_idx = 0
    for line in pascal_21_id_names:
        idx, name = line.split(':')
        idx = int(idx)
        PASCAL_VOC_21_CATEGORIES_[pas_idx]["name"] = name
        pas_idx += 1
    return PASCAL_VOC_21_CATEGORIES_
