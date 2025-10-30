import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

ALPHADENT_CATEGORIES = [
    {"color": [161, 196, 235], "id": 1, "name": "Abrasion"},
    {"color": [212, 0, 0], "id": 2, "name": "Filling"},
    {"color": [46, 139, 87], "id": 3, "name": "Crown"},
    {"color": [218, 165, 32], "id": 4, "name": "Caries Class 1"},
    {"color": [255, 0, 255], "id": 5, "name": "Caries Class 2"},
    {"color": [139, 69, 19], "id": 6, "name": "Caries Class 3"},
    {"color": [255, 140, 0], "id": 7, "name": "Caries Class 4"},
    {"color": [77, 87, 72], "id": 8, "name": "Caries Class 5"},
    {"color": [0, 0, 0], "id": 9, "name": "Caries Class 6"},
]

_PREDEFINED_SPLITS_AlphaDent = {
    "AlphaDent_train": ("images/train", "annotations/train.json"),
    "AlphaDent_valid": ("images/valid", "annotations/valid.json"),
}

def _get_alphadent_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    thing_ids = [k["id"] for k in ALPHADENT_CATEGORIES]
    assert len(thing_ids) == 9, len(thing_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ALPHADENT_CATEGORIES]

    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def register_all_Alphadent(root):
    root = os.path.join(root, "AlphaDent")
    meta = _get_alphadent_meta()
    for name, (image_root, json_file) in _PREDEFINED_SPLITS_AlphaDent.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            name,
            meta,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_Alphadent(_root)