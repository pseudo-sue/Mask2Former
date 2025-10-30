import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager

ALPHADENT_CLASSES = [
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


# ---- (2) 유틸: COCO categories에서 클래스명 추출 ----
def _extract_thing_classes_from_json(json_file: str) -> Optional[List[str]]:
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        cats = data.get("categories", [])
        names = [c["name"] for c in cats if "name" in c]
        return names if len(names) > 0 else None
    except Exception:
        return None

def register_all_Alphadent(root):
    root = os.path.join(root, "AlphaDent")
    meta = _get_AlphaDent_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images_detectron2/train", "annotations_detectron2/train"),
        ("test", "images_detectron2/test", "annotations_detectron2/test"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"AlphaDent_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_Alphadent(_root)
