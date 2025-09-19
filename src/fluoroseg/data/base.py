from datetime import datetime
from typing import Any
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
import ijson

from ..utils import coco_utils
import itertools


def call_clsinit(cls):
    cls._clsinit()
    return cls


@call_clsinit
class FluoroSegBase:
    @staticmethod
    def make_dataset_info() -> dict:
        today = datetime.today()
        return dict(
            description="FluoroSeg",
            version="0.0.0",
            year=str(today.year),
            contributor="Benjamin D. Killeen",
            date_created=today.strftime("%Y-%m-%d"),
            url="https://github.com/arcadelab/fluoroseg",
        )

    LICENSES = [
        {
            "url": "https://nmdid.unm.edu/resources/data-use",
            "id": 1,
            "name": "NMDID Data Use Agreement",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 2,
            "name": "Attribution-NonCommercial-ShareAlike License",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 3,
            "name": "Attribution-NonCommercial License",
        },
    ]

    CATEGORIES = [
        dict(supercategory="totalsegmentator", id=1, name="spleen"),
        dict(supercategory="totalsegmentator", id=2, name="kidney_right"),
        dict(supercategory="totalsegmentator", id=3, name="kidney_left"),
        dict(supercategory="totalsegmentator", id=4, name="gallbladder"),
        dict(supercategory="totalsegmentator", id=5, name="liver"),
        dict(supercategory="totalsegmentator", id=6, name="stomach"),
        dict(supercategory="totalsegmentator", id=7, name="pancreas"),
        dict(supercategory="totalsegmentator", id=8, name="adrenal_gland_right"),
        dict(supercategory="totalsegmentator", id=9, name="adrenal_gland_left"),
        dict(supercategory="totalsegmentator", id=10, name="lung_upper_lobe_left"),
        dict(supercategory="totalsegmentator", id=11, name="lung_lower_lobe_left"),
        dict(supercategory="totalsegmentator", id=12, name="lung_upper_lobe_right"),
        dict(supercategory="totalsegmentator", id=13, name="lung_middle_lobe_right"),
        dict(supercategory="totalsegmentator", id=14, name="lung_lower_lobe_right"),
        dict(supercategory="totalsegmentator", id=15, name="esophagus"),
        dict(supercategory="totalsegmentator", id=16, name="trachea"),
        dict(supercategory="totalsegmentator", id=17, name="thyroid_gland"),
        dict(supercategory="totalsegmentator", id=18, name="small_bowel"),
        dict(supercategory="totalsegmentator", id=19, name="duodenum"),
        dict(supercategory="totalsegmentator", id=20, name="colon"),
        dict(supercategory="totalsegmentator", id=21, name="urinary_bladder"),
        dict(supercategory="totalsegmentator", id=22, name="prostate"),
        dict(supercategory="totalsegmentator", id=23, name="kidney_cyst_left"),
        dict(supercategory="totalsegmentator", id=24, name="kidney_cyst_right"),
        dict(supercategory="totalsegmentator_bone", id=25, name="sacrum"),
        dict(supercategory="totalsegmentator_bone", id=26, name="vertebrae_S1"),
        dict(supercategory="totalsegmentator_bone", id=27, name="vertebrae_L5"),
        dict(supercategory="totalsegmentator_bone", id=28, name="vertebrae_L4"),
        dict(supercategory="totalsegmentator_bone", id=29, name="vertebrae_L3"),
        dict(supercategory="totalsegmentator_bone", id=30, name="vertebrae_L2"),
        dict(supercategory="totalsegmentator_bone", id=31, name="vertebrae_L1"),
        dict(supercategory="totalsegmentator_bone", id=32, name="vertebrae_T12"),
        dict(supercategory="totalsegmentator_bone", id=33, name="vertebrae_T11"),
        dict(supercategory="totalsegmentator_bone", id=34, name="vertebrae_T10"),
        dict(supercategory="totalsegmentator_bone", id=35, name="vertebrae_T9"),
        dict(supercategory="totalsegmentator_bone", id=36, name="vertebrae_T8"),
        dict(supercategory="totalsegmentator_bone", id=37, name="vertebrae_T7"),
        dict(supercategory="totalsegmentator_bone", id=38, name="vertebrae_T6"),
        dict(supercategory="totalsegmentator_bone", id=39, name="vertebrae_T5"),
        dict(supercategory="totalsegmentator_bone", id=40, name="vertebrae_T4"),
        dict(supercategory="totalsegmentator_bone", id=41, name="vertebrae_T3"),
        dict(supercategory="totalsegmentator_bone", id=42, name="vertebrae_T2"),
        dict(supercategory="totalsegmentator_bone", id=43, name="vertebrae_T1"),
        dict(supercategory="totalsegmentator_bone", id=44, name="vertebrae_C7"),
        dict(supercategory="totalsegmentator_bone", id=45, name="vertebrae_C6"),
        dict(supercategory="totalsegmentator_bone", id=46, name="vertebrae_C5"),
        dict(supercategory="totalsegmentator_bone", id=47, name="vertebrae_C4"),
        dict(supercategory="totalsegmentator_bone", id=48, name="vertebrae_C3"),
        dict(supercategory="totalsegmentator_bone", id=49, name="vertebrae_C2"),
        dict(supercategory="totalsegmentator_bone", id=50, name="vertebrae_C1"),
        dict(supercategory="totalsegmentator", id=51, name="heart"),
        dict(supercategory="totalsegmentator", id=52, name="aorta"),
        dict(supercategory="totalsegmentator", id=53, name="pulmonary_vein"),
        dict(supercategory="totalsegmentator", id=54, name="brachiocephalic_trunk"),
        dict(supercategory="totalsegmentator", id=55, name="subclavian_artery_right"),
        dict(supercategory="totalsegmentator", id=56, name="subclavian_artery_left"),
        dict(
            supercategory="totalsegmentator", id=57, name="common_carotid_artery_right"
        ),
        dict(
            supercategory="totalsegmentator", id=58, name="common_carotid_artery_left"
        ),
        dict(supercategory="totalsegmentator", id=59, name="brachiocephalic_vein_left"),
        dict(
            supercategory="totalsegmentator", id=60, name="brachiocephalic_vein_right"
        ),
        dict(supercategory="totalsegmentator", id=61, name="atrial_appendage_left"),
        dict(supercategory="totalsegmentator", id=62, name="superior_vena_cava"),
        dict(supercategory="totalsegmentator", id=63, name="inferior_vena_cava"),
        dict(
            supercategory="totalsegmentator", id=64, name="portal_vein_and_splenic_vein"
        ),
        dict(supercategory="totalsegmentator", id=65, name="iliac_artery_left"),
        dict(supercategory="totalsegmentator", id=66, name="iliac_artery_right"),
        dict(supercategory="totalsegmentator", id=67, name="iliac_vena_left"),
        dict(supercategory="totalsegmentator", id=68, name="iliac_vena_right"),
        dict(supercategory="totalsegmentator_bone", id=69, name="humerus_left"),
        dict(supercategory="totalsegmentator_bone", id=70, name="humerus_right"),
        dict(supercategory="totalsegmentator_bone", id=71, name="scapula_left"),
        dict(supercategory="totalsegmentator_bone", id=72, name="scapula_right"),
        dict(supercategory="totalsegmentator_bone", id=73, name="clavicula_left"),
        dict(supercategory="totalsegmentator_bone", id=74, name="clavicula_right"),
        dict(supercategory="totalsegmentator_bone", id=75, name="femur_left"),
        dict(supercategory="totalsegmentator_bone", id=76, name="femur_right"),
        dict(supercategory="totalsegmentator_bone", id=77, name="hip_left"),
        dict(supercategory="totalsegmentator_bone", id=78, name="hip_right"),
        dict(supercategory="totalsegmentator", id=79, name="spinal_cord"),
        dict(supercategory="totalsegmentator", id=80, name="gluteus_maximus_left"),
        dict(supercategory="totalsegmentator", id=81, name="gluteus_maximus_right"),
        dict(supercategory="totalsegmentator", id=82, name="gluteus_medius_left"),
        dict(supercategory="totalsegmentator", id=83, name="gluteus_medius_right"),
        dict(supercategory="totalsegmentator", id=84, name="gluteus_minimus_left"),
        dict(supercategory="totalsegmentator", id=85, name="gluteus_minimus_right"),
        dict(supercategory="totalsegmentator", id=86, name="autochthon_left"),
        dict(supercategory="totalsegmentator", id=87, name="autochthon_right"),
        dict(supercategory="totalsegmentator", id=88, name="iliopsoas_left"),
        dict(supercategory="totalsegmentator", id=89, name="iliopsoas_right"),
        dict(supercategory="totalsegmentator", id=90, name="brain"),
        dict(supercategory="totalsegmentator_bone", id=91, name="skull"),
        dict(supercategory="totalsegmentator_bone", id=92, name="rib_right_4"),
        dict(supercategory="totalsegmentator_bone", id=93, name="rib_right_3"),
        dict(supercategory="totalsegmentator_bone", id=94, name="rib_left_1"),
        dict(supercategory="totalsegmentator_bone", id=95, name="rib_left_2"),
        dict(supercategory="totalsegmentator_bone", id=96, name="rib_left_3"),
        dict(supercategory="totalsegmentator_bone", id=97, name="rib_left_4"),
        dict(supercategory="totalsegmentator_bone", id=98, name="rib_left_5"),
        dict(supercategory="totalsegmentator_bone", id=99, name="rib_left_6"),
        dict(supercategory="totalsegmentator_bone", id=100, name="rib_left_7"),
        dict(supercategory="totalsegmentator_bone", id=101, name="rib_left_8"),
        dict(supercategory="totalsegmentator_bone", id=102, name="rib_left_9"),
        dict(supercategory="totalsegmentator_bone", id=103, name="rib_left_10"),
        dict(supercategory="totalsegmentator_bone", id=104, name="rib_left_11"),
        dict(supercategory="totalsegmentator_bone", id=105, name="rib_left_12"),
        dict(supercategory="totalsegmentator_bone", id=106, name="rib_right_1"),
        dict(supercategory="totalsegmentator_bone", id=107, name="rib_right_2"),
        dict(supercategory="totalsegmentator_bone", id=108, name="rib_right_5"),
        dict(supercategory="totalsegmentator_bone", id=109, name="rib_right_6"),
        dict(supercategory="totalsegmentator_bone", id=110, name="rib_right_7"),
        dict(supercategory="totalsegmentator_bone", id=111, name="rib_right_8"),
        dict(supercategory="totalsegmentator_bone", id=112, name="rib_right_9"),
        dict(supercategory="totalsegmentator_bone", id=113, name="rib_right_10"),
        dict(supercategory="totalsegmentator_bone", id=114, name="rib_right_11"),
        dict(supercategory="totalsegmentator_bone", id=115, name="rib_right_12"),
        dict(supercategory="totalsegmentator_bone", id=116, name="sternum"),
        dict(supercategory="totalsegmentator_bone", id=117, name="costal_cartilages"),
        dict(supercategory="totalsegmentator_bone", id=301, name="patella"),
        dict(supercategory="totalsegmentator_bone", id=302, name="tibia"),
        dict(supercategory="totalsegmentator_bone", id=303, name="fibula"),
        dict(supercategory="totalsegmentator_bone", id=304, name="tarsal"),
        dict(supercategory="totalsegmentator_bone", id=305, name="metatarsal"),
        dict(supercategory="totalsegmentator_bone", id=306, name="phalanges_feet"),
        dict(supercategory="totalsegmentator_bone", id=307, name="ulna"),
        dict(supercategory="totalsegmentator_bone", id=308, name="radius"),
        dict(supercategory="totalsegmentator_bone", id=309, name="carpal"),
        dict(supercategory="totalsegmentator_bone", id=310, name="metacarpal"),
        dict(supercategory="totalsegmentator_bone", id=311, name="phalanges_hand"),
        # dict(supercategory="totalsegmentator", id=500, name="body"),
        dict(supercategory="tool", id=1000, name="tool"),
    ]

    CATEGORY_NAMES = [cat["name"] for cat in CATEGORIES]

    CUTOFF_CATEGORIES = [
        dict(
            supercategory=f"{cat['supercategory']}_cutoff_{ax}",
            id=i + 10000,
            name=f"{cat['name']}_cutoff_{ax}",
        )
        for i, (ax, cat) in enumerate(itertools.product(["I", "S"], CATEGORIES))
        if cat["supercategory"] in ["totalsegmentator", "totalsegmentator_bone"]
    ]

    ALL_CATEGORIES = CATEGORIES + CUTOFF_CATEGORIES

    @classmethod
    def _clsinit(cls):
        # assert unique names and ids
        assert len(cls.ALL_CATEGORIES) == len(
            set(cat["name"] for cat in cls.ALL_CATEGORIES)
        )
        assert len(cls.ALL_CATEGORIES) == len(
            set(cat["id"] for cat in cls.ALL_CATEGORIES)
        )

    def is_bone(self, cat_id):
        return (
            self.get_annotation_cat(cat_id)["supercategory"] == "totalsegmentator_bone"
        )

    def is_organ(self, cat_id):
        return self.get_annotation_cat(cat_id)["supercategory"] == "totalsegmentator"

    _idx_from_category_id = dict(
        (cat["id"], idx) for idx, cat in enumerate(ALL_CATEGORIES)
    )
    _annotation_id_from_name = dict((ann["name"], ann["id"]) for ann in ALL_CATEGORIES)
    _annotation_from_id = dict((ann["id"], ann) for ann in ALL_CATEGORIES)

    _super_categories: dict[str, list[int]] = dict()
    for cat in ALL_CATEGORIES:
        _super_categories.setdefault(cat["supercategory"], [])
        _super_categories[cat["supercategory"]].append(cat["id"])

    def get_annotation_catid(self, name: str) -> int | None:
        return self._annotation_id_from_name.get(name)

    def get_annotation_cat(self, catid: int) -> dict:
        return self._annotation_from_id[catid]

    def get_annotation_name(self, catid: int) -> str:
        return self._annotation_from_id[catid]["name"]

    def get_annotation_pretty_name(self, catid: int) -> str:
        return self._annotation_from_id[catid]["name"]

    def make_empty_dataset_annotation(self):
        return dict(
            info=PrephixBase.make_dataset_info(),
            licenses=PrephixBase.LICENSES.copy(),
            categories=PrephixBase.ALL_CATEGORIES.copy(),
            volumes=[],
            images=[],
            annotations=[],
        )

    @staticmethod
    def decode_annotations_efficient(
        size: tuple[int, int], ann_path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w = size
        category_ids = []
        masks = []
        travels = []
        bboxes = []
        f = open(ann_path, "r")
        for anno in ijson.items(f, "item"):
            bbox = anno["bbox"]
            # if bbox[2] < 2 or bbox[3] < 2:
            #     continue
            bboxes.append(bbox)
            category_ids.append(anno["category_id"])

            segm = anno["segmentation"]
            if isinstance(segm, list):
                # Convert polygon
                mask = mask_util.decode(mask_util.frPyObjects(segm, h, w))
                mask = mask[:, :, 0]
            elif isinstance(segm, dict):
                # RLE
                mask = coco_utils.decode_segmentation(segm)
            else:
                raise ValueError(
                    "Cannot transform segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )
            masks.append(mask)

            seg_type = coco_utils.detect_segmentation_type(segm)
            if seg_type == coco_utils.SegmentationType.TRAVEL:
                travels.append(coco_utils.decode_travel(segm))
            else:
                travels.append(np.zeros((h, w), dtype=np.float32))

        if len(category_ids) == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, h, w), dtype=bool),
                np.empty((0, h, w), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )

        return (
            np.array(category_ids),
            np.array(masks),
            np.array(travels),
            np.array(bboxes),
        )

    @staticmethod
    def decode_annotations(
        image_info: dict[str, Any], annos: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode a list of annotations.

        Args:
            annos (list[dict[str, Any]]): List of `n` annotations.

        Returns:
            category_ids: (n,) integer category IDs.
            masks: (n, h, w) bool masks.
            travels: (n, h, w) float32 travel maps.
            bboxes: (n, 4) float32 bounding boxes in (x, y, size_x, size_y) style (COCO).

        """
        h, w = image_info["height"], image_info["width"]
        n = len(annos)

        if n == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0, h, w), dtype=bool),
                np.empty((0, h, w), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )

        category_ids = np.zeros((n,), dtype=np.int32)
        masks = np.zeros((n, h, w), dtype=bool)
        travels = np.zeros((n, h, w), dtype=np.float32)
        bboxes = np.zeros((n, 4), dtype=np.float32)
        for i, anno in enumerate(annos):
            bbox = anno["bbox"]
            # if bbox[2] < 2 or bbox[3] < 2:
            #     continue
            bboxes[i] = bbox
            category_ids[i] = anno["category_id"]

            segm = anno["segmentation"]
            if isinstance(segm, list):
                # Convert polygon
                mask = mask_util.decode(
                    mask_util.frPyObjects(
                        segm, image_info["height"], image_info["width"]
                    )
                )
                mask = mask[:, :, 0]
            elif isinstance(segm, dict):
                # RLE
                mask = coco_utils.decode_segmentation(segm)
            else:
                raise ValueError(
                    "Cannot transform segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )
            masks[i] = mask

            seg_type = coco_utils.detect_segmentation_type(segm)
            if seg_type == coco_utils.SegmentationType.TRAVEL:
                travels[i] = coco_utils.decode_travel(segm)
            else:
                pass

        return (
            category_ids,
            masks,
            travels,
            bboxes,
        )
