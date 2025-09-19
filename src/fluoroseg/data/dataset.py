"""A Pytorch dataset for training a segmentation model."""

from __future__ import annotations
from typing import Any
from pathlib import Path
import cv2
import ijson
from tqdm import tqdm
import logging
import numpy as np

from ..utils import json_utils
from ..alb.build_aug import build_augmentation
from .base import FluoroSegBase
from ..utils import coco_utils

log = logging.getLogger(__name__)


class FluoroSegDataset(FluoroSegBase):
    """Coco-like Segmentation dataset for FluoroSeg.

    Expects the following directory structure:

    drr-dataset-name
        drr_dataset_1/
            - 0000000000.tiff
            - 0000000000.json
            - 0000000001.tiff
            - 0000000001.json
            - ...
        drr_dataset_2/
        ...
        drr_dataset_1.json
        drr_dataset_2.json
        ...
        train.txt
        val.txt

    The train.txt and val.txt files contain the names of the case annotation files
    (e.g. drr_dataset_1.json) to use for training and validation,
    respectively. There should be no overlap between the two files.

    """

    def process_image_paths(self):
        """Process the image paths for the dataset."""
        if self.train:
            split_path = self.data_dir / "train.txt"
        else:
            split_path = self.data_dir / "val.txt"

        if split_path.exists():
            with open(split_path, "r") as f:
                case_ann_paths = [
                    self.data_dir / line.strip() for line in f.readlines()
                ]
        else:
            case_ann_paths = sorted(list(self.data_dir.glob("*.json")))
            split = int(len(case_ann_paths) * self.split)
            case_ann_paths = (
                case_ann_paths[:split] if self.train else case_ann_paths[split:]
            )

        image_paths: list[str] = []
        focal_lengths: list[float] = []
        pixel_sizes: list[list[float]] = []
        num_removed = 0
        bad_nccs = []

        for case_path in tqdm(case_ann_paths):
            if not (case_path.parent / case_path.stem).exists():
                # Image dir does not exist, skip it
                continue

            ann = json_utils.load_json(str(case_path))
            for i, image_dict in enumerate(ann["images"]):
                image_path = str(
                    self.data_dir / case_path.stem / image_dict["file_name"]
                )

                # SUPER SLOW, only use when dataset hasn't been checked.
                # image_ann_path = Path(image_path).with_suffix(".json")
                # if not image_ann_path.exists() or not Path(image_path).exists():
                #     continue

                grad_ncc_key = "grad_ncc"
                if (
                    grad_ncc_key in image_dict
                    and image_dict[grad_ncc_key] < self.exclude_grad_ncc
                ):
                    num_removed += 1
                    bad_nccs.append(image_dict[grad_ncc_key])
                    continue

                if grad_ncc_key in image_dict:
                    exclude_by_id = [
                        {
                            "patient_id": "=20231213",
                            "study_id": "=1.2.826.0.1.3680043.2.940.2022.14.20231213.81858.506",
                        },
                        {
                            "patient_id": "=20231110",
                            "study_id": "=1.2.276.0.20.1.2.33.1017633001505.8760.1699819294.755195.0",
                            "series_id": ">1.2.826.0.1.3680043.2.940.2022.14.20231112.172839.173",
                        },
                    ]

                    for exclude in exclude_by_id:
                        # for each term in exclude dict
                        matched = True
                        for key, value in exclude.items():
                            match_type = value[0]
                            value = value[1:]
                            if match_type == "=":
                                if key in image_dict and image_dict[key] != value:
                                    matched = False
                                    break
                            elif match_type == ">":
                                if key in image_dict and image_dict[key] <= value:
                                    matched = False
                                    break
                            elif match_type == "<":
                                if key in image_dict and image_dict[key] >= value:
                                    matched = False
                                    break
                            else:
                                break
                        if matched:
                            out_str = "Excluded: "
                            for key, value in exclude.items():
                                out_str += f"{key}={image_dict[key]}, "
                            log.info(out_str)
                            continue

                image_paths.append(image_path)
                focal_lengths.append(image_dict["camera"]["focal_length"])
                pixel_sizes.append(image_dict["camera"]["pixel_size"])

                if self.debug and len(image_paths) > 10000:
                    break
            if self.debug and len(image_paths) > 10000:
                break

        if bad_nccs:
            log.info(
                f"Excluded {num_removed} images with grad_ncc < {self.exclude_grad_ncc}"
            )
            bad_nccs = np.array(bad_nccs)
            log.info(
                f"grad_ncc: min={bad_nccs.min()}, max={bad_nccs.max()}, mean={bad_nccs.mean()}, sd={bad_nccs.std()}"
            )

        self.image_paths = np.array(image_paths).astype(np.string_)
        self.focal_lengths = np.array(focal_lengths)
        self.pixel_sizes = np.array(pixel_sizes)
        log.info(f"Loaded {len(self.image_paths)} images from {self.data_dir}")

    # For training on a reduced set of categories, e.g. combining the ribs
    REDUCED_CATEGORIES: list[dict[str, Any]] = [
        dict(supercategory="totalsegmentator", ids=[1], name="spleen"),
        dict(supercategory="totalsegmentator", ids=[2, 3], name="kidney"),
        dict(supercategory="totalsegmentator", ids=[4], name="gallbladder"),
        dict(supercategory="totalsegmentator", ids=[5], name="liver"),
        dict(supercategory="totalsegmentator", ids=[6], name="stomach"),
        dict(supercategory="totalsegmentator", ids=[7], name="pancreas"),
        dict(supercategory="totalsegmentator", ids=[8, 9], name="adrenal_gland"),
        dict(supercategory="totalsegmentator", ids=[10, 11, 12, 13, 14], name="lung"),
        dict(supercategory="totalsegmentator", ids=[15], name="esophagus"),
        dict(supercategory="totalsegmentator", ids=[16], name="trachea"),
        dict(supercategory="totalsegmentator", ids=[17], name="thyroid_gland"),
        dict(supercategory="totalsegmentator", ids=[18], name="small_bowel"),
        dict(supercategory="totalsegmentator", ids=[19], name="duodenum"),
        dict(supercategory="totalsegmentator", ids=[20], name="colon"),
        dict(supercategory="totalsegmentator", ids=[21], name="urinary_bladder"),
        dict(supercategory="totalsegmentator", ids=[22], name="prostate"),
        dict(supercategory="totalsegmentator_bone", ids=[25], name="sacrum"),
        dict(
            supercategory="totalsegmentator_bone",
            ids=[26, 27, 28, 29, 30, 31],
            name="vertebrae_L",
        ),
        dict(
            supercategory="totalsegmentator_bone",
            ids=[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
            name="vertebrae_T",
        ),
        dict(
            supercategory="totalsegmentator_bone",
            ids=[44, 45, 46, 47, 48, 49, 50],
            name="vertebrae_C",
        ),
        dict(supercategory="totalsegmentator", ids=[51], name="heart"),
        dict(supercategory="totalsegmentator", ids=[52], name="aorta"),
        dict(supercategory="totalsegmentator", ids=[53], name="pulmonary_vein"),
        dict(supercategory="totalsegmentator", ids=[54], name="brachiocephalic_trunk"),
        dict(supercategory="totalsegmentator", ids=[55, 56], name="subclavian_artery"),
        dict(
            supercategory="totalsegmentator", ids=[57, 58], name="common_carotid_artery"
        ),
        dict(
            supercategory="totalsegmentator", ids=[59, 60], name="brachiocephalic_vein"
        ),
        dict(supercategory="totalsegmentator", ids=[61], name="atrial_appendage_left"),
        dict(supercategory="totalsegmentator", ids=[62], name="superior_vena_cava"),
        dict(supercategory="totalsegmentator", ids=[63], name="inferior_vena_cava"),
        dict(
            supercategory="totalsegmentator",
            ids=[64],
            name="portal_vein_and_splenic_vein",
        ),
        dict(supercategory="totalsegmentator_bone", ids=[69, 70], name="humerus"),
        dict(supercategory="totalsegmentator_bone", ids=[71, 72], name="scapula"),
        dict(supercategory="totalsegmentator_bone", ids=[73, 74], name="clavicula"),
        dict(supercategory="totalsegmentator_bone", ids=[75, 76], name="femur"),
        dict(supercategory="totalsegmentator_bone", ids=[77, 78], name="pelvis"),
        dict(supercategory="totalsegmentator", ids=[79], name="spinal_cord"),
        dict(
            supercategory="totalsegmentator",
            ids=[80, 81, 82, 83, 84, 85],
            name="gluteus",
        ),
        dict(supercategory="totalsegmentator", ids=[86, 87], name="autochthon"),
        dict(supercategory="totalsegmentator", ids=[88, 89], name="iliopsoas"),
        dict(supercategory="totalsegmentator", ids=[90], name="brain"),
        dict(supercategory="totalsegmentator_bone", ids=[91], name="skull"),
        dict(supercategory="totalsegmentator_bone", ids=[*range(92, 116)], name="ribs"),
        dict(supercategory="totalsegmentator_bone", ids=[116], name="sternum"),
        dict(
            supercategory="totalsegmentator_bone", ids=[117], name="costal_cartilages"
        ),
        dict(supercategory="totalsegmentator_bone", ids=[301], name="patella"),
        dict(supercategory="totalsegmentator_bone", ids=[302], name="tibia"),
        dict(supercategory="totalsegmentator_bone", ids=[303], name="fibula"),
        dict(supercategory="totalsegmentator_bone", ids=[304], name="tarsal"),
        dict(supercategory="totalsegmentator_bone", ids=[305], name="metatarsal"),
        dict(supercategory="totalsegmentator_bone", ids=[306], name="phalanges_feet"),
        dict(supercategory="totalsegmentator_bone", ids=[307], name="ulna"),
        dict(supercategory="totalsegmentator_bone", ids=[308], name="radius"),
        dict(supercategory="totalsegmentator_bone", ids=[309], name="carpal"),
        dict(supercategory="totalsegmentator_bone", ids=[310], name="metacarpal"),
        dict(supercategory="totalsegmentator_bone", ids=[311], name="phalanges_hand"),
        dict(supercategory="tool", ids=[1000], name="tool"),
    ]

    NUM_CLASSES = len(REDUCED_CATEGORIES)
    reduced_category_from_cat_id = dict()
    label_from_category_id = dict()
    for i, cat in enumerate(REDUCED_CATEGORIES):
        for cat_id in cat["ids"]:
            reduced_category_from_cat_id[cat_id] = cat
            label_from_category_id[cat_id] = i

    category_id_from_label = {v: k for k, v in label_from_category_id.items()}
    category_names: list[str] = [cat["name"] for cat in REDUCED_CATEGORIES]

    def decode_segs(self, size: tuple[int, int], ann_path: Path) -> np.ndarray:
        h, w = size
        original_area = h * w
        H, W = self.image_size
        masks = np.zeros((H, W, self.NUM_CLASSES), dtype=bool)
        f = open(ann_path, "r")
        for anno in ijson.items(f, "item"):
            label = self.label_from_category_id.get(anno["category_id"])
            if label is None:
                continue
            segm = anno["segmentation"]
            mask = coco_utils.decode_segmentation(segm)

            # Filter out masks that are too small or too large.
            mask_area = mask.sum()
            if mask_area < 0.001 * original_area or mask_area > 0.9 * original_area:
                continue

            masks[:, :, label] |= cv2.resize(
                mask.astype(np.uint8), (H, W), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        return masks

    bone_cat_indices = [
        i for i, cat in enumerate(REDUCED_CATEGORIES) if "bone" in cat["supercategory"]
    ]
    nonbone_cat_indices = [
        i
        for i, cat in enumerate(REDUCED_CATEGORIES)
        if "bone" not in cat["supercategory"]
    ]

    def __init__(
        self,
        *,
        data_dir: Path,
        train: bool = True,
        image_size: list[int] = [224, 224],
        fliph: bool = False,
        split: float = 0.90,
        debug: bool = False,
        exclude_grad_ncc: float = -999,
    ):
        """Initialize the dataset.

        Args:
            datasets (list[PerphixDataset]): The datasets to use.
            train (bool, optional): Whether this is a training dataset. Defaults to True.
            image_size (int, optional): The size of the images to return. Defaults to 256.
            fliph (bool, optional): Whether to flip the images horizontally. This is not an augmentation
                strategy but rather a correction if the images were not properly flipped. Defaults to False.

        """

        self.data_dir = Path(data_dir).expanduser()
        self.split = split

        self.train = train
        self.image_size: tuple[int, int] = tuple(image_size)
        self.fliph = fliph
        self.debug = debug
        self.exclude_grad_ncc = exclude_grad_ncc

        self.process_image_paths()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Process the raw intensity image before any augmentation.

        Selection:
        1. Select upper/lower bounds randomly.
        2. Apply negative log transformation.

        """

        # Convert to 3-channel.
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image.astype(np.float32)

    def __getitem__(self, index: int):
        """Get the image and masks corresponding to the given index and augment.

        Args:
            index (int): The index into the images.

        Returns:
            image (dict[str, Any]): Dict for the given image, which contains:
                'image' (torch.Tensor): Image as a tensor in 3xHxW format, float32, in [0,1].
                'original_size' (tuple[int, int]): The original size of the image.
                'boxes' (torch.Tensor): Bounding boxes for this image, with shape (N,4) in [0,1] range (in
                    the input frame to the model), in the (x_min, y_min, x_max, y_max) format.
                'masks' (torch.Tensor): Binary masks for the annotations, with shape (N,H,W) where H and W are the
                    original size of the image.
                'descriptions' list[str]: Text descriptions for the boxes, with shape (N,) where N is the
                    number of boxes.

        """
        image_path = Path(self.image_paths[index].decode("utf-8"))
        ann_path = image_path.with_suffix(".json")

        image: np.ndarray | None
        try:
            image = cv2.imread(
                str(image_path), flags=(cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            )
        except Exception as e:
            log.warning(f"Could not load {image_path}.")
            image = None

        if image is None or not ann_path.exists():
            if not ann_path.exists():
                log.warning(f"Annotation {ann_path} does not exist.")

            h, w = self.image_size
            image = np.zeros((3, h, w), dtype=np.float32)
            segs = np.zeros((self.NUM_CLASSES, h, w), dtype=np.float32)
            return dict(
                image=image,
                original_size=(h, w),
                segs=segs,
            )

        # Convert to RGB
        image = self.preprocess(image)

        original_size: tuple[int, int] = image.shape[:2]  # type: ignore
        if self.image_size != original_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

        segs: np.ndarray
        segs = self.decode_segs(original_size, ann_path)

        if self.fliph:
            # Not for augmentation.
            image = image[:, ::-1]
            segs = segs[:, :, ::-1]

        # Augmentation with albumentations
        aug = build_augmentation(self.image_size, train=self.train)
        augmented = aug(image=image, mask=segs)
        image = augmented["image"]
        segs = augmented["mask"]

        if image is None:
            raise RuntimeError

        # Transpose and cast. Should be in [0,1] by now.
        image = image.transpose(2, 0, 1).astype(np.float32)
        segs = segs.transpose(2, 0, 1).astype(np.float32)

        outputs = dict(
            image=image,
            original_size=original_size,
            segs=segs,
        )

        return outputs

    def __len__(self) -> int:
        """Get the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.

        """
        return len(self.image_paths)
