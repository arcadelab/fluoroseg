from enum import Enum, auto
from typing import Type
from strenum import StrEnum


class Total(Enum):
    spleen = 1
    kidney_right = 2
    kidney_left = 3
    gallbladder = 4
    liver = 5
    stomach = 6
    pancreas = 7
    adrenal_gland_right = 8
    adrenal_gland_left = 9
    lung_upper_lobe_left = 10
    lung_lower_lobe_left = 11
    lung_upper_lobe_right = 12
    lung_middle_lobe_right = 13
    lung_lower_lobe_right = 14
    esophagus = 15
    trachea = 16
    thyroid_gland = 17
    small_bowel = 18
    duodenum = 19
    colon = 20
    urinary_bladder = 21
    prostate = 22
    kidney_cyst_left = 23
    kidney_cyst_right = 24
    sacrum = 25
    vertebrae_S1 = 26
    vertebrae_L5 = 27
    vertebrae_L4 = 28
    vertebrae_L3 = 29
    vertebrae_L2 = 30
    vertebrae_L1 = 31
    vertebrae_T12 = 32
    vertebrae_T11 = 33
    vertebrae_T10 = 34
    vertebrae_T9 = 35
    vertebrae_T8 = 36
    vertebrae_T7 = 37
    vertebrae_T6 = 38
    vertebrae_T5 = 39
    vertebrae_T4 = 40
    vertebrae_T3 = 41
    vertebrae_T2 = 42
    vertebrae_T1 = 43
    vertebrae_C7 = 44
    vertebrae_C6 = 45
    vertebrae_C5 = 46
    vertebrae_C4 = 47
    vertebrae_C3 = 48
    vertebrae_C2 = 49
    vertebrae_C1 = 50
    heart = 51
    aorta = 52
    pulmonary_vein = 53
    brachiocephalic_trunk = 54
    subclavian_artery_right = 55
    subclavian_artery_left = 56
    common_carotid_artery_right = 57
    common_carotid_artery_left = 58
    brachiocephalic_vein_left = 59
    brachiocephalic_vein_right = 60
    atrial_appendage_left = 61
    superior_vena_cava = 62
    inferior_vena_cava = 63
    portal_vein_and_splenic_vein = 64
    iliac_artery_left = 65
    iliac_artery_right = 66
    iliac_vena_left = 67
    iliac_vena_right = 68
    humerus_left = 69
    humerus_right = 70
    scapula_left = 71
    scapula_right = 72
    clavicula_left = 73
    clavicula_right = 74
    femur_left = 75
    femur_right = 76
    hip_left = 77
    hip_right = 78
    spinal_cord = 79
    gluteus_maximus_left = 80
    gluteus_maximus_right = 81
    gluteus_medius_left = 82
    gluteus_medius_right = 83
    gluteus_minimus_left = 84
    gluteus_minimus_right = 85
    autochthon_left = 86
    autochthon_right = 87
    iliopsoas_left = 88
    iliopsoas_right = 89
    brain = 90
    skull = 91
    rib_right_4 = 92
    rib_right_3 = 93
    rib_left_1 = 94
    rib_left_2 = 95
    rib_left_3 = 96
    rib_left_4 = 97
    rib_left_5 = 98
    rib_left_6 = 99
    rib_left_7 = 100
    rib_left_8 = 101
    rib_left_9 = 102
    rib_left_10 = 103
    rib_left_11 = 104
    rib_left_12 = 105
    rib_right_1 = 106
    rib_right_2 = 107
    rib_right_5 = 108
    rib_right_6 = 109
    rib_right_7 = 110
    rib_right_8 = 111
    rib_right_9 = 112
    rib_right_10 = 113
    rib_right_11 = 114
    rib_right_12 = 115
    sternum = 116
    costal_cartilages = 117


class Body(Enum):
    trunc = 1
    extremities = 2


class TotalSegmentatorAppendicular(Enum):
    patella = 1
    tibia = 2
    fibula = 3
    tarsal = 4
    metatarsal = 5
    phalanges_feet = 6
    ulna = 7
    radius = 8
    carpal = 9
    metacarpal = 10
    phalanges_hand = 11


# fmt: off
GROUPS = dict(
    pelvis=[77, 78],
    lungs=[10, 11, 12, 13, 14],
    lung_left=[10, 11],
    lung_right=[12, 13, 14],
    ribs=[92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    ribs_right=[106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
    ribs_left=[92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
    cervical_vertebrae=[44, 45, 46, 47, 48, 49, 50],
    thoracic_vertebrae=[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    lumbar_vertebrae=[27, 28, 29, 30, 31],
    vertebrae=[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    gluteus_right=[80, 81],
    gluteus_left=[82, 83],
)
# fmt: on


def get_totalsegmentator_classes(name: str) -> list[int]:
    if name in GROUPS:
        return GROUPS[name]
    elif name in TotalSegmentator.__members__:
        return [TotalSegmentator[name].value]
    # elif name in TotalSegmentatorAppendicular.__members__:
    #     return [TotalSegmentatorAppendicular[name].value]
    else:
        return []


class AppendicularBones(Enum):
    patella = 1
    tibia = 2
    fibula = 3
    tarsal = 4
    metatarsal = 5
    phalanges_feet = 6
    ulna = 7
    radius = 8
    carpal = 9
    metacarpal = 10
    phalanges_hand = 11


class Task(StrEnum):
    total = auto()
    total_mr = auto()
    lung_vessels = auto()
    body = auto()
    cerebral_bleed = auto()
    hip_implant = auto()
    coronary_arteries = auto()
    pleural_pericard_effusion = auto()
    head_glands_cavities = auto()
    head_muscles = auto()
    headneck_bones_vessels = auto()
    headneck_muscles = auto()
    liver_vessels = auto()
    oculomotor_muscles = auto()
    heartchambers_highres = auto()
    appendicular_bones = auto()
    tissue_types = auto()
    tissue_types_mr = auto()
    brain_structures = auto()
    vertebrae_body = auto()
    face = auto()


TASK_CLASSES: dict[Task, Type[Enum]] = {
    Task.total: Total,
    Task.body: Body,
    Task.appendicular_bones: AppendicularBones,
}
