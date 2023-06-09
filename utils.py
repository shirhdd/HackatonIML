import enum


class FormName(enum.Enum):
    visit_the_clinic = 0
    medical_anamnesis = 1
    short_nursing_anamnesis = 2
    nursing_anamnesis = 3
    assessment_symptoms_and_nursing_report = 4
    hemato_oncological_medical_anamnesis = 5
    visit_the_hemato_oncology_clinic = 6
    visit_the_radiation_clinic = 7
    nursing_report = 8


class SurgeryActivity(enum.Enum):
    hero_lup_hot_ring = 1
    chiro_breast_lymphectomy_glands = 2
    chiro_breast_mastectomy_glands = 3
    surgery_breast_lymphectomy = 4
    breast_resection_with_small_gland_access = 5
    chiro_breast_lumpectomy_glandular_radiation_intrabeam = 6
    breast_resection_small_access_through_the_crown = 7
    removal_of_lymph_glands = 8
    removal_armpit_glands = 9
    mastectomy_breast_surgery = 10


class BasicStage(enum.Enum):
    clinical = 1
    pathological = 2
    recurrent = 3


class MarginType(enum.Enum):
    without = 0
    clean = 1
    infected = 2


class Side(enum.Enum):
    none = 0
    left = 1
    right = 2
    both = 3