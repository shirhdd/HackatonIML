import pandas as pd
import numpy as np
from utils import BasicStage
from utils import FormName
from utils import MarginType
from utils import Side
from utils import SurgeryActivity

stage_map_size = {
    '0': 0,
    '0a': 0,
    '0is': 0,  # invasive cancer
    '1': 0.1,
    '1a': 0.3,
    '1b': 0.75,
    '1c': 1.5,
    '2': 3,
    '2a': 3.5,
    '2b': 4.5,
    '3': 5,
    '3a': 6,
    '3b': 7,
    '3c': 8,
    '4': 10
}
stage_map_types = {
    '0': 0.0,
    '0a': 0.1,
    '0is': 0.2,  # invasive cancer
    '1': 1.0,
    '1a': 1.1,
    '1b': 1.2,
    '1c': 1.3,
    '2': 2,
    '2a': 2.1,
    '2b': 2.2,
    '2c': 2.3,
    '3': 3.0,
    '3a': 3.1,
    '3b': 3.2,
    '3c': 3.3,
    '4': 4
}

side_map = {
    'nan': Side.none.value,
    'שמאל': Side.left.value,
    'ימין': Side.right.value,
    'דו צדדי': Side.both.value,
}
margin_type_map = {
    'ללא': MarginType.without.value,
    'נקיים': MarginType.clean.value,
    'נגועים': MarginType.infected.value,
}
basic_stage_map = {
    'c - Clinical': BasicStage.clinical.value,
    'p - Pathological': BasicStage.pathological.value,
    'Null': None,
    'r - Reccurent': BasicStage.recurrent.value
}
form_name_map = {
    'דיווח סיעודי': FormName.nursing_report.value,
    'ביקור במרפאה': FormName.visit_the_clinic.value,
    'אומדן סימפטומים ודיווח סיעודי': FormName.assessment_symptoms_and_nursing_report.value,
    'ביקור במרפאה קרינה': FormName.visit_the_radiation_clinic.value,
    'אנמנזה סיעודית': FormName.nursing_anamnesis.value,
    'אנמנזה רפואית': FormName.medical_anamnesis.value,
    'ביקור במרפאה המטו-אונקולוגית': FormName.visit_the_hemato_oncology_clinic.value,
    'אנמנזה סיעודית קצרה': FormName.short_nursing_anamnesis.value,
    'אנמנזה רפואית המטו-אונקולוגית': FormName.hemato_oncological_medical_anamnesis.value,
}
surgery_bef_aft_activity_map = {
    'nan': None,
    'כיר-לאפ-הוצ טבעת/שנוי מי': SurgeryActivity.hero_lup_hot_ring.value,
    'כירו-שד-למפקטומי+בלוטות': SurgeryActivity.chiro_breast_lymphectomy_glands.value,
    'כירו-שד-מסטקטומי+בלוטות': SurgeryActivity.chiro_breast_mastectomy_glands.value,
    'כירורגיה-שד למפקטומי': SurgeryActivity.surgery_breast_lymphectomy.value,
    'שד-כריתה בגישה זעירה+בלוטות': SurgeryActivity.breast_resection_with_small_gland_access.value,
    '(intrabeam)': SurgeryActivity.chiro_breast_lumpectomy_glandular_radiation_intrabeam.value,
    'שד-כריתה בגישה זעירה דרך העטרה': SurgeryActivity.breast_resection_small_access_through_the_crown.value,
    'כירור-הוצאת בלוטות לימפה': SurgeryActivity.removal_of_lymph_glands.value,
    'כיר-שד-הוצ.בלוטות בית שח': SurgeryActivity.removal_armpit_glands.value,
    'כירורגיה-שד מסטקטומי': SurgeryActivity.mastectomy_breast_surgery.value,
    'כירו-שד-למפקטומי+בלוטות+קרינה תוך ניתוחית': SurgeryActivity.chiro_breast_lumpectomy_glandular_radiation_in_surgery.value,
}

side_map = {'nan': Side.none,
            'שמאל': Side.left,
            'ימין': Side.right,
            'דו צדדי': Side.both}
margin_type_map = {'ללא': MarginType.without,
                   'נקיים': MarginType.clean,
                   'נגועים': MarginType.infected}
basic_stage_map = {'c - Clinical': BasicStage.clinical,
                   'p - Pathological': BasicStage.pathological,
                   'Null': None,
                   'r - Reccurent': BasicStage.recurrent}
form_name_map = {'דיווח סיעודי': FormName.nursing_report,
                 'ביקור במרפאה': FormName.visit_the_clinic,
                 'אומדן סימפטומים ודיווח סיעודי': FormName.assessment_symptoms_and_nursing_report,
                 'ביקור במרפאה קרינה': FormName.visit_the_radiation_clinic,
                 'אנמנזה סיעודית': FormName.nursing_anamnesis,
                 'אנמנזה רפואית': FormName.medical_anamnesis,
                 'ביקור במרפאה המטו-אונקולוגית': FormName.visit_the_hemato_oncology_clinic,
                 'אנמנזה סיעודית קצרה': FormName.short_nursing_anamnesis,
                 'אנמנזה רפואית המטו-אונקולוגית': FormName.hemato_oncological_medical_anamnesis}
surgery_bef_aft_activity_map = {'nan': None,
                                'כיר-לאפ-הוצ טבעת/שנוי מי': SurgeryActivity.hero_lup_hot_ring,
                                'כירו-שד-למפקטומי+בלוטות': SurgeryActivity.chiro_breast_lymphectomy_glands,
                                'כירו-שד-מסטקטומי+בלוטות': SurgeryActivity.chiro_breast_mastectomy_glands,
                                'כירורגיה-שד למפקטומי': SurgeryActivity.surgery_breast_lymphectomy,
                                'שד-כריתה בגישה זעירה+בלוטות': SurgeryActivity.breast_resection_with_small_gland_access,
                                'כירו-שד-למפקטומי+בלוטות+קרינה תוך ניתוחית'
                                '(intrabeam)': SurgeryActivity.chiro_breast_lumpectomy_glandular_radiation_intrabeam,
                                'שד-כריתה בגישה זעירה דרך העטרה': SurgeryActivity.breast_resection_small_access_through_the_crown,
                                'כירור-הוצאת בלוטות לימפה': SurgeryActivity.removal_of_lymph_glands,
                                'כיר-שד-הוצ.בלוטות בית שח': SurgeryActivity.removal_armpit_glands,
                                'כירורגיה-שד מסטקטומי': SurgeryActivity.mastectomy_breast_surgery}
COLUMN_NAMES = ['ADR-Adrenals', 'BON-Bones', 'BRA-Brain', 'HEP-Hepatic', 'LYM-Lymphnodes', 'MAR-BoneMarrow',
                'OTH-Other', 'PER-Peritoneum', 'PLE-Pleura', 'PUL-Pulmonary',  'SKI-Skin']

def load_data(filename: str):
    df = pd.read_csv(filename, dtype={
        'אבחנה-Surgery date3': str,
        'אבחנה-Surgery name3': str,
        'אבחנה-Ivi -Lymphovascular invasion': str
    })
    df.columns = ['Form name', 'Hospital', 'User Name', 'Age', 'Basic stage',
                  'Diagnosis date', 'Her2', 'Histological diagnosis',
                  'Histopatological degree', 'Ivi -Lymphovascular invasion',
                  'KI67 protein', 'Lymphatic penetration',
                  'M -metastases mark (TNM)', 'Margin Type',
                  'N -lymph nodes mark (TNM)', 'Nodes exam', 'Positive nodes',
                  'Side', 'Stage', 'Surgery date1', 'Surgery date2',
                  'Surgery date3', 'Surgery name1', 'Surgery name2',
                  'Surgery name3', 'Surgery sum', 'T -Tumor mark (TNM)',
                  'Tumor depth', 'Tumor width', 'er', 'pr',
                  'surgery before or after-Activity date',
                  'surgery before or after-Actual activity',
                  'id-hushed_internalpatientid']
    return df


def handle_Ivi(df):
    conditions = [
        df['Ivi -Lymphovascular invasion'].str.contains(r'neg', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'pos', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'-', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'no', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'\(-\)', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'\(+\)', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'extensive',
                                                        case=False, na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'none', case=False,
                                                        na=False)]

    choices = [-1, 1, -1, -1, -1, 1, 1, 0]

    df['Ivi -Lymphovascular invasion'] = np.select(conditions, choices,
                                                   default=None)
    return df


def handle_KI_protein(df):
    pattern1 = r'(\d+)\s*%'
    pattern2 = r'^(\d+)$'

    # Create a new column 'Percent KI67 protein' and extract the values
    df['Percent KI67 protein'] = df['KI67 protein'].str.extract(pattern1,
                                                                expand=False)
    df['Percent KI67 protein'] = df['Percent KI67 protein'].fillna(
        df['KI67 protein'].str.extract(pattern2, expand=False))

    # Handle the case when a range of percentages is present
    range_pattern = r'(\d+)\s*%?\s*-\s*(\d+)\s*%?'
    ranges = df['KI67 protein'].str.extract(range_pattern)
    mask = ranges.notnull().all(axis=1)
    df.loc[mask, 'Percent KI67 protein'] = ranges[mask].astype(float).max(
        axis=1)

    # Fill 0 for date values
    date_pattern = r'\d{1,2}-[A-Za-z]{3}'
    is_date = df['KI67 protein'].str.contains(date_pattern, na=False)
    df.loc[is_date, 'Percent KI67 protein'] = 0

    # Convert the column to float
    df['Percent KI67 protein'] = df['Percent KI67 protein'].fillna(0).astype(
        float)
    df.drop('KI67 protein', axis=1, inplace=True)
    return df


def handle_stage(df):
    df['Stage'] = df['Stage'].str.extract(
        r'Stage(.*)', expand=False)
    df['Stage'] = df['Stage'].map(stage_map_size)
    return df


def handle_pr_er(df):
    for _ in ['pr', 'er']:
        conditions = [
            df[_].str.contains(r'neg', case=False, na=False),
            df[_].str.contains(r'pos', case=False, na=False),
            df[_].str.contains(r'\+', case=False, na=False),
            df[_].str.contains(r'3', case=False, na=False),
            df[_].str.contains(r'\(-\)', case=False, na=False),
            (df[_].str.contains(r'(\d{2,})%?', case=False, na=False)) & (
                    df[_].str.extract(r'(\d{2,})%?', expand=False).astype(
                        float) >= 50),
            (df[_].str.contains(r'(\d{2,})%?', case=False, na=False)) & (
                    df[_].str.extract(r'(\d{2,})%?', expand=False).astype(
                        float) < 50)]

        choices = [-1, 1, 1, 1, -1, 1, -1]

        df[_] = np.select(conditions, choices, default=None)
    return df


# -1=neg,0=undefined,1=pos
def handle_Her2(df):
    conditions = [
        df['Her2'].str.contains(r'neg', case=False, na=False),
        df['Her2'].str.contains(r'pos', case=False, na=False),
        df['Her2'].str.contains(r'[0|1]', case=False, na=False),
        df['Her2'].str.contains(r'2', case=False, na=False),
        df['Her2'].str.contains(r'3', case=False, na=False),
        df['Her2'].str.contains(r'\(-\)', case=False, na=False),
        df['Her2'].str.contains(r'^FISH$', case=False, na=False),
        df['Her2'].str.contains(r'\+\+\+', case=False, na=False),
        df['Her2'].str.contains(r'non', case=False, na=False),
        df['Her2'].str.contains(r'\@', case=False, na=False),
        df['Her2'].str.contains(r'^\+$', case=False, na=False),
    ]

    choices = [-1, 1, -1, 0, 1, -1, 1, 1, 0, 0, 1]

    df['Her2'] = np.select(conditions, choices, default=None)
    return df


def extracting_data(df: pd.DataFrame):
    df['Histopatological degree'] = df['Histopatological degree'].str.extract(
        r'G(.)', expand=False)
    df.loc[df['Histopatological degree'] == 'X', 'Histopatological degree'] = 5
    df['M -metastases mark (TNM)'] = df[
        'M -metastases mark (TNM)'].str.extract(
        r'M(.)', expand=False)
    df.loc[df[
               'M -metastases mark (TNM)'] == 'X', 'M -metastases mark (TNM)'] = 2  # cannot be measured
    df['N -lymph nodes mark (TNM)'] = df[
        'N -lymph nodes mark (TNM)'].str.extract(
        r'N(\d)', expand=False).replace(np.nan, 0).astype(int)
    df['T -Tumor mark (TNM)'] = df['T -Tumor mark (TNM)'].str.extract(
        r'T(\d)', expand=False).replace(np.nan, 0).astype(int)
    df['Lymphatic penetration'] = df['Lymphatic penetration'].str.extract(
        r'L(.)', expand=False)
    df.loc[df['Lymphatic penetration'] == 'I', 'Lymphatic penetration'] = 3
    df['User Name'] = df['User Name'].str.extract(
        r'(\d+)_Onco', expand=False).replace(np.nan, 0).astype(int)
    return df


def map_data(df):
    df['Side'] = df['Side'].map(side_map)
    df['Form name'] = df['Form name'].map(form_name_map)
    df['Basic stage'] = df['Basic stage'].map(basic_stage_map)
    df['Margin Type'] = df['Margin Type'].map(margin_type_map)
    df['surgery before or after-Actual activity'] = df[
        'surgery before or after-Actual activity'].map(
        surgery_bef_aft_activity_map)
    return df


def date_process(df):
    date_columns = ['surgery before or after-Activity date', 'Surgery date3',
                    'Surgery date2', 'Surgery date1', 'Diagnosis date']
    for date_type in date_columns:
        df[date_type] = pd.to_datetime(df[date_type], errors="coerce",
                                       infer_datetime_format=True)
    diag_to_surg = (df['Diagnosis date'] - df['Surgery date1']).dt.days
    df = df[~(diag_to_surg < 0)]
    df["diagnoses_to_surgery_days"] = diag_to_surg[diag_to_surg > 0]
    return df


def preprocess_labels_q1(file_path):
    A = 'אבחנה-Location of distal metastases'
    df = pd.read_csv(file_path)
    myList = []
    for strings in df[A]:
        labels = strings[2:-2].replace("'", "").split(", ")
        if labels != ['']:
            myList.append(labels)
        else:
            myList.append('[]')
    df = pd.DataFrame({'labels': myList})
    df = pd.get_dummies(df.explode(column='labels'), prefix="", prefix_sep="").groupby(level=0).sum()
    return df


def to_number(df):
    df['Hospital'] = pd.to_numeric(df['Hospital'], errors='coerce').astype(
        'Int64')
    df['Nodes exam'] = pd.to_numeric(df['Nodes exam'], errors='coerce').astype(
        'Int64')
    df['Positive nodes'] = pd.to_numeric(df['Positive nodes'],
                                         errors='coerce').astype(
        'Int64')
    df['Surgery sum'] = pd.to_numeric(df['Surgery sum'],
                                      errors='coerce').astype(
        'Int64')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype(
        float)
    df['Tumor width'] = pd.to_numeric(df['Tumor width'],
                                      errors='coerce').astype(
        float)
    return df


def handling_features(df: pd.DataFrame):
    df = extracting_data(df)
    df = map_data(df)
    df = handle_Her2(df)
    df = handle_stage(df)
    df = handle_pr_er(df)
    df = handle_KI_protein(df)
    df = handle_Ivi(df)
    df = date_process(df)
    return df


def preprocessor(df: pd.DataFrame):
    columns_to_drop = [  # 'Form name',
        # 'User Name',
        # 'Basic stage',
        # 'Diagnosis date',
        # 'Her2',
        'Histological diagnosis',
        # 'Histopatological degree',
        # 'Ivi -Lymphovascular invasion',
        # 'KI67 protein',
        # 'Lymphatic penetration',
        # 'M -metastases mark (TNM)',
        # 'Margin Type',
        # 'N -lymph nodes mark (TNM)',
        # 'Side',
        # 'Stage',
        # 'Surgery date1', 'Surgery date2', 'Surgery date3',
        'Surgery name1',
        'Surgery name2',
        'Surgery name3',
        # 'T -Tumor mark (TNM)',
        # 'er',
        # 'pr',
        # 'surgery before or after-Activity date',
        #  'surgery before or after-Actual activity',
        'id-hushed_internalpatientid']
    df.drop(columns_to_drop, axis=1, inplace=True)
    df = to_number(df)
    df = handling_features(df)
    X = df.fillna(0)
    return X


def load_and_preproc(filename: str):
    df = load_data(filename)
    # if not test:
    # fit
    return preprocessor(df)


if __name__ == '__main__':
    print(load_and_preproc("train_sets/train.csv"))
