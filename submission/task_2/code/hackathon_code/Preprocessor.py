import numpy as np
import pandas as pd

from hackathon_code.utils import *

YOUNG_AGE = 40

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
    np.nan: Side.none,
    'שמאל': Side.left,
    'ימין': Side.right,
    'דו צדדי': Side.both,
}
margin_type_map = {
    'ללא': MarginType.without,
    'נקיים': MarginType.clean,
    'נגועים': MarginType.infected,
}
basic_stage_map = {
    'r - Reccurent': BasicStage.recurrent,
    'c - Clinical': BasicStage.clinical,
    'p - Pathological': BasicStage.pathological,
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
    # 'nan': None,
    'כירו-שד-למפקטומי+בלוטות': SurgeryActivity.chiro_breast_lymphectomy_glands,
    'כירורגיה-שד למפקטומי': SurgeryActivity.surgery_breast_lymphectomy,
    'שד-כריתה בגישה זעירה+בלוטות': SurgeryActivity.breast_resection_with_small_gland_access,
    'כירו-שד-מסטקטומי+בלוטות': SurgeryActivity.chiro_breast_mastectomy_glands,
    'כירו-שד-למפקטומי+בלוטות+קרינה תוך ניתוחית (intrabeam)': SurgeryActivity.chiro_breast_lumpectomy_glandular_radiation_intrabeam,
    'שד-כריתה בגישה זעירה דרך העטרה': SurgeryActivity.breast_resection_small_access_through_the_crown,
    'כירור-הוצאת בלוטות לימפה': SurgeryActivity.removal_of_lymph_glands,
    'כירורגיה-שד מסטקטומי': SurgeryActivity.mastectomy_breast_surgery,
    'כיר-לאפ-הוצ טבעת/שנוי מי': SurgeryActivity.hero_lup_hot_ring,
    'כיר-שד-הוצ.בלוטות בית שח': SurgeryActivity.removal_armpit_glands,
}


def load_data(filename: str):
    df = pd.read_csv(filename, dtype={
        'אבחנה-Surgery date2': str,
        'אבחנה-Surgery date3': str,
        'אבחנה-Surgery name2': str,
        'אבחנה-Surgery name3': str,
        'אבחנה-Ivi -Lymphovascular invasion': str
    })
    df.columns = [
        'Form name', 'Hospital', 'User Name', 'Age', 'Basic stage',
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
        'id-hushed_internalpatientid',
    ]
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
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'yes', case=False,
                                                        na=False),
        df['Ivi -Lymphovascular invasion'].str.contains(r'not', case=False,
                                                        na=False)]

    choices = [-1, 1, -1, -1, -1, 1, 1, 0, 1, -1]

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
    mean_percent = df['Percent KI67 protein'].mean()
    df['Percent KI67 protein'].fillna(mean_percent, inplace=True)
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
            (df[_].str.contains(r'\d{2,}%?', case=False, na=False)) &
            (df[_].str.extract(r'(\d{2,})%?', expand=False).astype(float) >= 50),
            (df[_].str.contains(r'\d{2,}%?', case=False, na=False)) & (
                    df[_].str.extract(r'(\d{2,})%?', expand=False).astype(float) < 50),
        ]

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


def count_surgeries(df: pd.DataFrame) -> pd.DataFrame:
    new_column = df.loc[:, ['Surgery name1', 'Surgery name2', 'Surgery name3']].count(axis=1)
    df['num_of_surgeries'] = new_column
    return df


def create_dummies(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, prefix="", prefix_sep="", columns=['Side'])
    df = pd.get_dummies(df, prefix="", prefix_sep="", columns=['Margin Type'])
    df = pd.get_dummies(df, prefix="", prefix_sep="", columns=['Basic stage'])
    df = pd.get_dummies(df, prefix="", prefix_sep="", columns=['surgery before or after-Actual activity'])
    return df


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_young"] = np.where(df["Age"] < YOUNG_AGE, 1, 0)

    # handle tumor volume
    # df['tumor_volume'] = df['Lymphatic penetration'] * df['Tumor width']
    #
    # # Define the bin edges
    # tumor_volume_bins = [0, 5, 10, 20, float('inf')]
    #
    # # Define the labels for the bins
    # labels = ['0-5', '5-10', '10-20', '20+']
    #
    # # Create a new column with the bin labels based on the numeric_column values
    # df['bin_column'] = pd.cut(df['tumor_volume'], bins=tumor_volume_bins, labels=labels, right=False)
    #
    # print(df['bin_column'].head())
    return df


def extracting_data(df: pd.DataFrame):
    df['Histopatological degree'] = df['Histopatological degree'].str.extract(
        r'G(.)', expand=False)
    df.loc[df['Histopatological degree'] == 'X', 'Histopatological degree'] = 5
    df['M -metastases mark (TNM)'] = df[
        'M -metastases mark (TNM)'].str.extract(
        r'M(.)', expand=False)
    df.loc[df['M -metastases mark (TNM)'] == 'X', 'M -metastases mark (TNM)'] = 2  # cannot be measured
    df['N -lymph nodes mark (TNM)'] = df[
        'N -lymph nodes mark (TNM)'].str.extract(
        r'N(\d)', expand=False).replace(np.nan, 0).astype(int)
    df['T -Tumor mark (TNM)'] = df['T -Tumor mark (TNM)'].str.extract(
        r'T(\d)', expand=False).replace(np.nan, 0).astype(int)
    df['Lymphatic penetration'] = df['Lymphatic penetration'].str.extract(
        r'L(.)', expand=False)
    df['has penetration'] = 0
    df.loc[df['Lymphatic penetration'] == 'I', 'Lymphatic penetration'] = 3
    df.loc[df['Lymphatic penetration'] == 3, 'has penetration'] = 1
    df['User Name'] = df['User Name'].str.extract(
        r'(\d+)_Onco', expand=False).replace(np.nan, 0).astype(int)
    return df


def map_data(df):
    df['Side'] = df['Side'].map(side_map)
    df['Form name'] = df['Form name'].map(form_name_map)
    df['Basic stage'] = df['Basic stage'].map(basic_stage_map)
    df['Margin Type'] = df['Margin Type'].map(margin_type_map)
    df['surgery before or after-Actual activity'] = df['surgery before or after-Actual activity'].map(
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
        myList.append(strings[2:-2].replace(" ", "").replace("'", "").split(","))
    df = pd.DataFrame({'labels': myList})
    return pd.get_dummies(df.explode(column='labels')).groupby(level=0).sum()


def to_number(df):
    df['Hospital'] = pd.to_numeric(df['Hospital'], errors='coerce').astype(
        'Int64')
    df['Nodes exam'] = pd.to_numeric(df['Nodes exam'], errors='coerce').astype(
        'Int64')
    mean_nodes = int(df['Nodes exam'].mean())
    df['Nodes exam'].fillna(mean_nodes, inplace=True)

    df['Positive nodes'] = pd.to_numeric(df['Positive nodes'],
                                         errors='coerce').astype(
        'Int64')
    mean_positive_nodes = int(df['Positive nodes'].mean())
    df['Positive nodes'].fillna(mean_positive_nodes, inplace=True)

    df['Surgery sum'] = pd.to_numeric(df['Surgery sum'],
                                      errors='coerce').astype(
        'Int64')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype(
        float)
    mean_age = df['Age'].mean()
    df['Age'].fillna(mean_age, inplace=True)

    df['Tumor width'] = pd.to_numeric(df['Tumor width'],
                                      errors='coerce').astype(
        float)
    mean_width = df['Tumor width'].mean()
    df['Tumor width'].fillna(mean_width, inplace=True)
    return df


def handling_features(df: pd.DataFrame):
    df = extracting_data(df)
    df = map_data(df)
    df = handle_Her2(df)
    df = handle_stage(df)
    df = handle_pr_er(df)
    df = handle_KI_protein(df)
    df = handle_Ivi(df)
    # df = date_process(df)
    df = create_dummies(df)
    df = create_new_features(df)
    return df


def preprocessor(df: pd.DataFrame):
    columns_to_drop = [
        'Diagnosis date',
        'Histological diagnosis',
        'Surgery date1', 'Surgery date2', 'Surgery date3',
        'Surgery name1',
        'Surgery name2',
        'Surgery name3',
        'surgery before or after-Activity date',
        'id-hushed_internalpatientid',

        # columns to use

        # 'Form name',
        # 'User Name',
        # 'Basic stage',
        # 'Her2',
        # 'Histopatological degree',
        # 'Ivi -Lymphovascular invasion',
        # 'KI67 protein',
        # 'Lymphatic penetration',
        # 'M -metastases mark (TNM)',
        # 'Margin Type',
        # 'N -lymph nodes mark (TNM)',
        # 'Side',
        # 'Stage',
        # 'T -Tumor mark (TNM)',
        # 'er',
        # 'pr',
        #  'surgery before or after-Actual activity',
    ]
    df = to_number(df)
    df = handling_features(df)
    df.drop(columns_to_drop, axis=1, inplace=True)
    X = df.fillna(0)
    return X


def load_and_preproc(filename: str):
    df = load_data(filename)
    return preprocessor(df)
