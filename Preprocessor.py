import pandas as pd
from utils import Side, MarginType, BasicStage, FormName, SurgeryActivity

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


def load_data(filename: str):
    df = pd.read_csv(filename)
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


def extracting_data(df: pd.DataFrame):
    df['Histopatological degree'] = df['Histopatological degree'].str.extract(
        r'G(.))', expand=False)
    # df['Stage'] = df['Stage'].str.extract(
    #     r'(Stage.+))', expand=False) //TODO: there are values with nan/LA/not yet
    df['M -metastases mark (TNM)'] = df[
        'M -metastases mark (TNM)'].str.extract(
        r'M(.)', expand=False)
    df['N -lymph nodes mark (TNM)'] = df[
        'N -lymph nodes mark (TNM)'].str.extract(
        r'N(.)', expand=False)
    df['T -Tumor mark (TNM)'] = df['T -Tumor mark (TNM)'].str.extract(
        r'T(.)', expand=False)
    df['Lymphatic penetration'] = df['Lymphatic penetration'].str.extract(
        r'L(.)', expand=False)  # TODO: there is rest of the sentence to cut
    df['User Name'] = df['User Name'].str.extract(
        r'(\d+)_Onco', expand=False)
    return df


def map_data(df):
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
        myList.append(
            strings[2:-2].replace(" ", "").replace("'", "").split(","))
    df = pd.DataFrame({'labels': myList})
    return pd.get_dummies(df.explode(column='labels')).groupby(level=0).sum()


def toNumber(df):
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


def preprocessor(df: pd.DataFrame):
    columns_to_drop = ['Form name', 'User Name', 'Basic stage',
                       'Diagnosis date', 'Her2', 'Histological diagnosis',
                       'Histopatological degree',
                       'Ivi -Lymphovascular invasion',
                       'KI67 protein', 'Lymphatic penetration',
                       'M -metastases mark (TNM)', 'Margin Type',
                       'N -lymph nodes mark (TNM)',
                       'Side', 'Stage', 'Surgery date1', 'Surgery date2',
                       'Surgery date3', 'Surgery name1', 'Surgery name2',
                       'Surgery name3', 'T -Tumor mark (TNM)', 'er', 'pr',
                       'surgery before or after-Activity date',
                       'surgery before or after-Actual activity',
                       'id-hushed_internalpatientid']
    df.drop(columns_to_drop, axis=1, inplace=True)
    df = toNumber(df)
    X = df.fillna(0)
    return X


def load_and_preproc(filename: str):
    df = load_data(filename)
    # if not test:
    # fit
    return preprocessor(df)


if __name__ == '__main__':
    print(load_and_preproc("train_sets/train.csv"))
