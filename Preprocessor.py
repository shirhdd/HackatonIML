import pandas as pd
import enum


class FormName(enum.Enum):
    nursing_report = 1
    visit_the_clinic = 2
    assessment_symptoms_and_nursing_report = 3
    visit_the_radiation_clinic = 4
    nursing_anamnesis = 5
    medical_anamnesis = 6
    visit_the_hemato_oncology_clinic = 7
    short_nursing_anamnesis = 8
    hemato_oncological_medical_anamnesis = 9


class SurgeryActivity(enum.Enum):
    hero_lup_hot_ring = 1,
    chiro_breast_lymphectomy_glands = 2,
    chiro_breast_mastectomy_glands = 3,
    surgery_breast_lymphectomy = 4,
    breast_resection_with_small_gland_access = 5,
    chiro_breast_lumpectomy_glandular_radiation_intrabeam = 6,
    breast_resection_small_access_through_the_crown = 7,
    removal_of_lymph_glands = 8,
    removal_armpit_glands = 9
    mastectomy_breast_surgery = 10


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
    df = pd.read_csv(filename, dtype=str)
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
        r'(G.))', expand=False)
    df['Stage'] = df['Stage'].str.extract(
        r'(Stage.+))', expand=False)
    df['M -metastases mark (TNM)'] = df[
        'M -metastases mark (TNM)'].str.extract(
        r'(M.)', expand=False)
    df['N -lymph nodes mark (TNM)'] = df[
        'N -lymph nodes mark (TNM)'].str.extract(
        r'(N.)', expand=False)
    df['T -Tumor mark (TNM)'] = df['T -Tumor mark (TNM)'].str.extract(
        r'(T.)', expand=False)
    df['Lymphatic penetration'] = df['Lymphatic penetration'].str.extract(
        r'(L.)', expand=False)  # TODO: there is rest of the sentence to cut
    return df


def map_data(df):
    df['Form name'] = df['Form name'].map(form_name_map)
    df['surgery before or after-Actual activity'] = df[
        'surgery before or after-Actual activity'].map(
        surgery_bef_aft_activity_map)
    return df


def date_process(df):
    date_columns = ['surgery before or after-Activity date', 'Surgery date3',
                    'Surgery date2', 'Surgery date1', 'Diagnosis date']


def preprocess_labels_q1(file_path):
    A = 'אבחנה-Location of distal metastases'
    df = pd.read_csv(file_path)
    myList = []
    # for strings in df[A]:
    #     myList.append(strings[2:-2].replace(" ", "").replace("'", "").split(","))
    # df = pd.DataFrame({'labels': myList})
    return pd.get_dummies(df)


def preprocessor(df: pd.DataFrame):
    # TODO: check valid date for the 5 dates from the columns
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
    X = df.fillna(0)
    return X




def main(filename: str):
    df = load_data(filename)
    # if not test:
    # fit
    return preprocessor(df)


if __name__ == '__main__':
    print(main("given_sets/train.feats.csv"))
