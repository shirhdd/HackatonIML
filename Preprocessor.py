import pandas as pd


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

# def extracting_data(df: pd.DataFrame):


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
    print(main("train.feats.csv"))
