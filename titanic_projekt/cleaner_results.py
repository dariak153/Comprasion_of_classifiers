import pandas as pd


def extract_cleaned_data_from_csv(csv_filepath: str, ignore_columns: list[str] = None):
    df = pd.read_csv(csv_filepath)
    if ignore_columns:
        for c in ignore_columns:
            if c in df.columns:
                df = df.drop(columns=[c])
    df = df.rename(columns={'Unnamed: 0': 'predictor'})
    df['features'] = df['predictor'].map(lambda x: '[' + x.split(' [')[-1] if '[' in x else None)
    df['predictor'] = df['predictor'].map(lambda x: x.split(' [')[0])
    df['cross_val_dict'] = df['cross_val_dict'].fillna(-1.0)
    df = df.fillna('None')
    df['best_estimators'] = df['best_estimators'].map(lambda x: ' '.join(x.replace('\n', '').split()))

    return df


r = extract_cleaned_data_from_csv('titanic_project/visresults/data/results.csv')
r.to_csv('../cln_results.csv', index=False)
del r
r1 = extract_cleaned_data_from_csv('titanic_project/visresults/data/results_1.csv', ['feature_importance_values'])
r1.to_csv('../cln_results_1.csv', index=False)
del r1
r2 = extract_cleaned_data_from_csv('titanic_project/visresults/data/results_2.csv')
r2.to_csv('../cln_results_2.csv', index=False)
del r2
r3 = extract_cleaned_data_from_csv('titanic_project/visresults/data/results_3.csv')
r3.to_csv('../cln_results_3.csv', index=False)
del r3