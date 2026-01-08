import pandas as pd
import numpy as np
import logging as log
import os

from consts import ISCO_MAPPING


def normalize_grades_column(series, base, limit):
    """
    Takes a scale of <base> to <limit> and turn it into a scale of 1 to 100
    :param series: list of grades
    :param base: The minimum possible score
    :param limit: The maximum possible score
    :return: Normalized series
    """
    if limit < 1:
        raise ArithmeticError('normalize_grades_column cannot get a limit smaller than 1')
    if limit < base:
        raise ArithmeticError('Limit cannot be smaller than base')
    result = series - base
    result = result / limit
    result *= 100
    return result


def convert_strings_to_numbers(series):
    """Convert a string representing a grade to a number"""
    series = series.replace("<1", 0)
    series = pd.to_numeric(series)
    return series


def fetch_total_math_score(df):
    """
    Take the columns corresponding to math abilities, and sum them to a total math score
    :return: pd.series containing a total math score for each participant
    """

    # The following columns have a range of 0 to 19
    nineteen_limit_columns = [
        'KeyMath_Numeration_ScS',
        'KeyMath_Measurement_ScS',
        'KeyMath_ProblemSolving_ScS',
        'TOMA-2_Attitudes_StS',
    ]
    nineteen_limits_base = 0
    nineteen_limits_limit = 19

    # This column has a range of 40 to 160
    wj_column = 'WJ-III_MathFluency_StS'
    wj_base = 40
    wj_limit = 140

    normalized_series = []
    result = normalize_grades_column(df[wj_column], wj_base, wj_limit)

    for column_name in nineteen_limit_columns:
        series = convert_strings_to_numbers(df[column_name])
        result += normalize_grades_column(series, nineteen_limits_base, nineteen_limits_limit)

    # Normalize the final result
    final_base = 0
    final_limit = 500
    final_result = normalize_grades_column(result, final_base, final_limit)
    return final_result


def calculate_lisas(df):
    """
    Calculating LISAS - Linear Integrated Speed-Accuracy Score
    """
    df = df.dropna(subset=['response_time', 'accuracy'])
    correct_trials = df[df['accuracy'] == 1]
    if correct_trials.empty:
        return np.nan

    rt_mean, rt_std = correct_trials['response_time'].mean(), correct_trials['response_time'].std()
    acc_mean = df['accuracy'].mean()
    pe, spe = 1 - acc_mean, np.sqrt(acc_mean * (1 - acc_mean))

    if spe == 0 or np.isnan(rt_std):
        return rt_mean
    return rt_mean + (pe * (rt_std / spe))


def map_isco_score(occ):
    """
    Get a profession and return how humane\realistic it is, on a scale of 0-100. (humane=0)
    """
    occ_str = str(occ).strip().lower()

    # Unemployed = Nan , Self-employed = 65
    if occ_str in ['unemployed', 'nan', 'none']:
        return np.nan
    if 'self-employed' in occ_str:
        return 65

    return ISCO_MAPPING.get(occ_str[:2], np.nan)


# Z-score standardization
def standardize_scores(df):
    df['z'] = df.groupby('task')['lisas'].transform(lambda x: (x - x.mean()) / x.std() * -1)
    return df


def pivot_to_subject_level(df):
    """Reorganizing data (subject focus)"""
    df['col_name'] = df['task'].str.replace('task-', '') + "_" + df['run']
    pivot_df = df.pivot(index='subject', columns='col_name', values='z')
    pivot_df['overall_accuracy'] = df.groupby('subject')['acc'].mean()
    return pivot_df.reset_index()


# Calculating capacity and improvement and total score
def calculate_metrics(pivot_df):

    # Constructing Tests array
    tasks = set([col.split('_run')[0] for col in pivot_df.columns if '_run' in col])
    rel_improvement_cols = []

    for t in tasks:
        r1, r2 = f"{t}_run-01", f"{t}_run-02"
        if r1 in pivot_df.columns and r2 in pivot_df.columns:
            # 2nd run - 1st run
            raw_delta = pivot_df[r2] - pivot_df[r1]

            # Highest score - Lowest score (range of standard of improvement)
            all_task_scores = pd.concat([pivot_df[r1], pivot_df[r2]])
            task_range = all_task_scores.max() - all_task_scores.min()

            # Calculating improvement relative to the range
            pivot_df[f"{t}_relative_improvement"] = raw_delta / task_range if task_range != 0 else 0
            rel_improvement_cols.append(f"{t}_relative_improvement")

    # Capacity and relative improvement score
    pivot_df['global_efficiency_index'] = pivot_df.filter(like='run').mean(axis=1)
    pivot_df['total_relative_improvement'] = pivot_df[rel_improvement_cols].mean(axis=1)

    # Total score 70% capacity and 30% relative improvement
    pivot_df['final_composite_score'] = (pivot_df['global_efficiency_index'] * 0.7) + (
        pivot_df['total_relative_improvement'] * 0.3
    )

    return pivot_df


def merge_parental_bias(final_df, main_dataset_path):
    """Merging parental occupation data"""
    if not os.path.exists(main_dataset_path):
        return final_df

    parents_df = pd.read_csv(main_dataset_path)
    parents_df.rename(columns={'participant_id': 'subject'}, inplace=True)

    parents_df['mother_score'] = parents_df['mother_occupation'].apply(map_isco_score)
    parents_df['father_score'] = parents_df['father_occupation'].apply(map_isco_score)
    parents_df['parental_bias'] = parents_df[['mother_score', 'father_score']].mean(axis=1)

    # מיזוג - שומרים על אבא, אמא וממוצע
    return final_df.merge(
        parents_df[['subject', 'mother_score', 'father_score', 'parental_bias']], on='subject', how='left'
    )


# --- הפעלה מרכזית ---
def run_comprehensive_lisas_analysis(root_dir, main_dataset_path="main_dataset.csv"):
    raw_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tsv'):
                parts = file.split('_')
                sub, task, run = parts[0], parts[2], parts[3]
                df_file = pd.read_csv(os.path.join(root, file), sep='\t')
                raw_list.append({
                    'subject': sub, 'task': task, 'run': run,
                    'lisas': calculate_lisas(df_file), 'acc': df_file['accuracy'].mean()
                })
    
    df = pd.DataFrame(raw_list)
    df = standardize_scores(df)
    pivot_df = pivot_to_subject_level(df)

    # כל החישובים הקוגניטיביים מרוכזים כאן
    final_df = calculate_metrics(pivot_df)

    # הוספת נתוני הורים
    final_df = merge_parental_bias(final_df, main_dataset_path)

    # שמירה ל-CSV מסודר
    # final_df.round(3).to_csv("final_analysis_results.csv", index=False)
    return final_df

df = run_comprehensive_lisas_analysis('trial_data')
print(df.head(40))





