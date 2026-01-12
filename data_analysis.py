import logging as log

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


def series_to_z_score(series, avg, std):
    # Fill na with avg
    result = series.fillna(avg)
    result = (result - avg) / std
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
    nineteen_limit_avg = 10
    nineteen_limit_std = 3

    # 100 is avg, 15 is std
    hundred_column_names = ['WJ-III_MathFluency_StS', 'CMAT_BasicCalc_Comp_Quotient']
    hundred_avg = 100
    hundred_std = 15

    result = series_to_z_score(df['WJ-III_MathFluency_StS'], hundred_avg, hundred_std)
    result += series_to_z_score(df['CMAT_BasicCalc_Comp_Quotient'], hundred_avg, hundred_std)
    # for column_name in hundred_column_names:
    #     result += series_to_z_score(df[column_name], hundred_avg, hundred_std)

    for column_name in nineteen_limit_columns:
        series = convert_strings_to_numbers(df[column_name])
        result += series_to_z_score(series, nineteen_limit_avg, nineteen_limit_std)

    # Normalize the final result
    final_avg = 0
    final_std = 6  # std == num of columns
    final_result = series_to_z_score(result, final_avg, final_std)
    return final_result


def fetch_total_verbal_score(df):
    """
    Take the columns corresponding to verbal abilities, and sum them to a total verbal score
    :return: pd.series containing a total verbal score for each participant
    """

    # The following columns have a range of 40 to 160
    column_names = ['AWMA-S_VerbalWM_StS', 'AWMA-S_VerbalSTM_StS', 'TOWRE_Total_StS', 'TOWRE_PD_StS', 'WASI_VIQ']
    score_avg = 100
    score_std = 15

    ctop_column_names = ['CTOPP_BW_StS', 'CTOPP_RL_StS']
    ctop_avg = 10
    ctop_std = 3

    fifty_avg_columns = 'WASI_Vocab_T-Score'
    fifty_avg_avg = 50
    fifty_avg_std = 10

    result = series_to_z_score(df[fifty_avg_columns], fifty_avg_avg, fifty_avg_std)
    for column_name in column_names:
        series = convert_strings_to_numbers(df[column_name])
        result += series_to_z_score(series, score_avg, score_std)

    for column_name in ctop_column_names:
        series = convert_strings_to_numbers(df[column_name])
        result += series_to_z_score(series, ctop_avg, ctop_std)

    # Normalize the final result
    final_avg = 0
    final_std = len(column_names) + len(ctop_column_names) + 1
    final_result = series_to_z_score(result, final_avg, final_std)
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
def run_comprehensive_lisas_analysis(raw_data_df, main_dataset_path="main_dataset.csv"):

    df = standardize_scores(raw_data_df)
    pivot_df = pivot_to_subject_level(df)

    # כל החישובים הקוגניטיביים מרוכזים כאן
    final_df = calculate_metrics(pivot_df)

    # הוספת נתוני הורים
    final_df = merge_parental_bias(final_df, main_dataset_path)

    # שמירה ל-CSV מסודר
    # final_df.round(3).to_csv("final_analysis_results.csv", index=False)
    return final_df
