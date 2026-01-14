import os

import pandas as pd
import numpy as np
import logging as log
import statsmodels.api as sm

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


def standardize(series, invert=False):
    z = (series - series.mean()) / series.std()
    return z * -1 if invert else z


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


def calc_parents_income_level(df):
    for parent in ['father', 'mother']:

        # Take only the first number of occupation
        parent_jobs = df[f'{parent}_occupation'].str[0]

        # Unemployed = 9
        parent_jobs = parent_jobs.replace('U', 9)

        # Self employed =6
        parent_jobs = parent_jobs.replace('S', 6)

        # Nan = 9
        parent_jobs = parent_jobs.fillna(9)
        parent_jobs = pd.to_numeric(parent_jobs)
        parent_jobs = 9 - parent_jobs
        df[f'{parent}_income_level'] = parent_jobs

        # Instead of "1=rich" flip to "1=poor"

    return df


def model_stage1(df):
    # Dependent variable
    y = df["CMAT_BasicCalc_Comp_Quotient"]

    wm_col = "AWMA-S_VerbalWM_StS_t2"

    x_stage1 = df[["WASI_VIQ_t2", "WASI_PIQ_t2", "WASI_FSIQ_t2", wm_col]]

    x_stage1 = sm.add_constant(x_stage1)

    result = sm.OLS(y, x_stage1, missing="drop").fit()
    return result


def model_stage2(df):
    y = df["CMAT_BasicCalc_Comp_Quotient"]
    x_stage2 = df[
        [
            "WASI_VIQ_t2",
            "WASI_PIQ_t2",
            "WASI_FSIQ_t2",
            "AWMA-S_VerbalWM_StS_t2",
            "mother_highest_grade",
            "father_highest_grade",
            "regular_classroom",
        ]
    ]

    x_stage2 = sm.add_constant(x_stage2)

    result = sm.OLS(y, x_stage2, missing="drop").fit()
    return result


def calculate_lisas(df):
    """
    Calculating LISAS - Linear Integrated Speed-Accuracy Score

    """
    df = df.dropna(subset=['response_time', 'accuracy'])
    correct_trials = df[df['accuracy'] == 1]
    if correct_trials.empty:
        return np.nan

    # rt_ - response time for correct only

    rt_mean, rt_std = correct_trials['response_time'].mean(), correct_trials['response_time'].std()
    acc_mean = df['accuracy'].mean()
    # pe - personal error % , spe - std personal error
    pe, spe = 1 - acc_mean, np.sqrt(acc_mean * (1 - acc_mean))

    if spe == 0 or np.isnan(rt_std):
        return rt_mean
    return rt_mean + (pe * (rt_std / spe))


def map_isco_score(occ):
    """
    Map profession strings to ISCO scores (humane=0, realistic=100)
    """
    occ_str = str(occ).strip().lower()

    # Handle specific status cases
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
    pivot_df = df.pivot(index='participant_id', columns='col_name', values='z')
    pivot_df['overall_accuracy'] = df.groupby('participant_id')['acc'].mean()
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


def merge_parental_bias(final_df, main_dataset_path, stem_vars, verbal_vars):
    """
    Merges parental occupation scores and cognitive variables from the main dataset
    """
    if not os.path.exists(main_dataset_path):
        return final_df

    parents_df = pd.read_csv(main_dataset_path)

    # Calculate parental scores
    parents_df['mother_score'] = parents_df['mother_occupation'].apply(map_isco_score)
    parents_df['father_score'] = parents_df['father_occupation'].apply(map_isco_score)
    parents_df['parental_bias'] = parents_df[['mother_score', 'father_score']].mean(axis=1)

    # Prepare list of available columns (excluding ID to control its position)
    all_vars = list(set(stem_vars + verbal_vars + ['WASI_FSIQ', 'parental_bias']))
    available_cols = [c for c in all_vars if c in parents_df.columns]

    # Merge based on participant_id present in both dataframes
    merged = pd.merge(final_df, parents_df[['participant_id'] + available_cols], on='participant_id', how='left')

    return merged


def run_integrated_analysis():
    # --- A. Load LISAS data (Dynamic process) ---
    raw_list = []
    root_dir = '.\\trial_data'  # Ensure this points to your data folder

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.tsv'):
                # Extract subject ID from filename (e.g., sub-01)
                sub_id = file.split('_')[0]
                df_trial = pd.read_csv(os.path.join(root, file), sep='\t')

                try:
                    score = calculate_lisas(df_trial)
                    raw_list.append({'participant_id': sub_id, 'lisas': score})
                except:
                    continue

    # Group by participant_id without converting it to index (prevents 'subject' renaming issues)
    df_lisas = pd.DataFrame(raw_list).groupby('participant_id', as_index=False)['lisas'].mean()

    # --- B. Define Poles (STEM vs Verbal variables) ---
    stem_vars = [
        'lisas',
        'AWMA-S_VisuoSpatialSTM_StS',
        'AWMA-S_VisuoSpatialWM_StS',
        'CMAT_BasicCalc_Comp_Quotient',
        'KeyMath_Numeration_ScS',
        'KeyMath_Measurement_ScS',
        'KeyMath_ProblemSolving_ScS',
        'WASI_PIQ',
    ]

    verbal_vars = [
        'AWMA-S_VerbalSTM_StS',
        'AWMA-S_VerbalWM_StS',
        'CTOPP_PhonAwareness_Comp',
        'CTOPP_RapidNaming_Comp',
        'TOWRE_Total_StS',
        'WASI_VIQ',
    ]

    # --- C. Load and Merge Main Dataset ---
    final_df = merge_parental_bias(df_lisas, 'main_dataset.csv', stem_vars, verbal_vars)

    # --- D. Global Standardization Function ---
    # Calculate Z-scores for all relevant variables
    for var in stem_vars + verbal_vars + ['WASI_FSIQ', 'parental_bias']:
        if var in final_df.columns:
            # Invert lisas so higher score means better performance (matching other metrics)
            invert = True if 'lisas' in var.lower() else False
            final_df[f'{var}_z'] = standardize(final_df[var], invert=invert)

    # --- E. Calculate Indices and Cognitive Bias ---
    final_df['STEM_Index'] = final_df[[f'{v}_z' for v in stem_vars if f'{v}_z' in final_df.columns]].mean(axis=1)
    final_df['Verbal_Index'] = final_df[[f'{v}_z' for v in verbal_vars if f'{v}_z' in final_df.columns]].mean(axis=1)
    final_df['Child_Cognitive_Bias'] = final_df['STEM_Index'] - final_df['Verbal_Index']

    # --- F. Final Formatting and Output ---
    # Ensure participant_id is the leftmost column
    cols = ['participant_id'] + [c for c in final_df.columns if c != 'participant_id']
    final_df = final_df[cols]

    correlation = final_df['Child_Cognitive_Bias'].corr(final_df['parental_bias'])

    print(f"Success! The correlation is: {correlation:.3f}")
    print("\n--- Top 10 rows of final dataframe ---")
    print(final_df.head(10))

    # Save to CSV without the pandas internal index
    final_df.to_csv('analysis_results_full.csv', index=False)
    return final_df


def compute_bias_correlation(eda_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts relevant columns to numeric and computes Spearman correlation matrix.

    Notes:
      - 'school_type' is excluded because it's categorical text.
      - Spearman is used because education levels are ordinal.

    Args:
        eda_df (pd.DataFrame): EDA dataframe.

    Returns:
        pd.DataFrame: Spearman correlation matrix.
    """

    cols_to_numeric = ['parental_bias_z', 'STEM_Index', 'Verbal_Index', 'Child_Cognitive_Bias', 'WASI_FSIQ_z']
    for col in cols_to_numeric:
        if col in eda_df.columns:
            eda_df[col] = pd.to_numeric(eda_df[col], errors="coerce")

    return eda_df[cols_to_numeric].corr(method="pearson")
