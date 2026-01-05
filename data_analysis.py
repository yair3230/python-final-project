import logging as log

import pandas as pd
import numpy as np


def normalize_grades_column(series, base, limit):
    """
    Takes a scale of <base> to <limit> and turn it into a scale of 1 to 100
    :param series: list of grades
    :param base: The minimum possible score
    :param limit: The maximum possible score
    :return: Normalized series
    """
    if limit < 1:
        log.error('normalize_grades_column cannot get a limit smaller than 1')
    result = series - base
    result = result / limit
    result *= 100
    return result


def convert_strings_to_numbers(series):
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
