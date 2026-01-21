import os
import pandas as pd

from data_analysis import fetch_total_math_score, fetch_total_verbal_score, fetch_total_memory_score, \
    calc_parents_income_level


def load_main_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(file_path)
    df['total_math_score'] = fetch_total_math_score(df)
    df['total_verbal_score'] = fetch_total_verbal_score(df)
    df['total_memory_score'] = fetch_total_memory_score(df)
    df = calc_parents_income_level(df)
    return df
