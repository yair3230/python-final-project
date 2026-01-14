import pandas as pd
from consts import BACKGROUND_VARS, STD_VARS


def create_eda_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a subset of the dataframe containing only the selected variables.

    Args:
        df (pd.DataFrame): Original dataframe.

    Returns:
        pd.DataFrame: Subset dataframe.
    """
    return df[BACKGROUND_VARS + STD_VARS].copy()


def preprocess_and_compute_correlations(eda_df: pd.DataFrame) -> pd.DataFrame:
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
    cols_to_numeric = ["mother_highest_grade", "father_highest_grade", "regular_classroom"]
    for col in cols_to_numeric:
        if col in eda_df.columns:
            eda_df[col] = pd.to_numeric(eda_df[col], errors="coerce")

    # Rename long column names, so we could see the entire spearman correlation in a single line
    # (prevent the line break)
    eda_df = eda_df.rename(
        columns={
            'total_math_score': 'math',
            'total_verbal_score': 'verbal',
            'mother_highest_grade': 'mother_HG',
            'father_highest_grade': 'father_HG',
        }
    )
    numeric_cols = [c for c in eda_df.columns if c != "school_type"]
    return eda_df[numeric_cols].corr(method="spearman")


def verbal_correlation(eda_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlation between parental education and Verbal IQ (VIQ).

    Args:
        eda_df (pd.DataFrame): EDA dataframe.

    Returns:
        pd.DataFrame: Spearman correlation matrix for the 3 variables.
    """
    subset = eda_df[["mother_highest_grade", "father_highest_grade", 'total_verbal_score']].copy()
    subset = subset.rename(
        columns={
            'total_verbal_score': 'verbal',
            'mother_highest_grade': 'mother_HG',
            'father_highest_grade': 'father_HG',
        }
    )
    subset["mother_HG"] = pd.to_numeric(subset["mother_HG"], errors="coerce")
    subset["father_HG"] = pd.to_numeric(subset["father_HG"], errors="coerce")
    return subset.corr(method="spearman")
