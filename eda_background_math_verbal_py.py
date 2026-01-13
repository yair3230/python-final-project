import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    return pd.read_csv(file_path)


def get_variable_selection():
    """
    Defines the lists of background and math-related cognitive variables.

    Returns:
        tuple: (background_vars, math_vars)
    """
    background_vars = [
        "mother_highest_grade",
        "father_highest_grade",
        "school_type",
        "regular_classroom",
    ]

    math_vars = [
        "CMAT_BasicCalc_Comp_Quotient",
        "KeyMath_Numeration_ScS",
        "KeyMath_Measurement_ScS",
        "KeyMath_ProblemSolving_ScS",
        "WJ-III_MathFluency_StS",
        "WASI_VIQ_t2",
    ]

    return background_vars, math_vars


def create_eda_subset(df: pd.DataFrame, background_vars, math_vars) -> pd.DataFrame:
    """
    Creates a subset of the dataframe containing only the selected variables.

    Args:
        df (pd.DataFrame): Original dataframe.
        background_vars (list): Background variable names.
        math_vars (list): Math/cognitive variable names.

    Returns:
        pd.DataFrame: Subset dataframe.
    """
    return df[background_vars + math_vars].copy()


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
    subset = eda_df[["mother_highest_grade", "father_highest_grade", "WASI_VIQ_t2"]].copy()
    subset["mother_highest_grade"] = pd.to_numeric(subset["mother_highest_grade"], errors="coerce")
    subset["father_highest_grade"] = pd.to_numeric(subset["father_highest_grade"], errors="coerce")
    return subset.corr(method="spearman")
