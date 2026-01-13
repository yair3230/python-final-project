import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from graphs import plot_correlation_heatmap


def load_data(file_path):
    """
    Loads the dataset and displays the first few rows.
    Corresponds to Cell 2 of the notebook.
    """
    if not os.path.exists(file_path):
        # Fallback for local testing if the specific relative path doesn't exist
        print(f"Warning: File not found at {file_path}. Please check the path.")
        return None

    df = pd.read_csv(file_path)
    print("--- Data Loaded ---")
    print(df.head())
    return df


def define_variable_lists():
    """
    Defines the lists of background and math-related variables.
    Corresponds to Cell 4 of the notebook.
    """
    # --- Select background (socio-educational) variables ---
    background_vars = [
        "mother_highest_grade",
        "father_highest_grade",
        "school_type",
        "regular_classroom"
    ]

    # --- Select math-related cognitive variables ---
    math_vars = [
        "CMAT_BasicCalc_Comp_Quotient",
        "KeyMath_Numeration_ScS",
        "KeyMath_Measurement_ScS",
        "KeyMath_ProblemSolving_ScS",
        "WJ-III_MathFluency_StS"
    ]

    return background_vars, math_vars


def prepare_eda_dataframe(df, background_vars, math_vars):
    """
    Filters the dataframe to keep relevant columns and displays basic stats.
    Corresponds to Cell 5 of the notebook.
    """
    # Keep only relevant columns
    eda_df = df[background_vars + math_vars].copy()

    # Display basic info
    print("\n--- EDA DataFrame Info ---")
    print(eda_df.info())
    print("\n--- EDA DataFrame Description ---")
    print(eda_df.describe())

    return eda_df


def calculate_correlations(eda_df):
    """
    Preprocesses data (numeric conversion) and computes Spearman correlations.
    Corresponds to Cell 7 of the notebook.
    """
    # Convert background education columns to numeric (safe)
    for col in ["mother_highest_grade", "father_highest_grade", "regular_classroom"]:
        eda_df[col] = pd.to_numeric(eda_df[col], errors="coerce")

    # Numeric columns for correlation (exclude school_type because it's categorical text)
    numeric_cols = [c for c in eda_df.columns if c != "school_type"]

    # Correlation matrix
    corr = eda_df[numeric_cols].corr(method="spearman")  # spearman is robust for ordinal scales

    print("\n--- Spearman correlation matrix (numeric vars) ---")
    print(corr.round(2))

    return corr


def main():
    # 1. Load Data
    # Note: Adjust path as necessary based on your file structure
    file_path = "../main_dataset.csv"
    df = load_data(file_path)

    if df is not None:
        # 2. Define Variables
        bg_vars, math_vars = define_variable_lists()

        # 3. Prepare Data
        eda_df = prepare_eda_dataframe(df, bg_vars, math_vars)

        # 4. Calculate Correlations
        correlation_matrix = calculate_correlations(eda_df)

        # 5. Plot Heatmap
        plot_correlation_heatmap(correlation_matrix)


if __name__ == "__main__":
    main()