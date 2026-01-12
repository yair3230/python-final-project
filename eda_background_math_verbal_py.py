import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphs


def load_data(file_path):
    """
    Loads the dataset from the specified CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def get_variable_selection():
    """
    Defines the lists of background and math-related cognitive variables
    selected for the analysis.

    Returns:
        tuple: (background_vars, math_vars) lists.
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
        "WJ-III_MathFluency_StS",
        "WASI_VIQ_t2",
    ]

    return background_vars, math_vars


def create_eda_subset(df, background_vars, math_vars):
    """
    Creates a subset of the dataframe containing only the selected variables
    and displays basic information.

    Args:
        df (pd.DataFrame): The original dataframe.
        background_vars (list): List of background variable names.
        math_vars (list): List of math variable names.

    Returns:
        pd.DataFrame: The subsetted dataframe for EDA.
    """
    # Keep only relevant columns
    # Using .copy() to avoid SettingWithCopyWarning later
    eda_df = df[background_vars + math_vars].copy()

    # Display basic info
    print("\n--- EDA Subset Info ---")
    print(eda_df.info())
    print("\n--- EDA Subset Description ---")
    print(eda_df.describe())

    return eda_df


def preprocess_and_compute_correlations(eda_df):
    """
    Converts relevant columns to numeric types and computes the
    Spearman correlation matrix.

    Args:
        eda_df (pd.DataFrame): The EDA dataframe.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    # Convert background education columns to numeric (safe)
    # Using errors='coerce' to turn non-numeric values into NaN
    cols_to_numeric = ["mother_highest_grade", "father_highest_grade", "regular_classroom"]

    for col in cols_to_numeric:
        if col in eda_df.columns:
            eda_df[col] = pd.to_numeric(eda_df[col], errors="coerce")

    # Numeric columns for correlation (exclude school_type because it's categorical text)
    numeric_cols = [c for c in eda_df.columns if c != "school_type"]

    # Correlation matrix
    # Spearman is selected because variables like education levels are ordinal
    corr = eda_df[numeric_cols].corr(method="spearman")

    print("\nSpearman correlation matrix (numeric vars):")
    print(corr.round(2))

    return corr


def plot_correlation_heatmap(corr):
    """
    Plots a heatmap of the correlation matrix.

    Args:
        corr (pd.DataFrame): The correlation matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Spearman Correlation Matrix: Background vs. Math Abilities")
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the EDA workflow.
    """
    # 1. Load Data
    csv_path = "main_dataset.csv"
    df = load_data(csv_path)

    if df is not None:
        # 2. Define Variables
        bg_vars, math_vars = get_variable_selection()

        # 3. Create Subset
        eda_df = create_eda_subset(df, bg_vars, math_vars)

        # 4. Preprocess and Correlate
        correlation_matrix = preprocess_and_compute_correlations(eda_df)

        # 5. Visualizations via graphs.py
        # Correlation Matrix
        graphs.plot_correlation_heatmap(correlation_matrix)

        # (Optional) Add these if you paste the extra functions below into graphs.py:
        # graphs.plot_distributions(eda_df, math_vars)
        # graphs.plot_categorical_comparison(eda_df, "school_type", "CMAT_BasicCalc_Comp_Quotient")


if __name__ == "__main__":
    main()