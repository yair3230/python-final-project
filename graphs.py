import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(corr_matrix, title="Spearman Correlations"):
    """
    Visualizes a correlation matrix using a heatmap.

    Parameters:
    - corr_matrix (pd.DataFrame): The correlation matrix to plot.
    - title (str): The title of the plot.
    """
    plt.figure(figsize=(12, 9))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 10},
        cbar_kws={"label": "Spearman correlation"},
    )

    plt.title(title, fontsize=15, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def primal_education_verbal_scatter_plot(df):
    """
    Scatter plot: father's education level vs verbal IQ (VIQ).
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["father_highest_grade"],
        df["WASI_VIQ_t2"],
        alpha=0.7,
    )
    plt.xlabel("Father's Education Level")
    plt.ylabel("Verbal IQ (VIQ)")
    plt.title("Parental Education and Verbal Ability")
    plt.tight_layout()
    plt.show()


def verbal_scatter_plot(df):
    """
    Alias for the same plot (kept for backward compatibility).
    """
    primal_education_verbal_scatter_plot(df)


def father_education_math_vs_verbal_scatter(df):
    """
    Plots scatter plots comparing father's education level with both
    verbal IQ (VIQ) and math ability (CMAT).
    """
    plt.figure(figsize=(10, 4))

    # Verbal
    plt.subplot(1, 2, 1)
    plt.scatter(
        df["father_highest_grade"],
        df["WASI_VIQ_t2"],
        alpha=0.7,
    )
    plt.xlabel("Father's Education")
    plt.ylabel("VIQ")
    plt.title("Education vs Verbal Ability")

    # Math
    plt.subplot(1, 2, 2)
    plt.scatter(
        df["father_highest_grade"],
        df["CMAT_BasicCalc_Comp_Quotient"],
        alpha=0.7,
    )
    plt.xlabel("Father's Education")
    plt.ylabel("Math Ability (CMAT)")
    plt.title("Education vs Math Ability")

    plt.tight_layout()
    plt.show()
