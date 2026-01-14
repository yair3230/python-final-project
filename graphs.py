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


def scatter_point_graph(df, highest_grade_column, std_column):
    corr_val = df[highest_grade_column].corr(df[std_column])

    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=highest_grade_column,
        y=std_column,
        data=df,
        scatter_kws={'alpha': 0.6},  # Transparency for points
        line_kws={'color': 'red'},  # Distinct color for the correlation line
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return corr_val


def father_education_verbal(df):
    """
    Scatter plot: father's education level vs verbal IQ (VIQ).
    """
    corr_val = scatter_point_graph(df, "father_highest_grade", "total_verbal_score")

    plt.xlabel("Father's Education Level")
    plt.ylabel("Verbal STD score")
    plt.title(f"Correlation between Father's Grade and child's Verbal STD (r = {corr_val:.2f})")
    plt.show()


def father_education_math(df):
    """
    Plots scatter plots comparing father's education level with both
    verbal IQ (VIQ) and math ability (CMAT).
    """
    corr_val = scatter_point_graph(df, "father_highest_grade", "total_math_score")
    plt.xlabel("Father's Education Level")
    plt.ylabel("Math STD score")
    plt.title(f"Correlation between Father's Grade and child's Math STD (r = {corr_val:.2f})")
    plt.show()


def mother_education_verbal(df):
    """
    Scatter plot: mother's education level vs verbal IQ (VIQ).
    """
    corr_val = scatter_point_graph(df, "mother_highest_grade", "total_verbal_score")

    plt.xlabel("Mother's Education Level")
    plt.ylabel("Verbal STD score")
    plt.title(f"Correlation between Mother's Grade and child's Verbal STD (r = {corr_val:.2f})")
    plt.show()


def mother_education_math(df):
    """
    Plots scatter plots comparing mother's education level with both
    verbal IQ (VIQ) and math ability (CMAT).
    """
    corr_val = scatter_point_graph(df, "mother_highest_grade", "total_math_score")
    plt.xlabel("Mother's Education Level")
    plt.ylabel("Math STD score")
    plt.title(f"Correlation between Mother's Grade and child's Math STD (r = {corr_val:.2f})")
    plt.show()
