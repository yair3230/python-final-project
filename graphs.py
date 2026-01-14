import matplotlib.pyplot as plt
import seaborn as sns
from data_analysis import model_stage1, model_stage2
import statsmodels.api as sm
import scipy.stats as stats


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
        cbar_kws={"label": title},
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


def race_correlation(df):
    median_scores = df.groupby('race')['total_math_score'].median().sort_values(ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    colors = ['yellow', 'white', 'brown', 'red', 'black']
    median_scores.plot(kind='bar', color=colors, edgecolor='black')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero line')

    # Add labels and title
    plt.title('Median Total Math Score by race', fontsize=14)
    plt.xlabel('Race', fontsize=12)
    plt.ylabel('Median Math Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def income_correlation(df, parent):
    median_scores = df.groupby('race')[f'{parent}_income_level'].mean().sort_values(ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    # colors = ['yellow', 'white', 'black', 'red', 'brown']
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero line')

    # Add labels and title
    plt.title('Income level by race', fontsize=14)
    plt.xlabel('Race', fontsize=12)
    plt.ylabel('Income level', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return median_scores


def mother_income_correlation(df):
    median_scores = income_correlation(df, 'mother')

    # Create the bar chart
    colors = ['yellow', 'white', 'black', 'red', 'brown']
    median_scores.plot(kind='bar', color=colors, edgecolor='black')
    plt.xticks(rotation=45)
    plt.show()


def father_income_correlation(df):
    median_scores = income_correlation(df, 'father')

    # Create the bar chart
    colors = ['yellow', 'red', 'brown', 'white', 'black']
    median_scores.plot(kind='bar', color=colors, edgecolor='black')
    plt.xticks(rotation=45)
    plt.show()


def stage_A(df):

    result = model_stage1(df)
    print(result.summary())

    print("Stage A N:", int(result.nobs))


def stage_B(df):

    result = model_stage2(df)
    print(result.summary())


def hierarchial_model(df):
    stage1 = model_stage1(df)
    stage2 = model_stage2(df)
    delta_r2 = stage2.rsquared - stage1.rsquared
    print(f"ΔR² (Stage B – Stage A) = {delta_r2:.3f}")
    anova_results = sm.stats.anova_lm(stage1, stage2)
    print(anova_results)


def residuals_graph(df):
    stage2 = model_stage2(df)
    plt.figure(figsize=(6, 4))
    plt.scatter(stage2.fittedvalues, stage2.resid, alpha=0.7)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values (Stage B Model)")
    plt.tight_layout()
    plt.show()


def residuals_distribution_graph(df):
    stage2 = model_stage2(df)
    plt.figure(figsize=(6, 4))
    sns.histplot(stage2.resid, kde=True, bins=12)
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals (Stage B Model)")
    plt.tight_layout()
    plt.show()


def residuals_plot(df):
    stage2 = model_stage2(df)
    plt.figure(figsize=(5, 5))
    stats.probplot(stage2.resid, dist="norm", plot=plt)
    plt.title("Q–Q Plot of Residuals (Stage B Model)")
    plt.tight_layout()
    plt.show()


def influence_plot(df):
    stage2 = model_stage2(df)
    influence = stage2.get_influence()
    cooks = influence.cooks_distance[0]

    plt.figure(figsize=(6, 4))
    plt.stem(cooks, markerfmt=".")
    plt.axhline(4 / len(cooks), color="red", linestyle="--")
    plt.xlabel("Observation Index")
    plt.ylabel("Cook's Distance")
    plt.title("Influential Observations (Stage B Model)")
    plt.tight_layout()
    plt.show()

def bias_vs_stem(df):
    sns.regplot(
        x="parental_bias_z",
        y="STEM_Index",
        data=df,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"}
    )
    plt.title("Parental bias vs STEM ability")
    plt.tight_layout()
    plt.show()

def bias_vs_verbal(df):
    sns.regplot(
        x="parental_bias_z",
        y="Verbal_Index",
        data=df,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"}
    )
    plt.title("Parental bias vs Verbal ability")
    plt.tight_layout()
    plt.show()

def bias_vs_academic(df):
    sns.regplot(
        x="parental_bias_z",
        y="Child_Cognitive_Bias",
        data=df,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"}
    )
    plt.title("Parental bias vs Academic orientation (STEM − Verbal)")
    plt.tight_layout()
    plt.show()
