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
        cbar_kws={"label": "Spearman correlation"}
    )

    plt.title(title, fontsize=15, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_distributions(df, variables, cols=3):
    """
    Plots histograms with KDE for a list of variables.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - variables (list): List of column names to plot.
    - cols (int): Number of columns in the subplot grid.
    """
    rows = (len(variables) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        if var in df.columns:
            sns.histplot(data=df, x=var, kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Distribution of {var}')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Frequency')
        else:
            axes[i].set_visible(False)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_categorical_comparison(df, cat_var, num_var):
    """
    Plots a boxplot comparing a numerical variable across categories.

    Parameters:
    - df (pd.DataFrame): The dataframe.
    - cat_var (str): The categorical variable (x-axis).
    - num_var (str): The numerical variable (y-axis).
    """
    if cat_var not in df.columns or num_var not in df.columns:
        print(f"Error: Columns {cat_var} or {num_var} not found.")
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=cat_var, y=num_var, palette="Set2")
    sns.stripplot(data=df, x=cat_var, y=num_var, color='black', alpha=0.5, jitter=True)

    plt.title(f'{num_var} by {cat_var}', fontsize=14)
    plt.xlabel(cat_var)
    plt.ylabel(num_var)
    plt.tight_layout()
    plt.show()