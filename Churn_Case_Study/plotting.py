import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def main_plots(main_df, plotting_df):
    """
    Plots all seven eda plots.

    Args:
        main_df: Pre-cleaned dataframe for use in initial correlation
                    matrix.
        plotting_df: Cleaned dataframe.

    Returns:
        Seven plots.
    """
    plot_correlation_matrix(main_df)
    ratings_plots(plotting_df)
    plot_avg_dist(plotting_df)
    pointplot(plotting_df)
    phone_barplot(plotting_df)
    city_barplot(plotting_df)
    plot_first_30(plotting_df)

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of feature variables.

    Args:
        df: DataFrame

    Returns:
        Plotted correlation matrix of df features.
    """
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                cbar_kws={'shrink':.5})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.show()

def ratings_plots(df):
    """
    Two Violin plots of average_rating_by_driver and avg_rating_of_driver

    Args:
        None
    Returns:
        Two violin plots.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(10,8))
    fig.suptitle('Distribution of Ratings', fontsize=20)
    ax1 = sns.violinplot(x='Active', y='avg_rating_by_driver', data=df, ax=axs[0], palette='coolwarm')
    ax1.set_ylabel('Average Rating by Driver')
    ax1.set_xlabel('Active Status')
    ax2 = sns.violinplot(x='Active', y='avg_rating_of_driver', data=df, ax=axs[1], palette='coolwarm')
    ax2.set_ylabel('Average Rating of Driver')
    ax2.set_xlabel('Active Status')
    plt.show()

def plot_avg_dist(df):
    """
    Violin plot of Active Status by Average Trip Distance.

    Args:
        None

    Returns:
        Violin Plot
    """
    fig, ax = plt.subplots(figsize=(10,8))
    sns.violinplot(x='Active', y='avg_dist', data=df, palette='coolwarm')
    plt.title('Active Status by Average Trip Distance ')
    ax.set_xlabel('Active Status')
    ax.set_ylabel('Average Trip Distance')
    plt.show()

def pointplot(df):
    """
    Pointplot of active status by luxury car user and average surge.

    Args:
        None
    Returns:
        Pointplot
    """
    fig, ax = plt.subplots(figsize=(12,8))
    sns.pointplot(x='Active', y='avg_surge', hue='luxury_car_user', data=df,
                    palette='coolwarm')
    leg_handles = ax.get_legend_handles_labels()[0]
    ax.legend(leg_handles, ['Non-Luxury User', 'Luxury User'],
                title='Luxury User', loc='best')
    plt.title('Active Status by Luxury User and Average Surge')
    ax.set_ylabel('Average Surge')
    ax.set_xlabel('Active Status')
    plt.show()

def city_barplot(df):
    """
    Barplot of active status by city.

    Args:
        None

    Returns:
        Barplot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.barplot(x='city', y='Active',data=df, palette='coolwarm')
    ax.set_ylabel('Active Status')
    ax.set_xlabel('City')
    plt.title('Active Status by City')
    plt.show()

def phone_barplot(df):
    """
    Barplot of active status by phone type.

    Args:
        None

    Returns:
        Barplot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.barplot(x='phone', y='Active', data=df,palette='coolwarm')
    ax.set_ylabel('Active Status')
    ax.set_xlabel('Phone Type')
    plt.title('Phone Type by Active Status')
    plt.show()

def plot_first_30(df):
    """
    Barplot of active status by number of trips in first 30 days.

    Args:
        None

    Returns:
        Barplot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.barplot(x='Active', y='trips_in_first_30_days', data=df, palette='coolwarm')
    ax.set_ylabel('No. of Trips in First 30 Days')
    ax.set_xlabel('Active Status')
    plt.title('Active Status by Trips in First 30 Days')
    plt.show()


if __name__=='__main__':
    main_df = pd.read_csv('data/churn_train.csv')
    plotting_df = pd.read_pickle('data/train_data.pkl')
    main_plots(main_df, plotting_df)
