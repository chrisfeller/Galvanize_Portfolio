import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clean_data import load_data

# Plotting Style
plt.style.use('fivethirtyeight')

def display_plots(df):
    """
    Calls all seven EDA plotting functions

    Args:
        df: cleaned training dataframe

    Returns:
        Seven EDA plots
    """
    plot_correlation_matrix(df)
    plot_countryfraud(df)
    plot_currencyfraud(df)
    sale_duration(df)
    user_age(df)
    user_type(df)
    delivery_method(df)

def plot_correlation_matrix(df):
    """
    Plot correlation matrix of feature variables

    Args:
        df: cleaned training dataframe

    Returns:
        Plotted correlation matrix of df features.
    """
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, cbar_kws={'shrink':.5})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

def top_corr(df):
    """
    Returns the absolute value of correlation for each feature to fraud sorted
    in descending order

    Args:
        df: cleaned training dataframe

    Returns:
        Table with the absolute value of correlation for each feature to fraud
        sorted in descending order.
    """
    corr = df.corr()
    corr = abs(corr['fraud']).sort_values(ascending=False)
    return corr

def plot_countryfraud(df):
    """
    Plot of number of fraud/non-fraud observations for each of the top four most-
    common countries in the training data (United States, Great Britian, Canada
    Australia)

    Args:
        df: cleaned training dataframe

    Returns:
        Four sub-barplots
    """
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10,8))
    fig.suptitle('Fraud By Top Four Most Common Countries', fontsize=20)
    ax1 = sns.barplot(x='country_US', y='fraud', data=df, ax=axs[0, 0],
                        palette='coolwarm', estimator=sum)
    ax1.set_ylim(0, 2000)
    ax1.set_xlabel('United States')
    ax1.set_ylabel('Observations')
    ax2 = sns.barplot(x='country_GB', y='fraud', data=df, ax=axs[0, 1],
                        palette='coolwarm', estimator=sum)
    ax2.set_ylim(0, 2000)
    ax2.set_xlabel('Great Britian')
    ax2.set_ylabel('Observations')
    ax3 = sns.barplot(x='country_CA', y='fraud', data=df, ax=axs[1, 0],
                        palette='coolwarm', estimator=sum)
    ax3.set_ylim(0, 2000)
    ax3.set_xlabel('Canada')
    ax3.set_ylabel('Observations')
    ax4 = sns.barplot(x='country_AU', y='fraud', data=df, ax=axs[1, 1],
                        palette='coolwarm', estimator=sum)
    ax4.set_ylim(0, 2000)
    ax4.set_xlabel('Australia')
    ax4.set_ylabel('Observations')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plot_currencyfraud(df):
    """
    Plot of number of fraud/non-fraud observations for each of the top three
    most-common currencies in the training data (United States Dollar,
    Great Britian Pound, Canadian Dollar)

    Args:
        df: cleaned training dataframe

    Returns:
        Three sub-barplots
    """
    fig, axs = plt.subplots(ncols=3, figsize=(14,5))
    fig.suptitle('Fraud By Top Three Most Common Currencies', fontsize=20)
    ax1 = sns.barplot(x='cur_USD', y='fraud', data=df, ax=axs[0],
                        palette='coolwarm', estimator=sum)
    ax1.set_ylim(0, 2000)
    ax1.set_xlabel('US Dollar')
    ax1.set_ylabel('Observations')
    ax2 = sns.barplot(x='cur_GBP', y='fraud', data=df, ax=axs[1],
                        palette='coolwarm', estimator=sum)
    ax2.set_ylim(0, 2000)
    ax2.set_xlabel('GB Pound')
    ax2.set_ylabel('Observations')
    ax3 = sns.barplot(x='cur_CAD', y='fraud', data=df, ax=axs[2],
                        palette='coolwarm', estimator=sum)
    ax3.set_ylim(0, 2000)
    ax3.set_xlabel('CAN Dollar')
    ax3.set_ylabel('Observations')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def sale_duration(df):
    """
    Violin plots of sale duration for fraud and non-fraud observations in the
    training data

    Args:
        df: cleaned training dataframe

    Returns:
        Two sub-violin plots
    """
    fig, axs = plt.subplots(ncols=2, figsize=(10,5))
    fig.suptitle('Fraud By Sale Durations', fontsize=20)
    ax1 = sns.violinplot(x='fraud', y='sale_duration', data=df, ax=axs[0],
                            palette='coolwarm')
    ax1.set_ylim(-100, 300)
    ax1.set_xlabel('Fraud')
    ax1.set_ylabel('Sale Duration')
    ax2 = sns.violinplot(x='fraud', y='sale_duration2', data=df, ax=axs[1],
                            palette='coolwarm')
    ax2.set_ylim(-100, 300)
    ax2.set_xlabel('Fraud')
    ax2.set_ylabel('Sale Duration 2')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def user_age(df):
    """
    Violin plot of user age for fraud and non-fraud transactions in the
    training data

    Args:
        df: cleaned training dataframe

    Returns:
        Violin Plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.violinplot(x='fraud', y='user_age', data=df, palette='coolwarm')
    ax.set_ylabel('User Age')
    ax.set_xlabel('Fraud')
    plt.title('Fraud by User Age')
    plt.tight_layout()
    plt.show()

def user_type(df):
    """
    Factorplot of probability of fraud for each user type in the training dataframe

    Args:
        df: cleaned training dataframe

    Returns:
        Factorplot
    """
    ax = sns.factorplot(x='user_type', y='fraud', data=df, palette='coolwarm_r',
                        kind='bar', ci=None, estimator=lambda x:
                                                    sum(x==1.0)*100.0/len(x))
    ax.set_axis_labels('User Type', 'Probability of Fraud')
    plt.title('Probability of Fraud by User Type')
    plt.tight_layout()
    plt.show()

def delivery_method(df):
    """
    Factorplot of probability of fraud for each delivery type in the training dataframe

    Args:
        df: cleaned training dataframe

    Returns:
        Factorplot
    """
    ax = sns.factorplot(x='delivery_method', y='fraud', data=df,
                        palette='coolwarm_r', kind='bar', ci=None,
                        estimator=lambda x: sum(x==1.0)*100.0/len(x))
    ax.set_axis_labels('Delivery Method', 'Probability of Fraud')
    plt.title('Probability of Fraud by Delivery Method')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # Load Cleaned Data
    X,y = load_data()
    df = pd.concat([X, y], axis=1)

    # Investigate Most-Correlated Features w/ Fraud
    print(top_corr(df))

    # Display Plots
    display_plots(df)
