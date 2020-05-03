import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from cleaning import create_train_holdout, clean_train

# Plotting Style
plt.style.use('fivethirtyeight')

def VORP_histogram(train):
    """
    Histogram of the distribution of the target variable NBA_VORP prior to the
    imputation of null values.

    Args:
        train: cleaned training dataframe

    Returns:
        Histogram
    """
    fig, ax = plt.subplots(figsize=(12,5))
    sns.distplot(train[train['NBA_MP'].notnull()]['NBA_VORP'], color='royalblue')
    plt.title('Value Over Replacement Player (VORP)')
    ax.set_xlabel('VORP \n \n Skewness: {0:0.04f} Kurtosis: {1: 0.04f}'.format(
                  train[train['NBA_MP'].notnull()]['NBA_VORP'].skew(),
                  train[train['NBA_MP'].notnull()]['NBA_VORP'].kurt()),
                  fontsize=13)
    plt.tight_layout()
    plt.show()

def VORP_probability_plot(train):
    """
    Probability plot of the target variable NBA_VORP before the imputation of
    null values.

    Args:
        train: cleaned training dataframe

    Returns:
        Probability Plot
    """
    stats.probplot(train[train['NBA_MP'].notnull()]['NBA_VORP'], plot=plt)
    plt.tight_layout()
    plt.show()

def NBA_VORP_distribution(train):
    """
    Violin Plots of the distribution of target variable VORP for
    raw and cleaned training data. The cleaned training data imputes -2.0 for
    players who never made it into the second-year of their NBA career and the
    minimum NBA_VORP (-1.4) for players who played fewer than 50 minutes in
    their second NBA season.

    Args:
        train: raw training dataframe
        cleaned_train: cleaned training dataframe
    Returns:
        Violin Plots of target variable VORP
    """
    fig, axs = plt.subplots(nrows=2, figsize=(15,8))
    fig.suptitle('Distribution of NBA Value Over Replacement Player',
                 fontsize=20)
    ax1 = sns.violinplot(x='NBA_VORP', data=train[train['NBA_MP'].notnull()],
                         ax=axs[0], color='royalblue', cut=0)
    ax2 = sns.violinplot(x='NBA_VORP', data=train, ax=axs[1], color='royalblue',
                         cut=0)
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax1.set_title('Raw Data')
    ax2.set_title('Clean Data')
    ax1.set_xlim((-3, 7))
    ax2.set_xlim((-3, 7))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def NBA_VORP_Seasons(train):
    """
    Violin plots of the target variable VORP broken out by draft class (2012-15).

    Args:
        train: cleaned training dataframe

    Returns:
        Violin Plots
    """
    fig, ax = plt.subplots(figsize=(12,8))
    sns.violinplot(x='last_college_season', y='NBA_VORP',
                   data=train, palette='coolwarm',
                   order=['2011-12', '2012-13', '2013-14', '2014-15'], cut=0)
    plt.title('Distribution of VORP Across Draft Classes')
    ax.set_xlabel('Draft Classes')
    ax.set_ylabel('VORP')
    plt.tight_layout()
    plt.show()

def NBA_VORP_Age(train):
    """
    Violin plots of the target variable VORP broken out by the age of the player
    at the time of the draft.

    Args:
        train: cleaned training dataframe

    Returns:
        Violin Plots
    """
    fig, ax = plt.subplots(figsize=(12,8))
    train['age_at_draft'] = train['age_at_draft'].round()
    sns.violinplot(x='age_at_draft', y='NBA_VORP',
                   data=train, palette='coolwarm', cut=0)
    plt.title('Distribution of VORP Across Draft Age')
    ax.set_xlabel('Age of Draft Pick')
    ax.set_ylabel('VORP')
    plt.tight_layout()
    plt.show()

def subjective_correlation_matrix(train):
    """
    Plot correlation matrix of all feature variables I subjectively deemed
    as being highly predictive.

    Args:
        train: cleaned training dataframe

    Returns:
        Plotted correlation matrix of all feature variables in the training data
    """
    train = pd.concat([train.loc[:, ['age_at_draft', '3P_per100', '3PA_per100',
                                  '3P%_per100', 'FTA_per100', 'FT%_per100',
                                  'TRB_per100', 'STL_per100', 'PTS_per100',
                                  'college_TS%', 'college_eFG%', 'college_3PAr',
                                  'college_FTr', 'college_ORB%', 'college_DRB%',
                                  'college_TRB%', 'college_STL%', 'college_USG%',
                                  '%_shots_at_rim', 'FG%_at_rim', '%_shots_2pt_J']],
                                  train.iloc[:, -1]], axis=1)
    corr = train.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(13,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, cbar_kws={'shrink':.5})
    plt.title('''Correlation Matrix of Subjectively Selected Highly-Predictive
              Features''', fontsize=16)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    plt.show()

def plot_top_correlation_matrix(train):
    """
    Plot correlation matrix of top-15 features that are most correlated with
    NBA_VORP

    Args:
        df: cleaned training dataframe

    Returns:
        Plotted correlation matrix of top-15 features that are most correlated
        with NBA_VORP
    """
    train = pd.concat([train.iloc[:, 0:65], train.iloc[:, -1]], axis=1)
    corr = train.corr()
    corr = abs(corr['NBA_VORP']).sort_values(ascending=False)
    corr = corr.head(16).index
    train = pd.concat([train.loc[:, corr], train.iloc[:, -2]], axis=1)
    corr = train.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, annot=True, fmt='.2f', cbar_kws={'shrink':.5},
                annot_kws={"size": 9})
    plt.title('Top 15 Correlation Matrix')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

def plot_total_correlation_matrix(train):
    """
    Plot correlation matrix of all feature variables in the training data

    Args:
        train: cleaned training dataframe

    Returns:
        Plotted correlation matrix of all feature variables in the training data
    """
    train = pd.concat([train.iloc[:, 0:65], train.iloc[:, -1]], axis=1)
    corr = train.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(13,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, cbar_kws={'shrink':.5})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    plt.show()

def plot_per100_correlation_matrix(train):
    """
    Plot correlation matrix of per 100 possession variables

    Args:
        train: cleaned training dataframe

    Returns:
        Plotted correlation matrix of per 100 possession variables
    """
    train = pd.concat([train.iloc[:, 11:35], train.iloc[:, -1]], axis=1)
    corr = train.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, cbar_kws={'shrink':.5})
    plt.title('Per 100 Possession Correlation Matrix')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

def plot_advance_correlation_matrix(train):
    """
    Plot correlation matrix of advance variables

    Args:
        train: cleaned training dataframe

    Returns:
        Plotted correlation matrix of advance variables
    """
    train = pd.concat([train.iloc[:, 35:56], train.iloc[:, -1]], axis=1)
    corr = train.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, cbar_kws={'shrink':.5})
    plt.title('Advance Correlation Matrix')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

def plot_shooting_correlation_matrix(train):
    """
    Plot correlation matrix of shot distribution variables

    Args:
        train: cleaned training dataframe

    Returns:
        Plotted correlation matrix of shot distribution variables
    """
    train = pd.concat([train.iloc[:, 56:65], train.iloc[:, -1]], axis=1)
    corr = train.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(12,8))
    cmap = sns.color_palette('coolwarm')
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5,
                yticklabels=True, annot=True, fmt='.2f', cbar_kws={'shrink':.5},
                annot_kws={"size": 12})
    plt.title('Shot Distribution Correlation Matrix')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.show()

def scatter_plots(train):
    """
    Plot scatter matrix of the top nine most correlated features with NBA_VORP.
    Args:
        df: training dataframe
    Returns:
        Plotted scatter matrix of top nine most correlated features with NBA_VORP.
    """
    fig, axs = plt.subplots(3, 3, figsize=(14, 15))
    fig.suptitle('Top Nine Most Correlated Features w/ VORP', fontsize=20)
    ax1 = sns.regplot(x='college_BPM', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[0, 0], color='steelblue', scatter_kws={"s": 9})
    ax1.set_xlabel('College Box Plus-Minus', size=10)
    ax1.set_ylabel('VORP', size=10)
    ax1.set_ylim(-2, 5.5)
    ax2 = sns.regplot(x='college_DBPM', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[0, 1], color='steelblue', scatter_kws={"s": 9})
    ax2.set_xlabel('College Defensive Box Plus-Minus', size=10)
    ax2.set_ylabel('VORP', size=10)
    ax2.set_ylim(-2, 5.5)
    ax3 = sns.regplot(x='age_at_draft', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[0, 2], color='steelblue', scatter_kws={"s": 9})
    ax3.set_xlabel('Age at Draft', size=10)
    ax3.set_ylabel('VORP', size=10)
    ax3.set_ylim(-2, 5.5)
    ax4 = sns.regplot(x='DRtg_per100', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[1, 0], color='steelblue', scatter_kws={"s": 9})
    ax4.set_xlabel('College Defensive Rating', size=10)
    ax4.set_ylabel('VORP', size=10)
    ax4.set_ylim(-2, 5.5)
    ax5 = sns.regplot(x='BLK_per100', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[1, 1], color='steelblue', scatter_kws={"s": 9})
    ax5.set_xlabel('Blocks Per 100 Possessions', size=10)
    ax5.set_ylabel('VORP', size=10)
    ax5.set_ylim(-2, 5.5)
    ax6 = sns.regplot(x='college_BLK%', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[1, 2], color='steelblue', scatter_kws={"s": 9})
    ax6.set_xlabel('College Block Percentage', size=10)
    ax6.set_ylabel('VORP', size=10)
    ax6.set_ylim(-2, 5.5)
    ax7 = sns.regplot(x='college_WS_40', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[2, 0], color='steelblue', scatter_kws={"s": 9})
    ax7.set_xlabel('College WS/40', size=10)
    ax7.set_ylabel('VORP', size=10)
    ax7.set_ylim(-2, 5.5)
    ax8 = sns.regplot(x='college_PER', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[2, 1], color='steelblue', scatter_kws={"s": 9})
    ax8.set_xlabel('College Player Efficiency Rating', size=10)
    ax8.set_ylabel('VORP', size=10)
    ax8.set_ylim(-2, 5.5)
    ax9 = sns.regplot(x='college_DWS', y='NBA_VORP', data=train, fit_reg=False,
                        ax=axs[2, 2], color='steelblue', scatter_kws={"s": 9})
    ax9.set_xlabel('College Defensive Win Shares', size=10)
    ax9.set_ylabel('VORP', size=10)
    ax9.set_ylim(-2, 5.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    fig.savefig('../images/scatter_plots.png')

def draft_age_distribution(train):
    """
    Violin Plot of the distribution of player age at the time of the draft.

    Args:
        train: cleaned training dataframe
    Returns:
        Violin Plot of player age at the time of the draft.
    """
    fig, ax = plt.subplots(figsize=(15,5))
    sns.violinplot(x='age_at_draft', data=train, color='royalblue', cut=0)
    plt.title('Distribution of Age at Time of Draft')
    ax.set_xlabel('Age in Years')
    ax.set_xlim(17, 29)
    plt.tight_layout()
    plt.show()

def height_distribution(train):
    """
    Violin Plot of the distribution of player height at the time of the draft.

    Args:
        train: cleaned training dataframe
    Returns:
        Violin Plot of player height at the time of the draft.
    """
    fig, ax = plt.subplots(figsize=(15,5))
    sns.violinplot(x='height', data=train, color='royalblue', cut=0)
    plt.title('Distribution of Height')
    ax.set_xlabel('Height (Inches)')
    ax.set_xlim(68, 87)
    plt.tight_layout()
    plt.show()

def weight_distribution(train):
    """
    Violin Plot of the distribution of player weight at the time of the draft.

    Args:
        train: cleaned training dataframe
    Returns:
        Violin Plot of player weight at the time of the draft.
    """
    fig, ax = plt.subplots(figsize=(15,5))
    sns.violinplot(x='weight', data=train, color='royalblue', cut=0)
    plt.title('Distribution of Weight')
    ax.set_xlabel('Weight (Pounds)')
    ax.set_xlim(155, 285)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_validation, y_pred):
    """
    Scatter plot of the residuals of the model on the validation data.

    Args:
        y_validation: True NBA_VORP values for the validation data.
        y_pred: PRedicted NBA_VORP values for the validation data.

    Returns:
        Scatter plot of validation data residuals from final model.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.residplot(y_validation, y_pred, color='royalblue')
    ax.set_xlim(-2.5, 5.5)
    ax.set_xlabel('VORP')
    ax.set_ylabel('Residuals')
    plt.title('Residuals')
    plt.tight_layout()
    plt.show()

def feature_importance_plot(features, feature_importance):
    """
    Barplot of most important 25 features in final model.

    Args:
        features: list of column names in training data.
        feature_importances: list or vector of feature importance values from
        final model.

    Returns:
        Barplot
    """
    df = pd.DataFrame(feature_importance, features ).reset_index()
    df.columns = ['feature', 'importance']
    df.sort_values(by='importance', ascending=False, inplace=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(x='feature', y='importance', data=df.head(25), palette='coolwarm_r')
    ax.set_xlabel('Features')
    plt.xticks(rotation=90, fontsize=10)
    ax.set_ylabel('Importance')
    plt.title('Feature Importance (Top 25)')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    # Load Data
    meta_data = pd.read_csv('../data/meta_data.csv')
    college_per100 = pd.read_csv('../data/college_per100.csv')
    college_advance = pd.read_csv('../data/college_advance.csv')
    college_shooting = pd.read_csv('../data/college_shooting.csv')
    nba_advance = pd.read_csv('../data/nba_advance.csv')

    # Create Training Data
    train, _ = create_train_holdout(meta_data, college_per100, college_advance, college_shooting, nba_advance)

    # Clean Train DataFrame
    train = clean_train(train)

    # Histogram of NBA VORP
    VORP_histogram(train)

    # Probability Plot
    VORP_probability_plot(train)

    # Plot Distributiuon of Clean and Unclean VORP
    NBA_VORP_distribution(train)

    # VORP By Season
    NBA_VORP_Seasons(train)

    # VORP By Draft Age
    NBA_VORP_Age(train)

    # Plot Subjective Correlation Matrix
    subjective_correlation_matrix(train)

    # Correlation Matrix w/ 15-Most Correlated Features
    plot_top_correlation_matrix(train)

    # Total Correlation Matrix
    plot_total_correlation_matrix(train)

    # Per 100 Possession Correlation Matrix
    plot_per100_correlation_matrix(train)

    # College Advance Correlation Matrix
    plot_advance_correlation_matrix(train)

    # College Shooting Distribution Correlation Matrix
    plot_shooting_correlation_matrix(train)

    # Scatter Plots of Top Six Most Correlated Features
    scatter_plots(train)

    # Distribution of Age
    draft_age_distribution(train)

    # Distribution of Height
    height_distribution(train)

    # Distribution of Weight
    weight_distribution(train)
