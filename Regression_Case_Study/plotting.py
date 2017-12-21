import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting Style
plt.style.use('fivethirtyeight')


def load_data():
    """
    Loads cleaned and unscaled data from pickle file.

    Args:
        None

    Returns:
        df: Cleaned and unscaled training data.
    """
    df = pd.read_pickle('data/plotting.pkl')
    return df

def distributions():
    """
    Plots the distribution of the SalePrice and Age features within the
    training data.

    Args:
        None

    Returns:
        Two side-by-side plots of the distribution of SalePrice and Age.
    """
    fig, axs = plt.subplots(nrows=2, figsize=(10,8))
    fig.suptitle('Distribution of Sale Price and Age', fontsize=20)
    ax1 = sns.distplot(df['SalePrice'], kde=True, bins=100, ax=axs[0],
                        color='steelblue')
    ax1.set_xlabel('SalePrice')
    ax1.set_yticks([])
    ax2 = sns.distplot(df['Age'], kde=True, bins=100, ax=axs[1],
                        color='steelblue')
    ax2.set_xlabel('Age')
    ax2.set_yticks([])
    plt.show()

def enclosure_barplot():
    """
    Plots the count of sales for each enclosure type in addition to the mean
    sale price for each enclosiure type.

    Args:
        None

    Returns:
        Two side-by-side barplots.
    """
    labels = df['Enclosure'].value_counts().index
    fig, axs = plt.subplots(ncols=2, figsize=(15,8))
    fig.suptitle('Enclosure Type', fontsize=20)
    ax1 = sns.countplot(x='Enclosure', data=df, ax=axs[0], palette="deep")
    ax1.set_xlabel('')
    ax1.set_ylabel('Count of Sales')
    ax1.set_xticklabels(labels, rotation=45, fontsize=10)
    ax2 = sns.barplot(x='Enclosure',y='SalePrice', data=df, ax=axs[1],
                        palette="deep")
    ax2.set_xlabel('')
    ax2.set_ylabel('Mean Sale Price')
    ax2.set_xticklabels(labels, rotation=45, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def product_group_barplot():
    """
    Plots the count of sales for each product group in addition to the mean
    sale price for each product group.

    Args:
        None

    Returns:
        Two side-by-side barplots.
    """
    labels = df['ProductGroupDesc'].value_counts().index
    fig, axs = plt.subplots(ncols=2, figsize=(15,8))
    fig.suptitle('Product Group', fontsize=20)
    ax1 = sns.countplot(x='ProductGroupDesc', data=df, ax=axs[0], palette="deep")
    ax1.set_xlabel('')
    ax1.set_ylabel('Count of Sales')
    ax1.set_xticklabels(labels, rotation=45, fontsize=10)
    ax2 = sns.barplot(x='ProductGroupDesc',y='SalePrice', data=df, ax=axs[1],
                        palette="deep")
    ax2.set_xlabel('')
    ax2.set_ylabel('Mean Sale Price')
    ax2.set_xticklabels(labels, rotation=45, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def state_plot(df):
    """
    Plots the mean sales price for each state.

    Args:
        None

    Returns:
        Barplot of mean sales price for each state.
    """
    fig, ax = plt.subplots(figsize=(6, 9))
    result = df.groupby(['state'])['SalePrice'].aggregate(np.mean).reset_index()\
                .sort_values('SalePrice', ascending=False)
    ax = sns.barplot(x='SalePrice', y='state', data=df, order=result['state'],
                        color='steelblue')
    ax.set_title('Mean Sale Price by State', fontsize=20)
    ax.set_xlabel('Mean Sale Price')
    ax.set_ylabel('State')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    df = load_data()
    distributions()
    product_group_barplot()
    enclosure_barplot()
    state_plot(df)
