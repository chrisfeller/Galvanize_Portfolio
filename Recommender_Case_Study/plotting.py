import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def error_violinplot(df1, df2):
    """
    Violin Plot of user rating prediction error for Mean of Means
    and KNNBaseline recommenders.

    Args:
        df1: mean of means prediction dataframe
        df2: knnbaseline prediction dataframe
    Returns:
        Violin Plots
    """
    fig, axs = plt.subplots(ncols=2, figsize=(18,8))
    fig.suptitle('Prediction Errors', fontsize=20)
    ax1 = sns.violinplot(x='User_Rating', y='Error_in_Rating_Prediction',data=df1, ax=axs[0])
    ax2 = sns.violinplot(x='User_Rating', y='Error_in_Rating_Prediction',data=df2, ax=axs[1])
    ax1.set(ylim=(-0.5,5))
    ax2.set(ylim=(-0.5,5))
    ax1.set_ylabel('Error in Prediction')
    ax2.set_ylabel('Error in Prediction')
    ax1.set_xlabel('Rating')
    ax2.set_xlabel('Rating')
    ax1.set_title('Mean of Means')
    ax2.set_title('KNNBaseline')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

if __name__=='__main__':
    mom = pd.read_csv('data/mean_of_means_predictions.csv')
    knn = pd.read_csv('data/knnbaseline_predictions.csv')

    error_violinplot(mom, knn)
