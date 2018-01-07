from surprise import KNNBasic, KNNWithMeans, KNNBaseline
from surprise import Dataset
from surprise import GridSearch
from surprise import evaluate, print_perf
from surprise.dataset import Reader
import pandas as pd

def df_from_csv(filepath):
    """
    Load ratings dataframe from data/ratings.csv

    Args:
        filepath: path to ratings.csv file

    Returns:
        df: ratings dataframe
    """
    df = pd.read_csv(filepath)
    return df

def load_from_panda(df):
    """
    Transform pandas dataframe into surprise dataframe

    Args:
        df: pandas dataframe to transform

    Returns:
        data: surprise dataframe
    """
    #['userId', 'movieId', 'rating', 'timestamp']
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df, reader)
    return data

def surprise_gridsearch(param_grid, model, data):
    """
    Gridsearch on a surprise recommender model to extract the best parameters

    Args:
        param_grid: dictionary of model parameters and potential parameter values
        model: instantiated recommender model
        data: surprise dataframe

    Returns:
        The best score from the gridsearch as well as the accompanying best parameters
    """
    grid_search = GridSearch(model, param_grid, measures=['RMSE'], verbose=False)
    grid_search.evaluate(data)
    return grid_search.best_score['RMSE'], grid_search.best_params['RMSE']

def surprise_cross_validate(algo, data, *options):
    """
    3-Fold cross-validation on surprise recommendation model.

    Args:
        algo: instansitated recommender model
        data: surprise dataframe
        *options: additional parameter options to gridsearch on

    Returns:
        Mean RMSE of 3-Fold cross-validated model.
    """
    perf = evaluate(algo, data, measures=['RMSE'])
    print_perf(perf)

def get_Iu(uid):
    """
    Return the number of items rated by given user.

    Args:
        uid: The raw id of the user.
    Returns:
        The number of items rated by the user.
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError:  # user was not part of the trainset
        return 0

def get_Ui(iid):
    """
    Return the number of users that have rated given item

    Args:
        iid: The raw id of the item.
    Returns:
        The number of users that have rated the item.
    """

    try:
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:  # item was not part of the trainset
        return 0

if __name__=='__main__':
    # Load data
    ratings_df = df_from_csv('data/ratings.csv')
    data = load_from_panda(ratings_df[['userId', 'movieId', 'rating']])
    data.split(3)

    # Gridsearch KNNBasic
    param_grid = {'k': [22, 24, 26, 28, 30]}
    print(surprise_gridsearch(param_grid, KNNBasic, data))

    # Cross-Validate KNNBasic
    sim_options = {'name': 'MSD', 'user_based': False}
    algo = KNNBasic(k=26, sim_options=sim_options)
    surprise_cross_validate(algo, data, sim_options)

    # Gridsearch KNNWithMeans
    param_grid = {'k': [37, 38, 39, 40, 41, 42, 43]}
    print(surprise_gridsearch(param_grid, KNNWithMeans, data))

    # Cross-Validate KNNWithMeans
    sim_options = {'name': 'MSD', 'user_based': False}
    algo = KNNWithMeans(k=42, sim_options=sim_options)
    surprise_cross_validate(algo, data, sim_options)

    # Gridsearch KNNBaseline
    param_grid = {'k': [18, 19, 20, 21, 22]}
    print(surprise_gridsearch(param_grid, KNNBasic, data))

    # Cross-Validate KNNBaseline
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(k=19, sim_options=sim_options)
    surprise_cross_validate(algo, data, sim_options)

    # Predictions
    trainset = data.build_full_trainset()
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(k=19, sim_options=sim_options)
    algo.train(trainset)
    predictions = algo.test(trainset.build_testset())

    # Build Pandas DF of Ratings and Predictions
    df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
    df['#_of_Movies_Rated_By_User'] = df.uid.apply(get_Iu)
    df['#_of_Users_That_Rated_This_Movie'] = df.iid.apply(get_Ui)
    df['Error_in_Rating_Prediction'] = abs(df.est - df.rui)
    df.rename(columns={'uid':'User_ID', 'iid':'Movie_ID', 'rui':'User_Rating',
                        'est':'Predicted_Rating'}, inplace=True)

    # Query to get Predictions for User #1
    user_1 = df[df['User_ID']==1].sort_values(by='Predicted_Rating',
                ascending=False)[['User_ID', 'Movie_ID', 'User_Rating',
                'Predicted_Rating', 'Error_in_Rating_Prediction',
                '#_of_Users_That_Rated_This_Movie', '#_of_Movies_Rated_By_User']]
    print(user_1)

    # Get Predictions for all users
    df.to_csv('knnbaseline_predictions.csv')
