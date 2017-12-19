import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train=True):
    """
    Load churn dataset from .csv file.

    Args:
        train: True if training data; False if testing data.

    Returns:
        Churn dataset
    """
    if train:
        df = pd.read_csv('data/churn_train.csv')
    else:
        df = pd.read_csv('data/churn_test.csv')
    return df

def to_date(df):
    """
    Convert last_trip_date and signup_date features from object to datetime

    Args:
        df: DataFrame

    Return:
        DataFrame with converted features
    """
    datecols = ['last_trip_date', 'signup_date']
    for col in datecols:
        df[col] = pd.to_datetime(df[col])
    return df

def lux_car_dummy(df):
    """
    Convert luxury_car_user to boolean 1, 0 instead of True, False

    Args:
        df: DataFrame

    Return:
        DataFrame with converted luxury_car_user feature
    """
    df['luxury_car_user'] = df['luxury_car_user'].map({True: 1, False: 0})
    return df

def create_active(df):
    active_date = pd.to_datetime('2014-06-01')
    df['Active'] = df.apply(lambda r: int(r.last_trip_date >= active_date) , axis=1)
    df.drop('last_trip_date', axis=1, inplace=True)
    pull_date = pd.to_datetime('2014-07-01')
    df['days_since_signup'] = pull_date - df['signup_date']
    df['days_since_signup'] = df['days_since_signup'].dt.days
    df.drop('signup_date', axis=1, inplace=True)
    return df

def handle_nulls(df):
    """
    Fill missing data in ratings features with mode of each respective feature.
    Drop null values in phone feature.

    Args:
        df: DataFrame with missing data

    Returns:
        DataFrame without missing data.
    """
    df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].median(), inplace=True)
    df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].median(), inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df

def make_dummies(df):
    """
    Make dummy columns for categorical variables.

    Args:
        df: DataFrame without dummy columns for categorical variables.

    Returns:
        df: Dataframe with dummy columns for categorical variables.
    """
    to_dummy = ['city', 'phone']
    for col in to_dummy:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)
        df.drop(col, inplace=True, axis=1)
    df.columns = [x.strip().replace(' ', '_').replace('\'', '') for x in df.columns]
    return df

def standardize_df(df):
    """
    Standardize Fare, Age, SibSp, Parch features.

    Args:
        df: DataFrame with non-standardized continuous features.

    Returns:
        df: DataFrame with staneardized continuous features.
    """
    to_standardize = ['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
                    'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct']
    scaler = StandardScaler()
    df[to_standardize] = scaler.fit_transform(df[to_standardize])
    return df

def main_clean(train=True):
    """
    Cleans data and saves the output as a pickle file.

    Args:
        train: True if training data, False if testing data.

    Returns:
        None: Saves pickle file in data directory.
    """
    df = load_data(train=True)
    to_date(df)
    create_active(df)
    lux_car_dummy(df)
    handle_nulls(df)
    standardize_df(df)
    df = make_dummies(df)
    # Pickle File
    if train==True:
        df.to_pickle('data/train_data.pkl')
    else:
        df.to_pickle('data/test_data.pkl')


if __name__=='__main__':
    main_clean(train=True)
