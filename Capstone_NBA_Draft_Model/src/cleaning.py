import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def player_names(df):
    """
    Extract all player names drafted since 2011. Remove all apostrphes,
    lowercase, and split each name into first_name and last_name.

    Args:
        df: pandas DataFrame, which includes all uncleaned player names.

    Return:
        df: pandas DataFrame with cleaned player names.
    """
    df = filter_college(df)
    df[['first_name', 'last_name', 'suffix_name']] = df['Player'].str.split(expand=True)
    df = df[['first_name', 'last_name']]
    df['first_name'] = df['first_name'].str.lower()
    df['first_name'] = df['first_name'].str.replace("'", "")
    df['first_name'] = df['first_name'].str.replace(".", "")
    df['last_name'] = df['last_name'].str.lower()
    df['last_name'] = df['last_name'].str.replace("'", "")
    df = player_names_edgecase(df)
    df = sports_ref_index(df)
    df.reset_index(inplace=True, drop=True)
    return df

def player_names_edgecase(df):
    """
    Handle edge cases involving naming conventions (i.e. edrice adebayo
    instaed of bam adebayo).

    Args:
        df: pandas DataFrame with uncleaned player names.

    Returns:
        df: pandas DataFrame with cleaned player names.
    """
    df.iloc[7, 1] = 'smithjr'
    df.iloc[12, 0] = 'edrice'
    df.iloc[79, 1] = 'zimmermanjr'
    df.iloc[88, 0] = 'kahlil'
    df.iloc[165, 1] = 'harris-'
    df.iloc[182, 1] = 'devyn-marble'
    df.iloc[205, 1] = 'hardaway-jr'
    df.iloc[212, 1] = 'rice-jr'
    df.iloc[262, 0] = 'jeffery'
    df.iloc[263, 1] = 'james-'
    df.drop(df.index[220], inplace=True) # Remove Ricky Ledo
    return df

def sports_ref_index(df):
    """
    Handle edge cases involving duplicate names in college player database.

    Args:
        df: pandas DataFrame with incorrect Sports-Reference indices for
        multiple players.

    Returns:
        df: pandas DataFrame with correct Sports-Reference incides for all
        players.
    """
    df['sports_ref'] = 1
    for i in [3, 27, 40, 81, 103, 116, 126, 171, 205, 212, 235, 253, 273, 283, 284, 301]:
        df.iloc[i, 2] = 2
    for i in [131, 134, 162]:
        df.iloc[i, 2] = 3
    for i in [13, 213, 271]:
        df.iloc[i, 2] = 4
    for i in [122, 231]:
        df.iloc[i, 2] = 5
    return df

def filter_college(df):
    """
    Filter draft picks who played in college.

    Args:
        df: pandas DataFrame with all players drafted into the NBA since 2011.

    Returns:
        df: pandas DataFrame with only players that played in college and were
        drafted into the NBA since 2011.
    """
    df = df[df['College'].notnull()]
    return df

def calculate_age(row):
    """
    Calculate the age of each player, in years, on the night they were draft.

    Args:
        row: row of pandas DataFrame that corresponds to a drafted player.

    Returns:
        row: row of pandas DataFrame that corresponds to a drafted player.
    """
    if row['Season'] == '2010-11':
        row['Age'] = (pd.to_datetime('06/23/2011') - row['birthday'])/ np.timedelta64(1, 'M')
    elif row['Season'] == '2011-12':
        row['Age'] = (pd.to_datetime('06/28/2012') - row['birthday'])/ np.timedelta64(1, 'M')
    elif row['Season'] == '2012-13':
        row['Age'] = (pd.to_datetime('06/27/2013') - row['birthday'])/ np.timedelta64(1, 'M')
    elif row['Season'] == '2013-14':
        row['Age'] = (pd.to_datetime('06/26/2014') - row['birthday'])/ np.timedelta64(1, 'M')
    elif row['Season'] == '2014-15':
        row['Age'] = (pd.to_datetime('06/25/2015') - row['birthday'])/ np.timedelta64(1, 'M')
    elif row['Season'] == '2015-16':
        row['Age'] = (pd.to_datetime('06/23/2016') - row['birthday'])/ np.timedelta64(1, 'M')
    elif row['Season'] == '2016-17':
        row['Age'] = (pd.to_datetime('06/22/2017') - row['birthday'])/ np.timedelta64(1, 'M')
    return row

def create_train_holdout(meta_data, college_per100, college_advance, college_shooting, nba_advance):
    """
    Create training data (2011-2015 draft classes) and hold-out data
    (2016 and 2017 draft classes). Combine all dataframes and remove
    duplicate features.

    Args:
        meta_data: pandas DataFrame with basic player information, such as
        the player's name, age, school, and draft class.
        college_per100: pandas DataFrame with college Per 100 Possessions data.
        college_advance: pandas DataFrame with college Advance data.
        college_shooting: pandas DataFrame with college shooting data.
        nba_advance: pandas DataFrame with NBA Advance data.

    Returns:
        train: training data (2011-2015 draft classes)
        holdout: holdout data (2016-2017 draft classes)
    """
    combined = pd.concat([meta_data, college_per100, college_advance,
                            college_shooting, nba_advance], axis=1)
    combined = combined.T.drop_duplicates().T
    holdout = combined.iloc[:94, :]
    train = combined.iloc[94:282, :]
    return train, holdout

def clean_holdout(holdout):
    """
    Clean the holdout data by imputing zero for all null values.

    Args:
        holdout: holdout data (2016-2017 draft classes)

    Returns:
        cleaned holdout data (2016-2017 draft classes)
    """
    holdout['3P%_per100'] = holdout['3P%_per100'].fillna(0)
    holdout.replace('---', 0, inplace=True)
    return holdout

def clean_train(train):
    """
    Clean the training data by imputing zero for all null values in the feature
    matrix. Impute the minimum NBA_VORP (-1.4) for all NBA players who played
    fewer than 50 minutes in their second NBA season. Impute -2.0 (the value
    for a replacement-level player) for missing values in NBA VORP, which
    corresponds with player's who never made it to their second NBA season.

    Args:
        train: training data (2011-2015 draft classes)

    Returns:
        cleaned train: training data (2011-2015 draft classes)
    """
    train['3P%_per100'] = train['3P%_per100'].fillna(0)
    train.replace('---', 0, inplace=True)
    train['NBA_VORP'][train['NBA_MP'] < 50] = train['NBA_VORP'].min()
    train['NBA_VORP'][train['NBA_MP'].isnull()] = -2.0
    return train

if __name__=='__main__':
    # Load Data
    meta_data = pd.read_csv('../data/meta_data.csv')
    college_per100 = pd.read_csv('../data/college_per100.csv')
    college_advance = pd.read_csv('../data/college_advance.csv')
    college_shooting = pd.read_csv('../data/college_shooting.csv')
    nba_advance = pd.read_csv('../data/nba_advance.csv')

    # Create Training and Test Data
    train, holdout = create_train_holdout(meta_data, college_per100, college_advance, college_shooting, nba_advance)

    # Clean Hold-Out DataFrame
    holdout = clean_holdout(holdout)

    # Clean Train DataFrame
    train = clean_train(train)
