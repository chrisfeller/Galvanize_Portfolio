import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """
    Load data from json file. Clean data and return feature matrix X and target
    vector y.

    Args:
        None

    Return:
        X: Feature Matrix
        y: Target vector (Fraud)
    """
    # Read in data from json file
    df = pd.read_json('../data/data.zip', compression = 'zip')
    # Columns to Keep
    cols = ['body_length', 'channels',
           'delivery_method',
           'fb_published', 'gts', 'has_analytics', 'has_logo','has_header',
           'name_length', 'num_order', 'num_payouts',
           'sale_duration',
           'sale_duration2', 'show_map', 'user_age',
           'user_type']
    cleandf = pd.DataFrame(df[cols])
    # Fill null values in has_header with 0
    cleandf['has_header'] = df['has_header'].fillna(0)
    # Create dummy variables for top-three most common currencies
    cur = pd.get_dummies(df['currency'])[['USD','GBP','CAD']]
    cleandf[['cur_USD','cur_GBP','cur_CAD']] = cur
    # Create dummy variables for listed
    listed = pd.get_dummies(df['listed'])
    cleandf[['listed_Y','listed_N']] = listed
    # Create dummy variables for top-four most common countries
    country = pd.get_dummies(df['country'])[['US','GB','CA','AU']]
    cleandf[['country_US','country_GB','country_CA','country_AU']] = country
    # Create dummy variables for payout_type
    payout = pd.get_dummies(df['payout_type'])[['ACH','CHECK']]
    cleandf[['payout_ACH','payout_CHECK']] = payout
    # Drop the remaining 2% of observations with any null values
    cleandf.dropna(axis=0,how='any',inplace=True)
    # Create fraud column from acct_type
    cleandf['acct_type'] = df['acct_type']
    cleandf['fraud'] = np.where(cleandf['acct_type']=='premium', 0, 1)
    y = cleandf['fraud']
    X = cleandf.drop(['acct_type','fraud'],1)
    return X, y

def clean_one_line(path):
    """
    Cleans one line of new data so that it contains same columns as training data

    Args:
        path: json file of new observation

    Returns:
        cleandf: cleaned observation
    """
    df = pd.DataFrame.from_dict(path, orient='index').T
    cleandf = pd.DataFrame(df[['body_length', 'channels', 'delivery_method',
                                'fb_published', 'gts', 'has_analytics',
                                'has_logo', 'name_length', 'num_order',
                                'num_payouts','sale_duration', 'sale_duration2',
                                'show_map', 'user_age', 'user_type','has_header']])
    add = pd.DataFrame(data=None, index=None, columns=[
       'cur_USD', 'cur_GBP', 'cur_CAD', 'listed_Y', 'listed_N',
       'country_US', 'country_GB', 'country_CA', 'country_AU', 'payout_ACH',
       'payout_CHECK'])
    # The code below, while redundant, is used to make sure that this function
    # returns the exact same columns as the previous clean data.
    if df['currency'][0] == 'US':
        add.set_value(0,'cur_US',1)
    elif df['currency'][0] == 'GB':
        add.set_value(0,'cur_GB',1)
    elif df['currency'][0] == 'CA':
        add.set_value(0,'cur_AU',1)
    if df['listed'][0] == 'y':
        add.set_value(0,'listed_Y',1)
    elif df['listed'][0] == 'n':
        add.set_value(0,'listed_N',1)
    if df['country'][0] == 'US':
        add.set_value(0,'country_US',1)
    elif df['country'][0] == 'GB':
        add.set_value(0,'country_GB',1)
    elif df['country'][0] == 'CA':
        add.set_value(0,'country_CA',1)
    elif df['country'][0] == 'AU':
        add.set_value(0,'country_AU',1)
    if df['payout_type'][0] == 'ACH':
        add.set_value(0,'payout_ACH',1)
    elif df['payout_type'][0] == 'CHECK':
        add.set_value(0,'payout_CHECK',1)
    cleandf = pd.concat([cleandf,add],axis=1)
    # Fill nulls with 0 for dummy values
    cleandf = cleandf.fillna(0)
    return cleandf

if __name__ == '__main__':
    X,y = load_data()
    print(X.head())
