import pickle
import json
import urllib.request
from pymongo import MongoClient
from clean_data import clean_one_line
from final_model import MyModel

def predict_one(example):
    """
    Makes prediciton on one new observation

    Args:
        example: json file of new observation

    Returns:
        fraud: probability that new observation is fraud and writes result to mongodb
    """
    # Find probability that new observation is fraud
    X = clean_one_line(example)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    p = model.predict_proba(X)[0]
    if p[1] > p[0]:
        fraud = True
    else:
        fraud = False
    example['fraud'] = fraud
    # Write to mongodb
    client = MongoClient()
    db = client.fraud_db
    collection = db.fraud_collection
    collection.insert_one(example)
    return fraud

def get_prediction():
    """
    Retrieves one observation from heroku server streaming data and makes
    prediciton as to whether teh observation is fraudulent.

    Args:
        None

    Returns:
        fraud: probability that new observation is fraud
    """
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    page = urllib.request.urlopen(url)
    d = json.load(page)
    fraud = predict_one(d)
    return fraud

if __name__ == '__main__':
    get_prediction()
