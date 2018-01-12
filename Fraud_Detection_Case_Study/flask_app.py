import predict
import time
import pickle
import warnings
from pymongo import MongoClient
from threading import Thread
from flask import Flask, render_template, request
from final_model import MyModel

warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
@app.route('/index')
def index():
    pred = predict.get_prediction(model,mongo)
    description = pred['description']
    return '''
<h1>Current Time: ''' + time.asctime() + '''</h1>
<h1>Current prediction: ''' + str(pred['fraud_risk']) + '''</h1>
    </br>
    <h2> Description from current datapoint:</h2>
    <p> ''' + description + '''</p>
'''

@app.route('/dashboard')
def dashboard():
    low = mongo.find({'fraud_risk': 'Low Risk'}).count()
    medium = mongo.find({'fraud_risk': 'Medium Risk'}).count()
    high = mongo.find({'fraud_risk': 'High Risk'}).count()
    not_fraud = mongo.find({'fraud_risk': 'Not Fraud'}).count()
    return '''
<h1>Dashboard</h1>
</br>
</br>
<h3>High Risk Cases: ''' + str(high) + '''</h3>
<h3>Medium Risk Cases: ''' + str(medium) + '''</h3>
<h3>Low Risk Cases: ''' + str(low) + '''</h3>
<h3>Not Fraudulent Cases: ''' + str(not_fraud) + '''</h3>'''

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/score', methods=['POST'])
def pred():
    predict.get_prediction(model,mongo)
    return 'Prediction done \n'

if __name__ == '__main__':
    with open('model.pkl','rb') as f:
        model = pickle.load(f)
    client = MongoClient()
    db = client.fraud_db
    mongo = db.fraud_collection
    t1 = Thread(app.run(host='0.0.0.0', port=5005, debug=True))
    t2 = Thread(predict.continuous_predict(model,mongo))
    t1.start()
    t2.start()
