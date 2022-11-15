import pandas as pd
import numpy as np
from flask import Flask, render_template, session, request,redirect
from sklearn.preprocessing import LabelEncoder
import pickle
import requests
import pyrebase

app = Flask(__name__)
# filename = 'resale_model.sav'
# model_rand = pickle.load(open(filename, 'rb'))

brandData = np.load(str('classesbrand.npy'), allow_pickle=True)
fuelData = np.load(str('classesfuelType.npy'), allow_pickle=True)
modelData = np.load(str('classesmodel.npy'), allow_pickle=True)
vehicleData = np.load(str('classesvehicleType.npy'), allow_pickle=True)

fconfig = {
    "apiKey": "FIREBASE_KEY",
    "authDomain": "carresale-fd630.firebaseapp.com",
    "projectId": "carresale-fd630",
    "storageBucket": "carresale-fd630.appspot.com",
    "messagingSenderId": "217742759498",
    "appId": "1:217742759498:web:8c55a5fa220ca091fa38b3",
    "measurementId": "G-VHBR6L69Q1",
    "databaseURL": ''
}

firebase = pyrebase.initialize_app(fconfig)
app.secret_key = 'SECRET_KEY'

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/auth")
def auth():
    return render_template('login.html')

@app.route("/login", methods=['POST'])
def login():
    auth = firebase.auth()
    email = request.form['email']
    password = request.form.get('password')
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        session['user'] = email
        print(user)
    except Exception as e:
        print(f'error:{e}')
    return "signed in"

@app.route("/register", methods=['POST'])
def register():
    auth = firebase.auth()
    name = request.form['name']
    email = request.form['email']
    password = request.form.get('password')
    try:
        user = auth.create_user_with_email_and_password(email, password)
        session['user'] = email
        print(user)
    except Exception as e:
        print(f'error:{e}')
    return "signed up"

@app.route('/logout')  
def logout():  
    if 'user' in session:  
        session.pop('user',None)
    return redirect('/')

def predictFromDeploymentModel(userInput):
    API_KEY = "IBM_DEPLOYMENT_KEY"
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
    payload_scoring = {"input_data": [{"fields": ['yearOfRegistration'	,'powerPS'	,'kilometer'	,'monthOfRegistration'	,'gearbox_labels',	'notRepairedDamage_labels',	'model_labels',	'brand_labels',	'fuelType_labels',	'vehicleType_labels'], "values": [userInput]}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/b0df73c1-d3dd-4e66-8f0d-90534cf3fe4a/predictions?version=2022-11-14', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    predictions = response_scoring.json()
    return predictions['predictions'][0]['values'][0][0]

if __name__ == '__main__':
    app.run(host='localhost', debug=True, threaded=False, port=3001)
