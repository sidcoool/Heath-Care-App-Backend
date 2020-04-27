from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
# import pymongo
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import os, shutil

'''from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model=Sequential()

model.add(Conv2D(32,(3,3), activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3), activation='relu',input_shape=(100,100,3)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.load_weights("best_model.h5")'''

app = Flask(__name__)
# CORS(app)



'''client = pymongo.MongoClient(
    'mongodb+srv://Sid:Sid_smart1@cluster0-mdc4k.mongodb.net/test?retryWrites=true&w=majority', maxPoolSize=50, connect=False)

db = client.minor
colUser = db.user




@app.route("/")
def hello():
    return render_template('healthcare_predictor.html')


@app.route("/signup", methods=['POST'])
def signup():
    req = request.get_json()
    FirstName = req['FirstName']
    LastName = req['LastName']
    Age = req['Age']
    Password = req['Password']

    insertId = colUser.insert_one({
        "FirstName": FirstName,
        "LastName": LastName,
        "Age": Age,
        "Password": Password}).inserted_id

    return jsonify({'status': 'true'}) 


@app.route("/login", methods=['POST'])
def login():
    req = request.get_json()
    FirstName = req['FirstName']
    Password = req['Password']

    userName = colUser.find_one({
        "FirstName": FirstName,
        "Password": Password}, {"_id": 0})

    print(userName)

    if userName is None:
        return jsonify({'status': 'false'})
    else:
        return jsonify({'status': 'true', 'name' : userName['FirstName']})




dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:, 0: 8].values
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

with open("Diabetes_Pickle", "rb") as f:
    modeld = pickle.load(f)


def pred(pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age):
    test = np.array(
        [[pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age]])
    test = pd.DataFrame(test)
    test = sc_x.transform(test)
    ans = str(modeld.predict(test)[0])
    return ans


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    req = request.get_json()
    pregnancies = req['pregnancies']
    glucose = req['glucose']
    bp = req['bp']
    skinThickness = req['skinThickness']
    Insulin = req['Insulin']
    BMI = req['BMI']
    DPF = req['DPF']
    age = req['age']

    results = pred(pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age)
    return jsonify({"dibetes": results})'''


# predicting heart disease
with open("heart_disease.pickle", "rb") as f:
    modelh = pickle.load(f)

'''age = 37
sex = 1
cp = 2
trestbps = 130
chol = 250
fbs = 0
restecg = 1
thalach = 187
exang = 0
oldpeak = 3.5
slope = 0
ca = 0
thal = 2 
test = np.array(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
test = pd.DataFrame(test)
test = sc_x_h.transform(test)
ans = str(modelh.predict(test)[0])'''



def heart_pred(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    dataset_heart = pd.read_csv('heart.csv')
    x_h = dataset_heart.iloc[:, :-1].values
    sc_x_h = StandardScaler()
    x_h = sc_x_h.fit_transform(x_h)
    
    test = np.array(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    test = pd.DataFrame(test)
    print(test)
    test = sc_x_h.transform(test)
    print(test)
    ans = str(modelh.predict(test)[0])
    print(ans)
    return ans


@app.route("/heart_pred", methods=['GET', 'POST'])
def heart_predict():
    req = request.get_json()
    age = req['age']
    sex = req['sex']
    cp = req['cp']
    trestbps = req['trestbps']
    chol = req['chol']
    fbs = req['fbs']
    restecg = req['restecg']
    thalach = req['thalach']
    exang = req['exang']
    oldpeak = req['oldpeak']
    slope = req['slope']
    ca = req['ca']
    thal = req['thal']
    

    results = heart_pred(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    print(results)
    return jsonify({"heart disease": results})


# prediction of breast cancer

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

modelb = Sequential()
modelb.add(Dense(units = 15, activation = 'relu', input_dim = 30))
modelb.add(Dense(units = 15, activation = 'relu'))
modelb.add(Dense(units = 1, activation = 'sigmoid'))
modelb.load_weights("beast_cancer.h5")

dataset_breast = pd.read_csv('data.csv')
x_h = dataset_breast.iloc[:, 2: 32].values
sc_x_h = StandardScaler()
x_h = sc_x_h.fit_transform(x_h)

def breast_pred(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23,
               b24, b25, b26, b27, b28, b29, b30):
    test = np.array(
        [[b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23,
               b24, b25, b26, b27, b28, b29, b30]])
    test = pd.DataFrame(test)
    test = sc_x_h.transform(test)
    ans = str(modelb.predict(test)[0])
    return ans

@app.route("/breast_pred", methods=['GET', 'POST'])
def breast_predict():
    req = request.get_json()   
    b1 = req['b1']
    b2 = req['b2']
    b3 = req['b3']
    b4 = req['b4']
    b5 = req['b5']
    b6 = req['b6']
    b7 = req['b7']
    b8 = req['b8']
    b9 = req['b9']
    b10 = req['b10']
    b11 = req['b11']
    b12 = req['b12']
    b13 = req['b13']
    b14 = req['b14']
    b15 = req['b15']
    b16 = req['b16']
    b17 = req['b17']
    b18 = req['b18']
    b19 = req['b19']
    b20 = req['b20']
    b21 = req['b21']
    b22 = req['b22']
    b23 = req['b23']
    b24 = req['b24']
    b25 = req['b25']
    b26 = req['b26']
    b27 = req['b27']
    b28 = req['b28']
    b29 = req['b29']
    b30 = req['b30']
    
    results = breast_pred(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23,
               b24, b25, b26, b27, b28, b29, b30)
    return jsonify({"breast cancer": results})
    
    predictedc=np.argmax(results)
    if(predictedc == 0):
        return jsonify({"result": "Benign"})
    else:
        return jsonify({"result": "Malignant"})


'''@app.route("/file", methods=['POST'])
def fileSaver():
    # print(request.files)
    # data = dict(request.form)
    # print(data)
    f = request.files['scfile']
    f.save(secure_filename(f.filename))
    print(f)
    src= f.filename
    dest="upload/"+f.filename
    shutil.move(src,dest)

    xtest=[]
    testimg=image.load_img("upload/" + f.filename, target_size=(100,100))
    X=image.img_to_array(testimg)/255.0
    xtest.append(X)
    xtest=np.array(xtest)
    #print(xtest)
    ans=model.predict(xtest)
    # print(ans)  

    predicted_class=np.argmax(ans)
    if(predicted_class == 0):
        return jsonify({"result": "Benign"})
    else:
        return jsonify({"result": "Malignant"})'''

app.run()


'''{
	'age'      : '37',
    'sex'      : '1',
    'cp'       : '2',
    'trestbps' : '130',
    'chol'     : '250',
    'fbs'      : '0',
    'restecg'  : '1',
    'thalach'  : '187',
    'exang'    : '0',
    'oldpeak'  : '3.5',
    'slope'    : '0',
    'ca'       : '0',
    'thal'     : '2'
}'''
