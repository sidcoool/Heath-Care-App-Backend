from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import pymongo
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import os, shutil

from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import hashlib

mobile = keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)

for layer in model.layers[:-23]:
    layer.trainable = False

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])
model.load_weights("output.h5")

app = Flask(__name__)
CORS(app)



client = pymongo.MongoClient(
    'mongodb+srv://Sid:sid@mongo@cluster0-mdc4k.mongodb.net/test?retryWrites=true&w=majority', maxPoolSize=50, connect=False)

db = client.minor
colUser = db.user
tracks = db.tracks
meds = db.meds
dr = db.dr



@app.route("/")
def hello():
    return render_template('healthcare_predictor.html')


@app.route("/signup", methods=['POST'])
def signup():
    req = request.get_json()
    FirstName = req['FirstName']
    Email = req['Email']
    Age = req['Age']
    Password = req['Password']
    userType = req['userType']

    h = hashlib.md5(Password.encode())

    userName = colUser.find_one({
        "Email": Email }, {"_id": 0})

    if userName is None:
        insertId = colUser.insert_one({
            "FirstName": FirstName,
            "Email": Email,
            "Age": Age,
            "userType": userType,
            "Password": h.hexdigest()}).inserted_id

        return jsonify({'status': 'true'})        
    else:     
        return jsonify({'status': 'false'})  

         

@app.route("/login", methods=['POST'])
def login():
    req = request.get_json()
    Email = req['Email']
    Password = req['Password']

    userName = colUser.find_one({
        "Email": Email,
        "Password": hashlib.md5(Password.encode()).hexdigest()}, {"_id": 0})

    print(userName)

    if userName is None:
        return jsonify({'status': 'false'})
    else:
        return jsonify({'status': 'true', 'name' : userName['Email'], 'userType': userName['userType']})



dataset = pd.read_csv('diabetes.csv')
x = dataset.iloc[:, 0: 8].values
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

with open("diabetes_pickled", "rb") as f:
    modeld = pickle.load(f)

def pred(pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age):
    test = np.array(
        [[pregnancies, glucose, bp, skinThickness, Insulin, BMI, DPF, age]])
    test = pd.DataFrame(test)
    test = sc_x.transform(test)
    ans = (modeld.predict(test)[0])
    ans=1.0/(1+np.exp(-1*ans))
    ans=str(ans)
    return ans

@app.route("/predictDiabetes", methods=['GET', 'POST'])
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
    return jsonify({"dibetes": results})




with open("Heart_Reg", "rb") as f:
    modelhr = pickle.load(f)    

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
    ans = str(modelhr.predict(test)[0])
    print(ans)


    return ans


@app.route("/predictHeart", methods=['GET', 'POST'])
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
    return jsonify({"result": results})


name_files=[]

def predict(model,name):
    xtest=[]
    testimg=image.load_img(name,target_size=(224,224))
    X=image.img_to_array(testimg)/255.0
    xtest.append(X)
    xtest=np.array(xtest)
    ans=model.predict(xtest)
    return ans[0]


@app.route("/predictSC", methods=['POST'])
def predict_sc():
    avg=[0,0,0,0,0,0,0]

    for i in name_files:
        ans=predict(model,i)
        for i in range(0,len(ans)):
            avg[i]=avg[i]+ans[i]

    for i in range(0,len(avg)):
        avg[i]=avg[i]/len(name_files)
    
    name_files=[]
    return jsonify({"result": "Actinic Keratoses" + str(avg[0]) + '\n' + "Basal Cell Carcinoma" + str(avg[1]) + '\n' + "Benign Keratosis" + str(avg[2]) + '\n' +
                    "Dermatofibroma" + str(avg[3]) + '\n' +  "Melanoma" + str(avg[4]) + '\n' + "Melanocytic Nevi" + str(avg[5]) + '\n' +
                    "Vascular Lesions" + str(avg[6]) + '\n'})


@app.route("/uploadSC", methods=['POST'])
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
    name_files.append(f.filename)


@app.route("/track", methods=['GET', 'POST'])
def tracksave():
    req = request.get_json()
    user = req['user']
    time = req['time']
    date = req['date']
    type = req['type']
    value = req['value']

    insertId = tracks.insert_one({
        "user": user,
        "type": type,
        "time": time,
        "date": date,
        "value": value}).inserted_id

    print(time)
    print(date)
    return jsonify({"status": "true"})


@app.route("/track/get", methods=['GET', 'POST'])
def trackget():
    req = request.get_json()

    print(req)

    result = tracks.find({
        "type": req['type'],
        "user": req['user'] }, {"_id": 0})

    arr = []    

    for unit in result:
        arr.append(unit)

    print(arr)    

    return jsonify({"result": arr})        



@app.route("/med", methods=['POST'])
def medsave():
    req = request.get_json()
    print(req)
    user = req['user']
    medName = req['medName']
    dateStart = req['dateStart']
    dateEnd = req['dateEnd']
    num = req['num']


    insertId = meds.insert_one({
        "user": user,
        "medName": medName,
        "dateStart": dateStart,
        "dateEnd": dateEnd,
        "num": num}).inserted_id


    return jsonify({"status": "true"})


@app.route("/diseaseRecords", methods=['POST'])
def diseaseRecords():
    req = request.get_json()
    print(req)
 
    insertId = dr.insert_one({
        "disease": req['disease'],
          "Date": req['Date'],
          "probability": req['probability'],
          "pEmail": req['pEmail'] }).inserted_id

    return jsonify({"status": "true"})    



@app.route("/getdiseaseRecords", methods=['POST'])
def getdiseaseRecords():
    req = request.get_json()
    print(req)
 
    result = dr.find({
        "disease": req['disease'], 
          "pEmail": req['pEmail'] }, {"_id": 0})

    arr = []    

    for unit in result:
        arr.append(unit)

    return jsonify({"result": arr}) 



@app.route("/med/get", methods=['GET', 'POST'])
def medget():
    req = request.get_json()

    print(req)

    result = meds.find({
        "user": req['user'] }, {"_id": 0})

    arr = []    

    for unit in result:
        arr.append(unit)

    print(arr)    

    return jsonify({"result": arr})

app.run()
