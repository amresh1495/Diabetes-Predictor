import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, url_for, request
import pandas as pd
import os
import numpy
import pdb

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    diab = pd.read_csv("data/diabetes.csv")
    train, test = train_test_split(
        diab, test_size=0.25, random_state=0, stratify=diab['Outcome'])
    train_X = train[train.columns[:8]]
    test_X = test[test.columns[:8]]
    train_Y = train['Outcome']
    test_Y = test['Outcome']
    model = svm.SVC(kernel='linear')
    model.fit(train_X, train_Y)
    prediction = model.predict(test_X)
    accuracy = metrics.accuracy_score(prediction, test_Y)*100
    print('Accuracy for SVM linear kernel is',
          accuracy, 'percent')

    if request.method == 'POST':
        preg = request.form['preg']
        glucose = request.form['glucose']
        bp = request.form['bp']
        skin = request.form['skin']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']
        features = numpy.array([
            [preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        my_prediction = model.predict(features)
    return render_template('result.html', prediction=my_prediction,
                           accuracy=accuracy)


if __name__ == '__main__':
    app.run()
