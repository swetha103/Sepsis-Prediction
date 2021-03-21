'''
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index(): 
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)
'''

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from xgboost import XGBClassifier
import sklearn
#print(xgboost.__version__)
#print(xgboost.__version__)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#model1 = pickle.load(open('model1.pkl','rb'))
#model2 = pickle.load(open('model2.pkl','rb'))
#model3 = pickle.load(open('model3.pkl','rb'))
test_data = pickle.load(open('X_test.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    final_features = final_features.reshape(1, -1)

    prediction = model.predict(final_features)
    #prediction1 = model1.predict(final_features)
    #prediction2 = model2.predict(final_features)
    #prediction3 = model3.predict(final_features)
    #pred_list = [int(prediction), int(prediction1), int(prediction2), int(prediction3)]
    output = round(prediction[0], 2)
    #zero = pred_list.count(0)
    #one = pred_list.count(1)
    output_value = "" 
    #if zero > one:
    #    output_value = "0 : No chance for sepsis"
    #elif one > zero:
    #    output_value = "1 : The patient is affected with sepsis"
    #elif one == zero:
    #    output_value = " There is a possiblity of sepsis"
    #print(pred_list)
    return render_template('index.html', prediction_text='Predicted value :  {}'.format(output))

@app.route('/testcasepredict',methods=['POST'])
def testcasepredict():
    int_features = [int(x) for x in request.form.values()][0]
    print(int_features)
    
    
    
    single_pred = (model.predict(np.array(test_data[int_features]).reshape(1,-1)))
    single_pred_proba = (model.predict_proba(np.array(test_data[int_features]).reshape(1,-1)))
    print(single_pred)
    print(single_pred_proba)
    single_pred_proba = single_pred_proba[0]
    ans = ""
    if int(single_pred[0]) == 0:
        ans = "0 : No chance for sepsis"
    else:
        ans = "1 : There is a chance for sepsis"
    
    return render_template('index.html', prediction_testcase='Predicted value :  {}'.format(ans), proba_0 = single_pred_proba[0], proba_1 = single_pred_proba[-1])


if __name__ == "__main__":
    app.run(debug=True)