from flask import Flask,render_template,request
app = Flask(__name__)
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

model = pickle.load(open('final_model.pkl','rb'))

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])

def predict():
    s=request.form["store"]
    d=request.form["dept"]  
    d1=request.form["date"]
    date_obj = datetime.strptime(d1, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    s1=request.form["size"]
    h=request.form["h"]
    if (h=="yes"):
        h=1
    else:
        h=0
    f=request.form["fuel"]
    c=request.form["CPI"]
    t1=request.form["temp"]
    u=request.form["unemployment"]
    t=request.form["type"]
    if (t=="A"):
        t=0
    else:
        t=1




    f=[[float(s),float(d),float(s1),float(h),float(t1),float(year), float(month), float(day)
        ,float(c),float(u),float(t),float(f)]]

    prediction=model.predict(f)
    print(prediction)
    # predicted_profit = np.round(prediction[0])

    return render_template("index.html",y="The predicted profit is "+str(np.round(prediction[0])))
    # return render_template("index.html", y="The predicted profit is {}".format(predicted_profit))



if __name__ == '__main__' :
    app.run(debug=True)