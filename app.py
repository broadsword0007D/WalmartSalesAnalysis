import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
import pickle
import datetime as dt
import calendar
import os
import sklearn

print(sklearn.__version__)
app= Flask (__name__)

loaded_model = pickle.load(open(r'final_model.pkl', 'rb')) 
# fet = pd.read_csv('merged_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    store =request.form.get('store')
    dept =request.form.get('dept') 
    date =request.form.get('date') 
    isHoliday =request.form['isHolidayRadio'] 
    size=request.form.get('size')
    temp=request.form.get('temp')
    d=dt.datetime.strptime(date, '%Y-%m-%d')
    year = (d.year)
    month =d.month
    month_name=calendar.month_name [month] 
    print("year = ", type(year))
    print("year val = ", year, type(year), month)
    x_test = pd.DataFrame({'Store': [store], 'Dept': [dept], 'Size':[size],
                            'Temperature': [temp], 'CPI': [212], 'MarkDown4':[2050],
                            'IsHoliday': [isHoliday], 'Type B':[0], 'Type C':[1],
                            'month':[month], 'year': [year]}) 
    print("x_test= ", x_test.head())
    print("type of x test = ", type (x_test))
    print("predict = ", store, dept, date, isHoliday)
    y_pred =loaded_model.predict(x_test)
    output=round(y_pred[0],2) 
    print("predicted = ", output)
    return render_template('index.html', output=output, store=store, dept=dept, 
                           month_name=month_name, year=year)


port = os.getenv("VCAP_APP_PORT", '8080')

if __name__ == '__main__':
    app.secret_key=os.urandom(12)
    app.run(debug=False,host='0.0.0.0', port=port)