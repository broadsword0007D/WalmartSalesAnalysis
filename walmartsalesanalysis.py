

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
!pip install pmdarima
import pickle

df_train=pd.read_csv("/content/train.csv")
df_features=pd.read_csv("/content/features.csv")
df_stores=pd.read_csv("/content/stores.csv")
df_test=pd.read_csv("/content/test.csv")

df_train.head()

df_features.head()

df_stores.head()

df=df_train.merge(df_features,how='left',indicator=True).merge(df_stores,how='left')

df.head()

df.isnull().sum()

df2=df.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)

df2.isnull().sum()

df2[df2.Weekly_Sales<0]

df3=pd.DataFrame(df2[df2.Weekly_Sales>0])
df3.head()

df3[df3.Weekly_Sales<0].sum()

df3['Type'].unique()

df3['Type'].value_counts()

stores = ['Type A','Type B']

data = df3['Type'].value_counts()

# Creating plot
fig, ax = plt.subplots()
plt.pie(data,labels=stores,autopct='%.0f%%')
ax.set_title('Which Type of stores has more sales')
# show plot
plt.show()

df3['year'] = pd.DatetimeIndex(df3['Date']).year
df3['month'] = pd.DatetimeIndex(df3['Date']).month
df3['week'] = pd.DatetimeIndex(df3['Date']).week
df3.head()

import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sns

# import file with data
data = df3

# prints data that will be plotted
# columns shown here are selected by corr() since
# they are ideal for the plot
print(data.corr())
sns.set_theme(style="whitegrid")
# plotting correlation heatmap
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
sns.set(rc = {'figure.figsize':(40,12)})

# displaying heatmap
mp.show()

sns.barplot(x='year', y="Fuel_Price", data=df3)
sns.set(rc = {'figure.figsize':(4,1)})

sns.barplot(x='Store', y="Weekly_Sales", data=df3)
sns.set(rc = {'figure.figsize':(12,8)})

sns.lineplot(x="Store",y="Unemployment",data=data)

df3['Dept'].unique()

sns.pointplot(x="Dept",y="Weekly_Sales",data=df3)
sns.set(rc = {'figure.figsize':(40,10)})

df4=df3.drop(columns=["Date"],axis=1)
df4.head()

month_wise_sales = pd.pivot_table(df4, values = "Weekly_Sales", columns = "year", index = "month")
month_wise_sales.plot()

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
le= preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df4['IsHoliday']= le.fit_transform(df4['IsHoliday'])
df4['Type']= le.fit_transform(df4['Type'])

df4.isnull().sum()

df4=df4.drop(columns=['_merge'],axis=1)

# df4=df4.drop([309678])
# df4

x=df4.drop(columns=['Weekly_Sales'],axis=1)
y=df4['Weekly_Sales']
x.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state=47)

import pmdarima
from pmdarima.arima import auto_arima

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt

DTRmodel = DecisionTreeRegressor(max_depth=3,random_state=0)
DTRmodel.fit(x_train,y_train)
y_pred = DTRmodel.predict(x_test)

print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred)))

rf1 = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=20,
                           max_features = 'sqrt',min_samples_split = 3)
rf1.fit(x_train,y_train)
y_pred1 = rf1.predict(x_test)

print("R2 score  :",r2_score(y_test, y_pred))
print("MSE score  :",mean_squared_error(y_test, y_pred1))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred1)))

from xgboost import XGBRegressor
model = XGBRegressor(objective='reg:squarederror', nthread= 4, n_estimators= 500, max_depth= 4, learning_rate= 0.5)
model.fit(x_train,y_train)

y_pred2 = model.predict(x_test)

print("R2 score  :",r2_score(y_test, y_pred2))
print("MSE score  :",mean_squared_error(y_test, y_pred2))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred2)))

y_pred2

from sklearn.linear_model import Ridge
rr_model = Ridge(alpha=0.5)
rr_model.fit(x_train,y_train)

y_pred3 = model.predict(x_test)

print("R2 score  :",r2_score(y_test, y_pred3))
print("MSE score  :",mean_squared_error(y_test, y_pred3))
print("RMSE: ",sqrt(mean_squared_error(y_test, y_pred3)))

y_test

from sklearn.model_selection import cross_val_score
rf = RandomForestRegressor(n_estimators=58, max_depth=27, min_samples_split=3,min_samples_leaf=1)
rf.fit(x_train, y_train.ravel())
y_pred =rf.predict(x_test)

from sklearn.model_selection import cross_val_score
xg_reg =XGBRegressor(objective="reg:squarederror", nthread= 4, n_estimators= 500, max_depth= 4, learning_rate= 0.5)
xg_reg.fit(x_train, y_train)
pred=xg_reg.predict(x_train)
y_pred=xg_reg.predict(x_test)

cv=cross_val_score(xg_reg,x,y,cv=10)
cv

np.mean(cv)

pickle.dump(rf,open('final_model.pkl','wb'))

import sklearn
print(sklearn.__version__)

pred1=xg_reg.predict(x_train)
pred2=xg_reg.predict(x_test)

print(pred1,pred2)