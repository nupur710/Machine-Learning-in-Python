import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

df=quandl.get('WIKI/GOOGL')

#updating dataframe to only List of all columns we want to have
df= df[['Open', 'High', 'Low', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
#feature engineering: volatility
df['HL_PCT']= (df['High']-df['Low']) / df['Low'] * 100.0
#feature engineering: daily % change
df['PCT_change']= (df['Adj. Close']-df['Open']) / df['Open'] * 100.0
#define new dataframe with only columns we care about
df= df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#label
forecast_col= 'Adj. Close'
#treat missing data as outlier in dataset
df.fillna(-99999, inplace=True)
#math.ceil rounds to nearest whole no.; making it to integer; predicting for 10 days
forecast_out= int(math.ceil(0.01*len(df)))
#print('Forecast out val: ')
#print(forecast_out)
#making sure the label column for each row will be adj. close price 10 days into future
df['label']= df[forecast_col].shift(-forecast_out)

#Train and Test
X= np.array(df.drop(['label'], 1)) #all columns are features except label column
X= preprocessing.scale(X) #normalizing
X_lately= X[-forecast_out:] #test data starts from after forecast_out
X=X[:-forecast_out] #training data only till forecast_out

df.dropna(inplace=True)
y= np.array(df['label'])

#X= X[:-forecast_out+1] #to make sure we only have x's where we have value for y

X_train, X_test, y_train, y_test= model_selection.train_test_split(X, y, test_size=0.2)
clf= LinearRegression(n_jobs=-1) #To use svm: clf= svm.SVR(); you can add kernel clf= svm.SVR(kernel='poly'); n_jobs->threading
clf.fit(X_train, y_train) #train
accuracy= clf.score(X_test, y_test) #test
print(accuracy) #prints accuracy on predicting what the price would be shifted 1% of the day
#We have data left to forecast because we only trained data for approx 30 values -- see y

#Preditction
forecast_set= clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['forecast']= np.nan #specifies the entire column is full of nan data. We will put info there
last_date= df.iloc[-1].name
last_unix= last_date.timestamp()
one_day= 86400
next_unix= last_unix + one_day

df['Adj. Close'].plot()
df['forecast'].plot()
plt.show()
