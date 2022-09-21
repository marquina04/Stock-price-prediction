import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start= '2010-01-01'
end= '2022-06-30'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter the stock ticker', 'AAPL')
df=data.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data from 2010 - NOW')
st.write(df.describe())

st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100MA')
ma100= df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 200MA')
ma200= df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scalar= MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training)


model = load_model('model.h5')

past_100_days= data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data=scalar.fit_transform(final_df)


X_test = []
Y_test=[]
for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    Y_test.append(input_data[i,0])

X_test, Y_test = np.array(X_test), np.array(Y_test)
Y_pred=model.predict(X_test)
scalar= scalar.scale_
scale_factor = 1/scalar[0]
Y_pred = Y_pred*scale_factor
Y_test = Y_test*scale_factor


st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(Y_test, 'b', label = 'Original Price')
plt.plot(Y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)