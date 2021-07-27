
print(r"""

      ***************************************************** 
    *                           __                          *
   *                           /  \                          *
  *                           /_  _\                          * 
 *                             / /                             *
*                             / /                               *
*                     \-\  /\/ /                                *
 *                     \ \/   /                                *      
  *                     \  /\/                                *
   *                     \/                                  * 
     *******************************************************   


  """)


print("\n****************************************************************")
print("\n* Copyright of Simple Stock, 2021                              *")
print("\n* https://www.gitCaptone.com                                   *")
print("\n****************************************************************")

import math
import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM



def lstm():
  df = pd.read_csv("./data/MLdataset.csv")
  data = df.filter(['Open'])
  dataset = data.values

    
  training_data_len = math.ceil( len(df) *.8) 

  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0, 1)) 
  scaled_data = scaler.fit_transform(dataset)

  train_data = scaled_data[0:training_data_len,:]
  x_train=[]
  y_train = []
  for i in range(60,len(train_data)):
      x_train.append(train_data[i-60:i,0])
      y_train.append(train_data[i,0])


  x_train, y_train = np.array(x_train), np.array(y_train)


  x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


  print("-"*30);print("LSTM MODEL")
  print("-"*30)


  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
  model.add(LSTM(units=50, return_sequences=False))
  model.add(Dense(units=25))
  model.add(Dense(units=1))


  model.compile(optimizer='adam', loss='mse')

  model.fit(x_train, y_train, batch_size=32, epochs=50)

  print("-"*30);print("Model Compiled Successfully");print("-"*30)

  print("-"*30);print("Loading Testing data");print("-"*30)

  test_data = scaled_data[training_data_len - 60: , : ]
  x_test = []
  y_test =  dataset[training_data_len : , : ]

  for i in range(60,len(test_data)):
      x_test.append(test_data[i-60:i,0])

  x_test = np.array(x_test)

  x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
  print("-"*30);print("Predicting Opening Price");print("-"*30)

  pred = model.predict(x_test) 
  pred = scaler.inverse_transform(pred)

  print(pred)
  print("-"*30);print("PLotting Graph");print("-"*30)

  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predictions'] = pred
  #Visualize the data
  plt.figure(figsize=(16,8))
  plt.title('Share Price Prediction')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Open Price', fontsize=18)
  plt.plot(train['Open'])
  plt.plot(valid[['Open', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()

  return valid



def main():
  lstm()


if __name__ == "__main__":
  main()




  


