import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from iqoptionapi.stable_api import IQ_Option
import matplotlib.pyplot as plt
import time

# กำหนดค่าสำหรับเข้าสู่ระบบ IQ Option
username = 'ioko.peezaza@gmail.com'
password = 'mondaydec1997'

# กำหนดข้อมูลการเทรด
instrument = 'EURUSD'
window_size = 60
train_size = 0.8
prediction_length = 5

# กำหนดพารามิเตอร์ LSTM
lstm_units = 50
dropout_rate = 0.2
epochs = 50
batch_size = 32

API = IQ_Option(username, password)
API.connect()

candles = API.get_candles(instrument, 60, (window_size + prediction_length) * 2, time.time())

df = pd.DataFrame(candles, columns=['id', 'from', 'to', 'open', 'close', 'min', 'max', 'volume'])
df['from'] = pd.to_datetime(df['from'], unit='s')
df = df.set_index('from')
df = df[['open', 'close', 'min', 'max']]

# สร้างฟีเจอร์ใหม่จากข้อมูลปัจจุบันและข้อมูลก่อนหน้า
for i in range(1, window_size + 1):
    for col in ['open', 'close', 'min', 'max']:
        df['{}_{}'.format(col, i)] = df[col].shift(i)

# ลบแถวที่มีค่า NA
df = df.dropna()

# แยกข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
train_size = int(len(df) * train_size)
train_data = df[:train_size].values
test_data = df[train_size:].values

# ปรับค่าข้อมูลให้อยู่ในช่วง 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# สร้างชุดข้อมูลฝึกสอนและชุดข้อมูลทดสอบ
X_train = []
y_train = []
for i in range(window_size, len(train_data_scaled) - prediction_length):
    X_train.append(train_data_scaled[i - window_size:i])
    y_train.append(train_data_scaled[i + prediction_length - 1][1])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(window_size, len(test_data_scaled) - prediction_length):
    X_test.append(test_data_scaled[i - window_size:i])
    y_test.append(test_data_scaled[i + prediction_length - 1][1])
X_test, y_test = np.array(X_test), np.array(y_test)

model = Sequential()
model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(window_size, df.shape[1] - 2)))
model.add(LSTM(units=lstm_units, dropout=dropout_rate))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

df_test = pd.DataFrame(test_data[window_size + prediction_length:], columns=df.columns)
df_test['prediction'] = predictions

df_test['actual_movement'] = df_test['close'].diff()
df_test['predicted_movement'] = df_test['prediction'].diff()

df_test['predicted_direction'] = np.where(df_test['predicted_movement'] > 0, 'Call', 'Put')
df_test['actual_direction'] = np.where(df_test['actual_movement'] > 0, 'Call', 'Put')

accuracy = (df_test['predicted_direction'] == df_test['actual_direction']).mean()
print('Accuracy:', accuracy)




