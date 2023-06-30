import time
import pandas as pd
from pyiqoptionapi import IQOption
import joblib

# ข้อมูลเข้าสู่ระบบของคุณ
username = 'ioko.peezaza@gmail.com'
password = 'mondaydec1997'

# เชื่อมต่อกับ IQ Option API
api = IQOption(username, password)
api.connect()

# ดึงข้อมูลราคาและตัวชี้วัด
def get_candles_data(symbol, timeframe, limit):
    candles_data = api.get_candles(symbol, timeframe, limit, time.time())
    candles_df = pd.DataFrame(candles_data, columns=['timestamp', 'open', 'close', 'min', 'max', 'volume'])
    return candles_df

# ทำการพยากรณ์ความน่าจะเป็นในการซื้อขาย
def predict_put_call(candles_df):
    # ตรวจสอบเงื่อนไขและทำการพยากรณ์

    # ตัวอย่างเงื่อนไข: หากราคาปิดของเทียนล่าสุดมากกว่าราคาเปิดในขณะนี้ พยากรณ์ในการซื้อ Call Option
    if candles_df['close'].iloc[-1] > candles_df['open'].iloc[-1]:
        return 'Call'

    # ตัวอย่างเงื่อนไข: หากราคาปิดของเทียนล่าสุดน้อยกว่าราคาเปิดในขณะนี้ พยากรณ์ในการซื้อ Put Option
    if candles_df['close'].iloc[-1] < candles_df['open'].iloc[-1]:
        return 'Put'

    # ถ้าไม่มีเงื่อนไขใดที่เข้ากันได้ คืนค่า None
    return None

# เปิดเชื่อมต่อกับ IQ Option API
if api.check_connect() == False:
    api.connect()

# เริ่มการทำงาน
while True:
    # ดึงข้อมูลราคาและตัวชี้วัดภายใน 1 นาที
    candles_df = get_candles_data('EURUSD', 1, 1000)

    # ทำการพยากรณ์ความน่าจะเป็นในการซื้อขาย
    prediction = predict_put_call(candles_df)
    print('--------')
    print(prediction)
    if prediction is not None:
        print(f'Prediction: {prediction}')

    # หน่วงเวลาก่อนดึงข้อมูลใหม่และทำการพยากรณ์อีกครั้ง
    time.sleep(60)
api.close_connect()
