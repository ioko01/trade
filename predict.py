from iqoptionapi.stable_api import IQ_Option
import time
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ฟังก์ชันสำหรับรับข้อมูลราคาตลาดจาก IQ Option API
def get_market_data(symbol, timeframe, count):
    API=IQ_Option("ioko.peezaza@gmail.com","mondaydec1997")
    API.connect()
    print("get candles")
    return API.get_candles(symbol,timeframe,count,time.time())


# ฟังก์ชันสำหรับวิเคราะห์และทำนาย Put Call
def predict_put_call(market_data):
    # ดึงข้อมูลราคาปิด
    close_prices = [data["close"] for data in market_data]

    # สร้าง DataFrame
    df = pd.DataFrame({"Close": close_prices})

    # สร้างตัวแปรเป้าหมาย Put Call โดยใช้ข้อมูลจากตัวถังต่อไป
    df["Target"] = df["Close"].shift(-1) > df["Close"]

    # ลบแถวสุดท้ายที่มีค่า Target เป็น NaN
    df = df.dropna()

    # แบ่งข้อมูลเป็นชุดฝึกหัดและชุดทดสอบ
    X = df[["Close"]]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้างแบบจำลอง RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # ทำนาย Put Call สำหรับข้อมูลทดสอบ
    predictions = model.predict(X_test)
    return predictions[-1]

# ตัวอย่างการใช้งาน
symbol = "EURUSD"
timeframe = 300
count = 1000

# รับข้อมูลตลาด
market_data = get_market_data(symbol, timeframe, count)

# ทำนาย Put Call
prediction = predict_put_call(market_data)
print("Predicted Put Call:", prediction)