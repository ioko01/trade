from iqoptionapi.stable_api import IQ_Option
from ta.trend import MACD
import pandas as pd

username = 'ioko.peezaza@gmail.com'
password = 'mondaydec1997'

# เข้าสู่ระบบ IQ Option
api = IQ_Option(username, password)
api.connect()

# ตั้งค่าตลาด
api.change_balance("PRACTICE")  # สำหรับบัญชีทดลอง

# ดึงข้อมูลราคา
candles = api.get_candles("EURUSD", 60, 100, api.get_server_timestamp())

# สร้าง DataFrame จากข้อมูลราคา
data = {
    "timestamp": [candle["from"] for candle in candles],
    "open": [candle["open"] for candle in candles],
    "high": [candle["max"] for candle in candles],
    "low": [candle["min"] for candle in candles],
    "close": [candle["close"] for candle in candles],
    "volume": [candle["volume"] for candle in candles]
}
df = pd.DataFrame(data)

# คำนวณตัวชี้วัดทางเทคนิค เช่น MACD
df["macd"] = MACD(df["close"]).macd()

# ทำนายการซื้อขาย
last_macd = df["macd"].iloc[-1]
if last_macd > 0:
    prediction = "Call"
else:
    prediction = "Put"

# พิมพ์ผลลัพธ์การทำนาย
print("Prediction:", prediction)

# ออกจากระบบ IQ Option

