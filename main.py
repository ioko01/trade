from iqoptionapi.stable_api import IQ_Option
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# เข้าสู่ระบบ IQ Option
Iq = IQ_Option("ioko.peezaza@gmail.com","mondaydec1997")
Iq.connect()

# ดึงข้อมูลราคา
asset = 'EURUSD'  # สินทรัพย์ที่ต้องการดึงข้อมูล
timeframe = 60  # ระยะเวลาของแท่งเทียน (หน่วย: วินาที)
num_candles = 1000  # จำนวนแท่งเทียนที่ต้องการดึง
endtime = time.time()
candles = Iq.get_candles(asset, timeframe, num_candles, endtime)

# สร้าง DataFrame จากข้อมูลราคา
df = pd.DataFrame(candles, columns=['timestamp', 'open', 'close', 'min', 'max', 'volume'])

# สร้างคอลัมน์ Target โดยตรวจสอบว่าราคาขึ้นหรือลง
df['target'] = np.where(df['close'].shift(-1) > df['close'], 'Call', 'Put')

# ลบแถวที่มีค่า NaN
# df.dropna(inplace=True)

# เลือกฟีเจอร์ที่ใช้ในการทำนาย
features = ['open', 'close', 'min', 'max', 'volume']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X = df[features].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ทำนายผลในชุดทดสอบ
y_pred = model.predict(X_test)

# คำนวณความแม่นยำของโมเดล
accuracy = accuracy_score(y_test, y_pred)
print('ความแม่นยำของโมเดล:', accuracy)

# ทำนายควรจะใช้ Put หรือ Call ในชุดข้อมูลล่าสุด
new_data = [df.iloc[-1][features]]
prediction = model.predict(new_data)[0]
print('ควรจะ', prediction, 'ใน IQ Option')
