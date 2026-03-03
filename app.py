import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 1. โครงสร้าง Model (ตามที่แกเทรนมา) ---
def build_model(action_size):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 225)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size, activation='softmax'))
    return model

# --- 2. โหลดโมเดล ---
actions = ['ไม่ชอบ', 'ป่วย', 'สวัสดี','ชอบ','ขอบคุณ','ทำไม']
MODEL_PATH = os.path.join('model', 'action_model (1).h5')
model = build_model(len(actions))

if os.path.exists(MODEL_PATH):
    try:
        model.load_weights(MODEL_PATH)
        print("✅ Load Model Weights Success!")
    except:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Load Full Model Success!")

# ตัวแปรเก็บลำดับเฟรม
sequence = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('predict_keypoints')
def handle_prediction(keypoints):
    global sequence
    
    # รับพิกัด (Array 225 ค่า) มาสะสม
    sequence.append(keypoints)
    sequence = sequence[-30:] # เก็บแค่ 30 เฟรมล่าสุด

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        predicted_idx = np.argmax(res)
        confidence = float(res[predicted_idx])
        
        # ส่งผลลัพธ์กลับไปหน้าบ้าน
        emit('response', {
            'detected': actions[predicted_idx],
            'confidence': confidence
        })

if __name__ == "__main__":
    # สำหรับรันในเครื่องตัวเอง
    socketio.run(app, debug=True)