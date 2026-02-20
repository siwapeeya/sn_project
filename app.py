import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

app = Flask(__name__)

# --- 1. ตั้งค่า MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. ฟังก์ชันสร้างโครงสร้างโมเดล (ป้องกัน Error Batch Shape) ---
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

# --- 3. โหลดโมเดลและตั้งค่าเริ่มต้น ---
actions = ['ไม่ชอบ', 'ป่วย', 'สวัสดี','ชอบ','ขอบคุณ','ทำไม'] # *** เปลี่ยนชื่อท่าให้ตรงกับที่คุณเทรน ***
MODEL_PATH = os.path.join('model', 'action_model (1).h5')
model = build_model(len(actions))

try:
    model.load_weights(MODEL_PATH)
    print("✅ โหลดน้ำหนักโมเดลสำเร็จ!")
except:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ โหลดโมเดลสำเร็จ (วิธีสำรอง)!")

# ตัวแปร Global สำหรับแชร์ข้อมูล
sequence = []
correct_counter = 0
target_action = 'ไม่ชอบ'
detected_action = "รอประมวลผล..."
confidence = 0.0

@app.route('/get_status')
def get_status():
    global correct_counter, target_action, detected_action, confidence
    return jsonify({
        'score': correct_counter,
        'target': target_action,
        'detected': detected_action,
        'confidence': float(confidence)
    })

def generate_frames():
    global sequence, correct_counter, target_action, detected_action, confidence
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.RGB2BGR if hasattr(cv2, 'RGB2BGR') else cv2.COLOR_RGB2BGR)

        # สกัดพิกัด (Keypoints)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        ps = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
        keypoints = np.concatenate([lh, rh, ps])

        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            detected_action = actions[predicted_idx]
            confidence = res[predicted_idx]

            # Logic นับคะแนนและเปลี่ยนท่า (Threshold 0.4 เพื่อให้ง่ายขึ้น)
            if confidence > 0.4:
                if detected_action == target_action:
                    correct_counter += 1
                    if correct_counter >= 20: # ผ่านเมื่อได้ 20 แต้ม
                        current_idx = actions.index(target_action)
                        target_action = actions[(current_idx + 1) % len(actions)]
                        correct_counter = 0
                else:
                    correct_counter = max(0, correct_counter - 0.5) # ผิดลบนิดหน่อย

        # วาดเส้นบน OpenCV
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)