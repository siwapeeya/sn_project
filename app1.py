from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf # สำหรับโหลดโมเดลที่เทรนเสร็จแล้ว

app = Flask(__name__)

# --- 1. โหลดโมเดลและตั้งค่า MediaPipe ---
# (หมายเหตุ: คุณต้องเทรนโมเดลให้เสร็จก่อนถึงจะมีไฟล์ .h5 นี้)
# model = tf.keras.models.load_model('hand_pose_model.h5') 
actions = ['Hello', 'Thank_you', 'Think', 'Love', 'Sick', 'Like', 'Dislike', 'Why', 'Who', 'Normal']

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # แปลงสีเพื่อ MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = hands.process(image)
            results_pose = pose.process(image)
            
            # เตรียมอาเรย์ 225 พิกัดเหมือนตอนเก็บข้อมูล
            full_landmarks = np.zeros(225)
            
            # ดึงพิกัดมือ
            if results_hands.multi_hand_landmarks:
                for i, hand_lms in enumerate(results_hands.multi_hand_landmarks):
                    if i < 2:
                        res_h = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()
                        full_landmarks[i*63:(i+1)*63] = res_h
                        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # ดึงพิกัดตัว
            if results_pose.pose_landmarks:
                res_p = np.array([[lm.x, lm.y, lm.z] for lm in results_pose.pose_landmarks.landmark]).flatten()
                full_landmarks[126:225] = res_p
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # --- ส่วนการทำนาย (Inference) ---
            # นำ full_landmarks ไปเข้าโมเดล (ทำเมื่อเทรนโมเดลเสร็จแล้ว)
            # input_data = np.expand_dims(full_landmarks, axis=0)
            # prediction = model.predict(input_data)
            # label = actions[np.argmax(prediction)]
            
            label = "Predicting..." # ใส่ไว้หลอกๆ ก่อนจนกว่าจะมีโมเดลจริง

            cv2.putText(frame, f"Sign: {label}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')