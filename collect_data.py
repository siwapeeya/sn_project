import cv2
import mediapipe as mp
import numpy as np
import os

# --- 1. ตั้งค่าการเก็บข้อมูล ---
actions = np.array(['100'])
no_sequences = 100  
sequence_length = 30 
data_path = "MP_Data" 

for action in actions: 
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(data_path, action, str(sequence)), exist_ok=True)

# --- 2. ตั้งค่า MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# ตัวแปรควบคุมการเริ่ม
start_collecting = False

print("โปรแกรมพร้อมแล้ว... กรุณากด 's' ที่คีย์บอร์ดเพื่อเริ่มเก็บข้อมูลทีละท่า")

for action in actions:
    # --- ส่วนที่เพิ่มเข้ามา: รอจนกว่าจะกด 's' เพื่อเริ่มท่าใหม่ ---
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'READY TO COLLECT: "{action}"', (100,200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press "s" to Start or "q" to Quit', (150,250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'): # ถ้ากด s ให้หลุดลูปเพื่อไปเริ่มเก็บข้อมูล
            break
        if key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # เริ่มเก็บข้อมูล 100 รอบสำหรับท่านั้นๆ
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if frame_num == 0: 
                cv2.putText(frame, 'STARTING COLLECTION', (120,200), 2, 0.7, (0,255,0), 4)
                cv2.putText(frame, f'Collecting: {action} | Video: {sequence}', (15,30), 2, 0.5, (0,0,255), 1)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(2000) # ลดเวลาเหลือ 2 วินาที เพราะเรากด s เริ่มเองแล้ว
            else: 
                cv2.putText(frame, f'Collecting: {action} | Video: {sequence}', (15,30), 2, 0.5, (0,0,255), 1)
                cv2.imshow('OpenCV Feed', frame)

            # สกัดพิกัด
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
            ps = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
            
            keypoints = np.concatenate([lh, rh, ps])
            npy_path = os.path.join(data_path, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

print("การเก็บข้อมูลเสร็จสมบูรณ์!")
cap.release()
cv2.destroyAllWindows()