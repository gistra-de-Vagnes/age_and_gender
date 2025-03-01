import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import Label
from tensorflow.keras.models import load_model

# Kiểm tra xem mô hình có tồn tại không
model_path = "age_gender_model1.keras"
if not os.path.exists(model_path):
    print(f"Error: Model '{model_path}' not found!")
    exit()

# Load mô hình đã train
model = load_model(model_path)

def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))  # Resize ảnh về 64x64
    img = img.astype("float32") / 255.0  # Chuẩn hóa
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension
    return img

def predict_age_gender(frame):
    img = preprocess_frame(frame)
    predictions = model.predict(img, verbose=0)  # Tắt log output

    age = int(round(predictions[0][0]))  # Làm tròn tuổi
    age = max(0, min(age, 100))  # Giới hạn tuổi từ 0-100

    gender = "Female" if predictions[0][1] > 0.5 else "Male"  # Chữ in hoa đầu

    return age, gender

def show_result_popup(age, gender):
    """Hiển thị form kết quả bằng Tkinter"""
    popup = tk.Tk()
    popup.title("Kết quả nhận diện")

    label_text = f"Giới tính: {gender}\nTuổi: {age} years"
    label = Label(popup, text=label_text, font=("Arial", 14))
    label.pack(padx=20, pady=20)

    popup.mainloop()

# Mở webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not cap.isOpened():
    print("Error: Không thể mở webcam!")
    exit()

captured_result = None  # Lưu kết quả nhận diện cuối cùng

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Không thể đọc frame từ webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        if face.shape[0] < 10 or face.shape[1] < 10:  # Kiểm tra khuôn mặt quá nhỏ
            continue

        age, gender = predict_age_gender(face)
        captured_result = (age, gender)  # Lưu kết quả nhận diện mới nhất

        label = f"{gender}, {age} years"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age & Gender Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and captured_result:
        show_result_popup(*captured_result)  # Hiển thị form khi nhấn "s"

cap.release()
cv2.destroyAllWindows()
