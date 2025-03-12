import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model

try:
    model = load_model("age_gender_model_1.keras")
except Exception as e:
    print(f"⚠️ Lỗi khi load mô hình: {e}")
    exit()

def preprocess_frame(frame):
    try:
        img = cv2.resize(frame, (64, 64)) / 255.0  # Chuẩn hóa giống khi train
        img = np.expand_dims(img, axis=0)  # Thêm batch dimension
        return img
    except Exception as e:
        print(f"⚠️ Lỗi khi xử lý frame: {e}")
        return None

def predict_frame(frame, debug=False):
    try:
        img = preprocess_frame(frame)
        if img is None:
            return None, None
        age_pred, gender_pred = model.predict(img, verbose=0)
        if debug:
            print(f'Raw gender prediction: {gender_pred[0][0]}')
        predicted_age = int(age_pred[0][0] * 100)
        predicted_gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
        return predicted_age, predicted_gender
    except Exception as e:
        print(f"⚠️ Lỗi khi dự đoán: {e}")
        return None, None

# Mở webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if not cap.isOpened():
    print("⚠️ Không thể mở webcam!")
    exit()

# Khởi tạo danh sách để lưu thông tin các khuôn mặt
face_predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Không thể đọc frame từ webcam!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Xóa danh sách cũ và cập nhật với các khuôn mặt mới
    face_predictions.clear()

    for i, (x, y, w, h) in enumerate(faces):
        face = frame[y:y + h, x:x + w]
        face = cv2.flip(face, 1)  # Lật ảnh để khớp với dữ liệu huấn luyện
        if face.shape[0] < 10 or face.shape[1] < 10:
            continue

        age, gender = predict_frame(face, debug=True)
        if age is not None and gender is not None:
            # Lưu thông tin khuôn mặt
            face_predictions.append((i + 1, gender, age))
            # Hiển thị trên frame
            label = f"{gender}, {age} years"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age & Gender Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and face_predictions:
        result_text = "Kết quả dự đoán:\n"
        for face_id, gender, age in face_predictions:
            result_text += f"Khuôn mặt {face_id}: Giới tính: {gender}, Tuổi: {age} years\n"
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Kết quả dự đoán", result_text)
        root.destroy()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()