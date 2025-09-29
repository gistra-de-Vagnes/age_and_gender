# Age & Gender Detection Using Deep Learning

This project uses a deep learning model to predict age and gender from webcam images in real-time. The application is built with Python, OpenCV, TensorFlow/Keras, and Tkinter for GUI popups.

## Features
- Detects faces from webcam input using Haar cascades
- Predicts age and gender for each detected face using a trained Keras model
- Displays results on the video frame and in a popup window

## Project Structure
- `main.py`: Main application script for webcam capture, face detection, and prediction
- `age_gender_model_1.keras`: Pre-trained Keras model for age and gender prediction
- `Mô hình.ipynb`: Jupyter notebook for model training and experimentation

## Requirements
- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- TensorFlow (with Keras)
- Tkinter (usually included with Python)

Install dependencies with:
```bash
pip install opencv-python numpy tensorflow
```

## Usage
1. Ensure `age_gender_model_1.keras` is present in the project directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The webcam will open and start detecting faces. Age and gender predictions will be shown on the video.
4. Press `a` to show a popup with prediction results for all detected faces.
5. Press `Esc` to exit.

## Notes
- The model expects face images of size 64x64 pixels, normalized to [0, 1].
- The application uses Haar cascades for face detection, which may not work perfectly in all lighting conditions.

## License
This project is for educational purposes.

