import cv2
import numpy as np
import os
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.vgg19 import preprocess_input

MODEL_DIR = 'models_trained_vgg19_full' 

EMOTION_MODEL_PATH = os.path.join('models_trained_vgg19', 'emotion_model_vgg19.h5') 
GENDER_MODEL_PATH = os.path.join(MODEL_DIR, 'gender_model_vgg19.h5')
AGE_MODEL_PATH = os.path.join(MODEL_DIR, 'age_model_vgg19.h5')
RACE_MODEL_PATH = os.path.join(MODEL_DIR, 'race_model_vgg19.h5')

# Load Face Detector
haarcascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')

if not os.path.exists(haarcascade_path):
    print(f"[ERROR] Could not find haarcascade file at: {haarcascade_path}")
    print("[INFO] Please check your OpenCV installation. The cascade file is missing.")
    exit()

face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Load Trained Models
model_paths = [EMOTION_MODEL_PATH, GENDER_MODEL_PATH, AGE_MODEL_PATH, RACE_MODEL_PATH]
for path in model_paths:
    if not os.path.exists(path):
        print(f"[ERROR] Model file not found: {path}. Please run the training script first.")
        exit()

print("[INFO] Loading all models...")
try:
    emotion_model = load_model(EMOTION_MODEL_PATH)
    gender_model = load_model(GENDER_MODEL_PATH)
    age_model = load_model(AGE_MODEL_PATH)
    race_model = load_model(RACE_MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load models: {e}")
    exit()
print("[INFO] Models loaded successfully.")

# Labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
GENDER_LABELS = ['Male', 'Female']
RACE_LABELS = ['White', 'Black', 'Asian', 'Indian', 'Other']

# Main Application Loop
print("[INFO] Starting video stream...")
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        
        utk_roi = cv2.resize(roi_color, (64, 64), interpolation=cv2.INTER_AREA)
        utk_roi = np.expand_dims(img_to_array(utk_roi), axis=0)
        utk_roi = preprocess_input(utk_roi)

        gender_pred = gender_model.predict(utk_roi, verbose=0)[0]
        age_pred = age_model.predict(utk_roi, verbose=0)[0]
        race_preds = race_model.predict(utk_roi, verbose=0)[0]
        
        emotion_roi_color = cv2.resize(roi_color, (48, 48), interpolation=cv2.INTER_AREA)
        emotion_roi_color = np.expand_dims(img_to_array(emotion_roi_color), axis=0)
        emotion_roi_color = preprocess_input(emotion_roi_color)
        emotion_preds = emotion_model.predict(emotion_roi_color, verbose=0)[0]

        gender_label = GENDER_LABELS[int(round(gender_pred[0]))]
        age_label = f"{int(age_pred[0])} yrs"
        race_label = RACE_LABELS[race_preds.argmax()]
        emotion_label = EMOTION_LABELS[emotion_preds.argmax()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_top = f"{gender_label}, {age_label}, {race_label}"
        label_bottom = f"Emotion: {emotion_label}"
        
        (w_top, h_top), _ = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        (w_bottom, h_bottom), _ = cv2.getTextSize(label_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        cv2.rectangle(frame, (x, y - h_top - 10), (x + w_top, y - 10), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (x, y + h + 5), (x + w_bottom, y + h + h_bottom + 10), (0, 0, 0), cv2.FILLED)

        cv2.putText(frame, label_top, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, label_bottom, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow('Full Facial Feature Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Video stream stopped.")