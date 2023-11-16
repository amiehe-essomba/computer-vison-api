import streamlit as st 
from ultralytics import YOLO
import logging
import tensorflow as tf 

@st.cache_resource()
def load_all_models():
    yolo_model_path = './yolo_model/'
    NAMES = ["yolov8n-seg.pt", 'yolov8n.pt', 'yolov8n-pose.pt', 'license_plate_detector.pt']
    PATHS = [f'./yolov8/{name}' for name in NAMES]

    all_models = {}

    for i, path in enumerate(PATHS):
        model = YOLO(path)
        all_models[NAMES[i]] = model
    
    tf.get_logger().setLevel(logging.ERROR)
    model = tf.keras.models.load_model(yolo_model_path, compile=False)

    all_models['my_model'] = model

    return all_models