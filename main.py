
import tensorflow as tf
import streamlit as st  
import pandas as pd 
import numpy as np
from PIL import Image
from yolo.utils.tools import read_video, total_precess
from streamlit_modules.sidebar import sidebar
from streamlit_modules.header_styles import styles
from streamlit_modules.links import links
from streamlit_modules.streamlit_yolo_code import code
from streamlit_modules.file_read import file_read, online_link
from yolo.utils.tools import read_classes, read_anchors
from yolo.predictions import prediction
from streamlit_modules.button_style import button_style
import plotly.express as px
import yad2k.models.keras_yolo
import logging
from yolo import video_settings as vs 
from model_body.model_prediction import pred
from model_body.project import project
from model_body.intro import intro
from model_body.modeling import modeling
from model_body.conclusion import conclusion

def head_img(st):
    file = Image.open('./images/img_pred.jpg', 'r')
    file = np.array(file).astype(np.float32) / 255 
    st.image(file, use_column_width=True)

def head(st = st):
    yolo_model_path = './yolo_model/' 
    
    yolo_logo = links('loyo_logo')
    git_page  = links('git_page')
    
    st.markdown(f'<a href="{git_page}" target="_blank"><img src="{yolo_logo}" width="450" height="200"></a>', unsafe_allow_html=True)
    
    # Définir le style CSS personnalisé
    custom_css = styles()

    # Appliquer le style CSS personnalisé
    st.write('<style>{}</style>'.format(custom_css), unsafe_allow_html=True)

    # Utiliser le style CSS personnalisé pour afficher du texte en surbrillance
    st.write('<h1 class="custom-text">REAL time Object Detection with YOLO model</h1>', unsafe_allow_html=True)
   

    [contain_feedback, yolo_feedback_contrain] = sidebar(streamlit=st)
    #st.write('<h1 class="custom-text-under"></h1>', unsafe_allow_html=True)
    
    if contain_feedback:
        if contain_feedback == "prediction":
            pred(st=st)
        if contain_feedback == "Project description":
            project(st=st)
        if contain_feedback == "Introduction":
            intro(st=st)
        if contain_feedback == "Modelling":
            modeling(st=st)
        if contain_feedback == "Conclusion":
            conclusion(st=st)
        else: pass 
    else :
        if yolo_feedback_contrain is not None:   
            st.code(code(yolo_feedback_contrain), language='python', line_numbers=True)    
        else: head_img(st=st)

if __name__ == '__main__':
    head()