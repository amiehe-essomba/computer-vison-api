
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
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform


def head_img(st, path='./images/img_pred.jpg', factor : int = 50, types : str='image'):
    if types == 'image':
        file = Image.open(path, 'r')
        st.image(file, use_column_width=True)
    else:
        video_file = open(path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, format="video/mp4")
        video_file.close()

def head(st = st):
    yolo_logo = './images/ocr.png' #links('loyo_logo')
    git_page  = links('git_page')
    
    st.image(plt.imread(yolo_logo))
    #st.markdown(f'<a href="{git_page}" target="_blank"><img src="{yolo_logo}" width="450" height="200"></a>', unsafe_allow_html=True)
    
    # Définir le style CSS personnalisé
    custom_css = styles()

    # Appliquer le style CSS personnalisé
    st.write('<style>{}</style>'.format(custom_css), unsafe_allow_html=True)

    # Utiliser le style CSS personnalisé pour afficher du texte en surbrillance
    st.write('<h1 class="custom-text">Optical Character Recognition (OCR) & REAL-time Object Detection with YOLO</h1>', unsafe_allow_html=True)
   
    [contain_feedback, yolo_feedback_contrain] = sidebar(streamlit=st)
    #st.write('<h1 class="custom-text-under"></h1>', unsafe_allow_html=True)
    
    if contain_feedback :
        if contain_feedback == ":brain: prediction":
            pred(st=st)
        if contain_feedback == ":writing_hand: Project description":
            project(st=st)
        if contain_feedback == ":recycle: Introduction":
            intro(st=st)
        if contain_feedback == ":desktop_computer: Modelling":
            modeling(st=st)
        if contain_feedback == ":stars: Conclusion":
            conclusion(st=st)
        else: pass 
    else :
        if yolo_feedback_contrain :  
            st.code(code(yolo_feedback_contrain), language='python', line_numbers=True)    
        else: 
            factor = None

            st.write('<h2 class="custom-text-under">Object Detection</h2>', unsafe_allow_html=True)
            with st.expander("HAVE A LOOK"):
                tab_od = st.tabs(['image 1 pred',  'image 2', 'image 2 pred'])
                image_location = ['./images/img_pred.jpg', './images/image2.jpg', 
                                  './images/image2_pred.jpg']
                
                types = ['image', 'image', 'image']

                for i in range(len(tab_od)):
                    with tab_od[i]:
                        head_img(st=st, factor=factor, types=types[i], path=image_location[i])

            st.write('<h2 class="custom-text-under2">Semantic image Segmentation</h2>', unsafe_allow_html=True)
            with st.expander("HAVE A LOOK"):
                tab_seg = st.tabs(['image 1', 'image 1 pred', 'image 2', 'image 2 pred'])
                image_location = ['./images/image2.jpg', './images/img_seg.png', './images/image3.jpg', './images/image3_seg.png']
                
                types = ['image', 'image', 'image', 'image']

                for i in range(len(tab_seg)):
                    with tab_seg[i]:
                        head_img(st=st, factor=factor, types=types[i], path=image_location[i])
                
            st.write('<h2 class="custom-text-under3"> OCR of Plates and Object Detection</h2>', unsafe_allow_html=True)
            with st.expander("HAVE A LOOK"):
                head_img(st=st, path='./images/tracked.jpg', factor=factor)

            st.write('<h2 class="custom-text-under4">Object Tracking</h2>', unsafe_allow_html=True)
            with st.expander("HAVE A LOOK"):
                tab_track = st.tabs(['Tracking objects'])
                image_location = ['./video/yolo_pred.mp4']
                types = ['video']

                for i in range(len(tab_track)):
                    with tab_track[i]:
                        head_img(st=st, types=types[i], path=image_location[i])

if __name__ == '__main__':
    head()