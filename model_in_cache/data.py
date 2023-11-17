import streamlit as st 

@st.cache_resource()
def data_cache():
    PATH            = "./video/"         
    names           = ["yolo_video2.mp4", "yolo_video2_pred.mp4", "yolo_video3_pred.mp4"]
    all_videos      = []

    for name in names:
        abs_path = f"{PATH}{name}"
        video_file = open(abs_path, 'rb')
        video_bytes = video_file.read()
        all_videos.append(video_bytes)

    return all_videos