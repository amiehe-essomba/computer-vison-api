def slider_video(st, *details):
    import streamlit as st 
    # [fps, video_frame, duration]
    [fps, video_frame, duration] = details 

    with st.expander(label='Video Details'):
        st.write(f"frame per second : {fps}, video frame : {video_frame}, duration : {round(duration, 4)}s")

    if duration <= 18.0 :
        col0, col1, col3 = st.columns(3)
        with col0:
            d = int(duration) - 1
            second  = st.slider('time(s)', min_value=1, max_value=d, step=1, value=1)
            if second > duration:  second = duration
            else: pass 
        with col1:
            start   = st.slider('start frame', min_value=0, max_value=int(video_frame-fps * second), value=0, step=int(fps))
        with col3:
            step = st.slider('step', min_value=1, max_value=10, value=1, step=1)
        end = int(start + fps * second)
        if end > video_frame: end = int(video_frame) - 1
    else:
        st.warning("Video too long > 18.0s, set video range to avoid streamlit cloud memory Error", icon="⚠️")

        col0, col1, col3 = st.columns(3)

        with col0:
            second  = st.slider('time(s)', min_value=1, max_value=18, step=1, value=1)
        with col1:
            start   = st.slider('start frame', min_value=0, max_value=int(video_frame-fps * second), value=0, step=int(fps))
        with col3:
            step    = st.slider('step', min_value=1, max_value=10, value=1, step=1)

        end = int(start + fps * second)

    return start, end, step

def slider_model(st, locked = False):
    col1, col2, col3 = st.columns(3)
 
    with col1:
        iou_threshold   = st.slider('iou threshold', max_value=1.0, min_value=0.0, step=0.1, value=0.5, disabled=locked)
    with col2:
        score_threshold = st.slider('score threshold', max_value=1.0, min_value=0., step=0.1, value=0.4)
    with col3:
        max_boxes       = st.slider('max boxes', max_value=50, min_value=1, step=1, value=20, disabled=locked) 

    return [iou_threshold, score_threshold, max_boxes]