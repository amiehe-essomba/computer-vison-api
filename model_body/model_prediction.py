import tensorflow as tf  
import pandas as pd 
import cv2
from yolo.utils.tools import total_precess, total_precess_youtuble
from streamlit_modules.file_read import file_read, online_link, camera, youtube
from yolo.utils.tools import read_classes, read_anchors
from yolo.predictions import prediction
from streamlit_modules.button_style import button_style
import plotly.express as px
import logging
from yolo import video_settings as vs 
from yolov8_ops import ocr_yolov8, ocr, yolov8, yolov8_seg, yolov8_pose
import streamlit 
from model_in_cache.models import load_all_models
from collections import defaultdict

def pred(st : streamlit):
    all_models = load_all_models()
    modeltype  = {
        "segmentation" : [("yolov8-seg", ""), True],
        "detection"    : [("my model", "yolov8"), False],
        "ocr"          : [("ocr", ""), True],
        "counting"     : [("yolov8", ''), True],
        "localization" : [("my model", ""), True],
        "tracking"     : [("yolov8", "yolov8-seg"), False],
        "pose"         : [("yolov8-pose", ""), True]
        }

    st.write('<style>{}</style>'.format(styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Welcome in prediction section</h1>', unsafe_allow_html=True)

    for i in range(2):
        st.write('', unsafe_allow_html=True)

    yolo_model_path = './yolo_model/' 
    is_yolo = False

    # three columns for local file, show file updated, image scale factor 
    col1, col2, col3, col4 = st.columns(4)
    
    locked_mod = True
    with col1:
        label_select = st.selectbox('Local or Online File', options=('Local', 'Online', 'Camera', 'YouTube'), index=None)
        if label_select: locked_mod = False
        else: locked_mod = True
    
    with col2:
        show = st.checkbox('Show files uploaded', disabled=locked_mod, value=True)
        st.write('state', show)

    with col3:
        method_cal = st.selectbox('Select method', options=("segmentation", "detection", "localization", "counting", 
                                    "ocr", "tracking", "pose"), index=None, disabled=locked_mod)
        tracking = True if method_cal=="tracking" else False 

    with col4:
        #if show : desable_scale = False 
        #desable_scale  = locked_mod
        if method_cal:
            disable_scale = modeltype[method_cal][1]
            options = modeltype[method_cal][0]
            model_type = st.selectbox(label='Select models', options=options, disabled=disable_scale, index=0)
            if model_type == "my model":
                is_yolo = False 
            else: is_yolo = True
        else: model_type = None 

    if model_type in ['ocr', 'ocr+yolov8']: factor = True 
    else: factor = False 
    
    if model_type:
        if model_type == 'my model': locked = False 
        else: locked = True 

        if   label_select == 'Local' :
            if label_select: 
                uploaded_file = st.file_uploader("upload local image or video", 
                                                type=["jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi", "mkv"],
                                                accept_multiple_files=True
                                                )
                if uploaded_file:
                    if show : show = True 
                    # get the informations about the video or image s
                    # video, image, image shape, and more ...
                    
                    all_files = file_read(st, uploaded_file=uploaded_file, show=False, factor=factor)
                    [iou_threshold, score_threshold, max_boxes] = vs.slider_model(st=st, locked=locked)

                    # classses and anchors 
                    Class_names         = read_classes()
                    anchors             = read_anchors()

                    # five columns for use all classes, class probabilitiess etc....
                    cp_col1, cp_col2, cp_col3, cp_col4, cp_col5 = st.columns(5)

                    with cp_col2:
                        cp_check = st.checkbox("Use all classes")
                        st.write('state', cp_check)

                    with cp_col1:
                        # if use all is true cp_disable become True 
                        if cp_check :  cp_disable = True  
                        else: cp_disable  = False 

                        # select multi-classs probabilities 
                        class_names = st.multiselect("Classes", options=Class_names, disabled=cp_disable)

                        if class_names: pass 
                        else: 
                            # Class_names can be different ro class_name 
                            if cp_check : class_names = Class_names
                            else: pass 
                            
                    if class_names:
                        with cp_col3:
                            # select file type (video or image)
                            file_type = st.radio('File Types', options=('image', 'video'))
                        
                        if file_type:
                            # index of one file (video or image)
                            if all_files[f'{file_type}']: indexes = range(len( all_files[f'{file_type}']))
                            else: indexes = None 

                            if indexes: 
                                with cp_col4:
                                    # select iindex of file
                                    index = st.selectbox('Select file index', options=indexes)
                                
                                if index >= 0:
                                    items  = {
                                    'class_names' : class_names,
                                    'anchors' : anchors,
                                    'Class_names' : Class_names,
                                    'max_boxes' : max_boxes,
                                    'score_threshold' : score_threshold,
                                    'iou_threshold' : iou_threshold,
                                    "method_cal" : method_cal
                                    }
                                    
                                    df = {'label' : [], 'score':[], 'top':[], "left":[], "bottom":[], 'right':[]}

                                    if file_type == 'image':
                                        items['image_file'] = [all_files['image'][index]]
                                        shape = all_files['image_shape'][index][:-1]
                                        Image(st=st, all_models=all_models, df=df, col=cp_col5, 
                                              shape=shape, model_type=model_type, show=show, **items)
                                    else:
                                        details = tuple( all_files['details'][index])
                                        details = vs.slider_video(st, *details)
                                        video   = all_files['video_reader'][index]
                                        Video(st=st, prediction=prediction, all_models=all_models, video=video, model_type=model_type,
                                                        df=df, details=details, show=show, tracking=tracking, youtube=False, **items)
                                else: pass
                            else: pass
                        else: pass 
                    else: pass
                else: pass
            else: pass 
        elif label_select == 'Online':
            if show : show = True 

            type_of_file ="image" #st.selectbox('File type', options=('image', 'video'), index=None)

            if type_of_file:
                url = st.text_input("Insert your image url here please :")
            else: url = ""

            if url:
                if type_of_file == 'image':
                    image, image_data, shape,  error = online_link(st=st, url=url, is_yolo=is_yolo)

                    if error is None:
                        [iou_threshold, score_threshold, max_boxes] = vs.slider_model(st=st, locked=locked)

                        # classses and anchors 
                        Class_names         = read_classes()
                        anchors             = read_anchors()

                        # five columns for use all classes, class probabilitiess etc....
                        cp_col1, cp_col2, cp_col3 = st.columns(3)

                        with cp_col2:
                            cp_check = st.checkbox("use all classes")
                            st.write('state', cp_check)

                        with cp_col1  :
                            # if use all is true cp_disable become True 
                            if cp_check :  cp_disable = True  
                            else: cp_disable  = False 

                            # select multi-classs probabilities 
                            class_names = st.multiselect("class probabilities", options=Class_names, disabled=cp_disable)

                            if class_names: pass 
                            else: 
                                # Class_names can be different ro class_name 
                                if cp_check : class_names = Class_names
                                else: pass 

                        if class_names: 
                            items  = {
                                'class_names' : class_names,
                                'anchors' : anchors,
                                'Class_names' : Class_names,
                                'max_boxes' : max_boxes,
                                'score_threshold' : score_threshold,
                                'iou_threshold' : iou_threshold,
                                'image_file'   : [(image, image_data)],
                                "method_cal" : method_cal
                                }
                            if type_of_file == 'image':
                                #tf.get_logger().setLevel(logging.ERROR)
                                #yolo_model = tf.keras.models.load_model(yolo_model_path, compile=False)
                                
                                df = {'label' : [], 'score':[], 'top':[], "left":[], "bottom":[], 'right':[]}
                                Image(st=st, all_models=all_models, df=df, col=cp_col3, shape=shape, model_type=model_type,
                                    show=show, **items) 
                            else: pass
                        else: pass
                    else: st.warning(f'{error}')
                else:
                    st.write("Cannot upload this video please use YouTube instead")
            else: pass
        elif label_select == 'Camera':
            image, image_data, shape = camera(st=st)

            if image:
                [iou_threshold, score_threshold, max_boxes] = vs.slider_model(st=st)

                # classses and anchors 
                Class_names         = read_classes()
                anchors             = read_anchors()

                # five columns for use all classes, class probabilitiess etc....
                cp_col1, cp_col2, cp_col3 = st.columns(3)

                with cp_col2:
                    cp_check = st.checkbox("use all classes")
                    st.write('state', cp_check)
                
                with cp_col1  :
                    # if use all is true cp_disable become True 
                    if cp_check :  cp_disable = True  
                    else: cp_disable  = False 

                    # select multi-classs probabilities 
                    class_names = st.multiselect("class probabilities", options=Class_names, disabled=cp_disable)

                    if class_names: pass 
                    else: 
                        # Class_names can be different ro class_name 
                        if cp_check : class_names = Class_names
                        else: pass 

                if class_names: 
                    items  = {
                        'class_names' : class_names,
                        'anchors' : anchors,
                        'Class_names' : Class_names,
                        'max_boxes' : max_boxes,
                        'score_threshold' : score_threshold,
                        'iou_threshold' : iou_threshold,
                        'image_file'   : [(image, image_data)],
                        "method_cal" : method_cal
                        }
              
                    df = {'label' : [], 'score':[], 'top':[], "left":[], "bottom":[], 'right':[]}
                    Image(st=st, all_models=all_models, df=df, col=cp_col3, shape=shape, model_type=model_type,
                        show=show, **items) 
               
                else: pass
            else: pass
        else:
            url = st.text_input("Insert YouTube url here please :")
            
            if url:
                if show : show = True 

                #if model_type == 'my model': locked = False 
                #else: locked = True 
                
                [iou_threshold, score_threshold, max_boxes] = vs.slider_model(st=st, locked=locked)

                # classses and anchors 
                Class_names         = read_classes()
                anchors             = read_anchors()
                

                # five columns for use all classes, class probabilitiess etc....
                cp_col1, cp_col2, _, _, _ = st.columns(5)

                with cp_col2:
                    cp_check = st.checkbox("Use all classes")
                    st.write('state', cp_check)

                with cp_col1:
                    # if use all is true cp_disable become True 
                    if cp_check :  cp_disable = True  
                    else: cp_disable  = False 

                    # select multi-classs probabilities 
                    class_names = st.multiselect("Classes", options=Class_names, disabled=cp_disable)

                    if class_names: pass 
                    else: 
                        # Class_names can be different ro class_name 
                        if cp_check : class_names = Class_names
                        else: pass 
                
                if class_names:
                    
                    items  = {
                    'class_names' : class_names,
                    'anchors' : anchors,
                    'Class_names' : Class_names,
                    'max_boxes' : max_boxes,
                    'score_threshold' : score_threshold,
                    'iou_threshold' : iou_threshold, 
                    "method_cal" : method_cal
                    }

                    df = {'label' : [], 'score':[], 'top':[], "left":[], "bottom":[], 'right':[]}
                    video, *details = youtube(st=st, url=url) 
                    items['fps'] = details[0]

                    if video:
                        details = tuple(details)
                        details = vs.slider_video(st, *details)   

                        Video(st=st, prediction=prediction, all_models=all_models, video=video, model_type=model_type,
                                df=df, details=details, show=show, tracking=tracking, youtube=True, **items)
                    else: pass 
                else: pass
            else: pass

def Image(st:streamlit, all_models:dict, df, col, shape, model_type, show, **kwargs):
    import numpy as np 
    from yolo.utils.tools import get_colors_for_classes
    import random

    class_names = kwargs['Class_names']
    colors_     = get_colors_for_classes(len(class_names) + 10)
  
    def f():
        s = random.sample(range(50), 1)
        return s[0] 
    
    def g():
        num = random.sample(range(len(colors_)), len(class_names))
        return num 
    
    colors      = {class_names[j] : colors_[i] if colors_[i] != (255, 255, 0) else colors_[j-1] for j, i in enumerate(g())}

    with col:
        if model_type == 'yolov8-pose':
            response = st.checkbox('With scores', disabled=False)
        else: 
            response = st.checkbox('With scores', disabled=False)
        st.write('state', response) 

    if model_type == 'yolov8-seg':
        ctt1, ctt2, ctt3, ctt4 = st.columns(4)
        with ctt1:
            alpha = st.slider('alpha', min_value=1, max_value=255, value=30, step=1)
        with ctt2:
            mode = st.selectbox('Background Mode', options=('gray', 'rbg'), index=0)
        with ctt3:
            only_mask = st.checkbox('Only Mask')
            st.write('status', only_mask)
        with ctt4:
            with_names = st.checkbox('With Names')
            st.write('status', with_names)
    
    if model_type == 'yolov8-pose':
        ctt1, ctt2, ctt3 = st.columns(3)
        with ctt1:
            radius = st.slider('radius', min_value=1, max_value=20, value=1, step=1)
        with ctt2:
            line_width = st.slider('line width', min_value=1, max_value=20, step=1)
        with ctt3:
            with_cls = st.checkbox('with Oject detection')
            st.write('state', with_cls)

    if model_type == 'my model': grad_cam_dis = False 
    else: grad_cam_dis = True 

    grad_cam_col, area_of_in, c_ = st.columns(3)

    with grad_cam_col:
        grad_cam = st.checkbox("show gradCAM", disabled=grad_cam_dis)
        st.write('state', grad_cam)
    
    with area_of_in:
        is_area = st.checkbox('Draw Area', disabled=True)
        st.write('state', is_area)

        domaine_area = {}

        if is_area:
            x, y = st.columns(2)
            with x:
                Ax = st.select_slider("coordinates A(x)", options=range(shape[1]), value=0)
                st.write(f'A(x) = {Ax}')
            with y :
                Ay = st.select_slider("coordinates A(y)", options=range(shape[0]), value=0)
                st.write(f'A(y) = {Ay}')
            A = [Ax, Ay]

            x, y = st.columns(2)
            with x:
                Ax = st.select_slider("coordinates B(x)", options=range(shape[1]), value=0)
                st.write(f'B(x) = {Ax}')
            with y :
                Ay = st.select_slider("coordinates B(y)", options=range(shape[0]), value=0)
                st.write(f'B(y) = {Ay}')
            B = [Ax, Ay]

            x, y = st.columns(2)
            with x:
                Ax = st.select_slider("coordinates C(x)", options=range(shape[1]), value=0)
                st.write(f'B(x) = {Ax}')
            with y :
                Ay = st.select_slider("coordinates C(y)", options=range(shape[0]), value=0)
                st.write(f'B(y) = {Ay}')
            C = [Ax, Ay]

            x, y = st.columns(2)
            with x:
                Ax = st.select_slider("coordinates D(x)", options=range(shape[1]), value=0)
                st.write(f'B(x) = {Ax}')
            with y :
                Ay = st.select_slider("coordinates D(y)", options=range(shape[0]), value=0)
                st.write(f'B(y) = {Ay}')
            D = [Ax, Ay]

            domaine_area = dict(A=A, B=B, C=C, D=D)

    with c_:
        font = st.selectbox("Font", options=["arial", 'arialbd', "calibri", "calibril", 
                                      'consolai', "consolab", "calibriz", "corbell", 'micross', "firamono"], index=3)
        
        font = f"./font/{font}.ttf" if font != "firamono" else f"./font/{font}.otf"
        #font = './font/FiraMono-Medium.otf'
    #with run_data_col:
    run_data = button_style(st=st, name='run')

    if run_data:
        if model_type == 'my model':
            tf.get_logger().setLevel(logging.ERROR)
            yolo_model = all_models['my_model'] 
            #tf.keras.models.load_model(yolo_model_path, compile=False)
            df = {'label' : [], 'score':[], 'top':[], "left":[], "bottom":[], 'right':[]}

            image_predicted = prediction(
                yolo_model=yolo_model, use_classes=kwargs['class_names'],
                image_file=kwargs['image_file'], anchors=kwargs['anchors'], class_names=kwargs['Class_names'], img_size=(608, 608),
                max_boxes=kwargs['max_boxes'], score_threshold=kwargs['score_threshold'], iou_threshold=kwargs['iou_threshold'], data_dict=df,
                shape=shape, file_type='image', with_score=response, colors=colors, grad_cam=grad_cam, area=domaine_area, font=font,
                type_of_cal=kwargs['method_cal']
            )
            if grad_cam is False: pass 
            else:
                grad, guided_grad = image_predicted[1:]
                image_predicted   = image_predicted[0]

            resume(st=st, df=df, show=show, img = kwargs['image_file'][0][0], **{"image_predicted" : image_predicted})

            if grad_cam:
                with st.expander('see the gradCAM here'):
                    g1, g2 = st.columns(2)

                    with g1:
                        st.header('GradCAM')
                        st.image(grad)
                    with g2:
                        st.header('Guided gradCAM')
                        st.image(guided_grad)
        
        if model_type == 'yolov8':
            model = all_models['yolov8n.pt']
            if kwargs["method_cal"] == 'detection':
                yolov8.yolov8(st, df, shape, show, response, resume, False, colors, model, font, **kwargs)
            else:
                track_history   = defaultdict(lambda: []) 
                tracker = ("bytetrack.yaml", "botsort.yaml")
                yolov8.yolov8_track(st, df, shape, show, response, resume, False, colors, tracker[1], track_history,
                                     model, font, **kwargs)
        
        if model_type == 'yolov8-seg':
            model = all_models["yolov8n-seg.pt"]
            yolov8_seg.yolov8_seg(st, df, shape, show, response, resume, False, colors, alpha, mode, 
                                  only_mask, with_names, model, font, **kwargs)
        
        if model_type == 'yolov8-pose':
            model = all_models["yolov8n-pose.pt"]
            yolov8_pose.yolov8_pose(st=st, df=df, colors=colors, radus=radius, line_width = line_width, 
                        shape=shape, resume=resume, shwo=show, response=response, od=with_cls, model=model, font=font, **kwargs)
            
        if model_type == 'ocr+yolov8':
            models = [all_models['license_plate_detector.pt'], all_models['yolov8n.pt']]
            ocr_yolov8.ocr_yolov8(st, df, shape, show, response, resume, scaling, False, colors, models, **kwargs)

        if model_type == 'ocr':
            model = all_models['license_plate_detector.pt']
            ocr.ocr(st, df, shape, show, response, resume, scaling, colors, model, font=font, **kwargs)

        if model_type == 'yolov8-cls':
            pass
    else: pass

def scaling(image = None, shape = (608, 608), boxes = None, S = None):
    import cv2

    # Mettez à l'échelle l'image
    scaled_image    = image.resize(shape)
    new_width       = shape[0]
    new_height      = shape[1]
    # Mettez à l'échelle les boîtes englobantes
    scaling_factor_x = new_width / S[0]
    scaling_factor_y = new_height / S[1]

    scaled_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        scaled_x_min = int(x_min * scaling_factor_x)
        scaled_y_min = int(y_min * scaling_factor_y)
        scaled_x_max = int(x_max * scaling_factor_x)
        scaled_y_max = int(y_max * scaling_factor_y)
        scaled_boxes.append([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max])

    return scaled_boxes, scaled_image

def Video(st, prediction, all_models:dict, video, df, details, show, model_type, tracking, youtube:bool=False, **kwargs):

    from yolo.utils.tools import get_colors_for_classes
    import random
    import streamlit as st 

    class_names = kwargs['Class_names']
    colors_     = get_colors_for_classes(len(class_names) + 100)

    def f():
        s = random.sample(range(50), 1)
        return s[0] 
    
    def g():
        num = random.sample(range(len(colors_)), len(class_names))
        return num 
    
    colors      = {class_names[j] : colors_[i] if colors_[i] != (255, 255, 0) else colors_[j-1] for j, i in enumerate(g())}

    items  = {
            'class_names' : kwargs['class_names'],
            'anchors' : kwargs['anchors'],
            'Class_names' : kwargs['Class_names'],
            'max_boxes' : kwargs['max_boxes'],
            'score_threshold' : kwargs['score_threshold'],
            'iou_threshold' : kwargs['iou_threshold'],
            "fps" : kwargs['fps'] if "fps" in list(kwargs.keys()) else None
            }
    
    if model_type == 'my model':
        yolo_model = all_models['my_model']
        if youtube is False:
            video_reader, fps = total_precess(st=st, prediction=prediction, 
                            estimator=yolo_model, video=video, df=df, details=details, colors=colors, **items)
        else:
            video_reader, fps = total_precess_youtuble(st=st, prediction=prediction, 
                            estimator=yolo_model, video=video, df=df, details=details, colors=colors, **items)
        if video_reader:
            resume(st=st, df=df, file_type='video', show=show, **{'fps' : fps, 'video_reader' : video_reader})
        else: pass 
    else:
        if (model_type == 'yolov8') or (model_type == 'yolov8-seg') or  (model_type =='ocr+yolov8'):
            ct1, ct2, ct3, ct4 = st.columns(4)
            dis = False if tracking else True
            
            with ct1:
                tracker = st.selectbox('Tracking models', ("bytetrack.yaml", 
                                                    "botsort.yaml"), disabled=dis, index=0)
            with ct2:
                response = st.checkbox("With score")
                st.write(response)
            with ct3:
                import streamlit as st 
                track_all = st.checkbox("Track all", disabled=dis)
                st.write(response)
            with ct4:
                if dis is False: dis = track_all
                else: pass

                track_num = st.multiselect('Object id', options=[str(x) for x in range(len(class_names))], disabled=True, default='0')
                
                if track_all: track_num = [x for x in range(len(class_names))]
                else:  track_num = [int(float(x)) for x in track_num]
            
            if model_type != "yolov8":  
                ctt1_, ctt2_ = st.columns(2)
                with ctt1_:
                    if model_type == 'yolov8-seg':
                        only_mask = st.checkbox('Only Mask')
                    else:
                        only_mask = st.checkbox('Only Mask', disabled=True)
                    st.write('status', only_mask)
                with ctt2_:
                    with_names = st.checkbox('With Names')
                    st.write('status', with_names)
                #with ctt3_:
                #    run = st.button('run')
           
            if tracking is False:
                if model_type == 'yolov8':
                    run = st.button('run')
                    model = all_models['yolov8n.pt']
                    if run:
                        if youtube is False:
                            yolov8.yolovo_video(st, video, df, details, show, resume, 
                                            response, run, colors, model, **items)
                        else:
                            yolov8.yolovo_video_youtube(st, video, df, details, show, resume, 
                                            response, run, colors, model, **items)
                if model_type == 'yolov8-seg':
                    model = all_models["yolov8n-seg.pt"]
                    ctt1, ctt2, ctt3 = st.columns(3)

                    with ctt1:
                        alpha = st.slider('alpha', min_value=1, max_value=255, value=30, step=1)
                    with ctt2:
                        mode = st.selectbox('Background Mode', options=('gray', 'rbg'), index=0)
                    with ctt3:
                        run = st.button('run')
                    if run:
                        yolov8_seg.yolovo_video_seg(st, video, df, details, show, resume, response, run, 
                                                    colors, alpha, mode, only_mask, with_names, model, **items)
                if model_type == 'ocr+yolov8':
                    models = [all_models['license_plate_detector.pt'], all_models['yolov8n.pt']]
                    run = st.button('run', key='run 1')
                    if run:
                        ocr_yolov8.ocr_yolovo_video(st, video, df, details, show, resume, scaling, 
                                                    response, run, colors, models, **items)
            else:
                #if track_num:
                track_num = [0]
                model = all_models['yolov8n.pt']
                run = st.button('run')
                if run:
                    if youtube is False:
                        yolov8.yolov8_video_track(st, video, df, details, show, resume, 
                                              response, run, colors, tracker, track_num, model=model, **items)
                    else:
                        yolov8.yolovo_video_youtube_track(st, video, df, details, show, resume, 
                                              response, run, colors, tracker, track_num, model=model, **items)
                  
def resume(st, df : dict, file_type: str='image', img=None, show=True, **kwargs):
    with st.expander("MODEL PERFORMANCES"):
        st.write("""
            Adjusting the parameters above allows you to fine-tune your 
            predictions for improving accuracy.

            Great job, you've done excellently!
        """)

        if file_type == 'image': 
            if show is False : 
                st.header("Prediction")
                st.image(kwargs['image_predicted'])
            else:
                c1, c2 = st.columns(2)
                pred = kwargs['image_predicted']
                with c1:
                    st.header("True image")
                    st.image(img.resize((pred.shape[1], pred.shape[0])))
                with c2:
                    st.header("Prediction")
                    st.image(pred)
        else :
            st.write(f"frame rate per second : {kwargs['fps']}") 
            st.video(kwargs['video_reader'])

        if df['label']:
            #fig_col1, fig_col2 = st.columns(2)
            data_frame = pd.DataFrame(df)
            data_frame.rename(columns={'label':'classes'}, inplace=True)
            data_frame['label'] = [1 for i in range(len(data_frame.iloc[:, 0]))]

            #with fig_col1:
            
            st.dataframe(data=data_frame.style.highlight_max(axis=0, color='skyblue', subset=['classes', 'score']))
            fig = px.pie(data_frame, names='classes', values='label', title='pie chart')

            #with fig_col2:                                                
            st.plotly_chart(fig, use_container_width=True)

        else: pass 

def styles():
     
    custom_css_title = """
        .header-text {
            color: black; /* Couleur du texte */
            /*background-color: white; Couleur de l'arrière-plan */
            font-size: 25px; /* Taille de police */
            font-weight: bolder; /* Gras */
            text-decoration: none; /* Souligné underline overline */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
            background-image: darkgray;
            border-radius: 5px; /* Coins arrondis */
            margin: 3px; /* Marge extérieure */
            border: 5px solid deepskyblue; /* Bordure */
            padding: 5px; /* Marge intérieure pour le texte */
            display: inline-block;
            box-shadow: 2px 4px 3px 0 rgba(20, 0, 0.5, 5); /* Ombre */
        }
        """
    return custom_css_title