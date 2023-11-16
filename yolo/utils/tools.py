import colorsys
import random
import numpy as np
import tempfile
import imageio
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.transform import resize
from keras import backend as K
from functools import reduce
from streamlit_modules.button_style import button_style
from stqdm import stqdm 

def preprocess_image(img_path, model_image_size, done : bool = False, factor = False):
    #image_type = imghdr.what(img_path)
    if done is False : image           = Image.open(img_path)
    else: image = img_path

    shape           = np.array(image).shape
    resized_image   = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data      = np.array(resized_image, dtype='float32')
    image_data /= 255.
    # Add batch dimension.
    image_data      = np.expand_dims(image_data, 0) 

    if factor is False : return image.resize(model_image_size), image_data, shape
    else: return image, image_data, shape

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def read_classes(classes_path : str = './data/coco_classes.txt'):

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path : str = './data/yolo_anchors.txt'):

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height          = float(image_shape[0])
    width           = float(image_shape[1])
    image_dims      = K.stack([height, width, height, width])
    image_dims      = K.reshape(image_dims, [1, 4])
    boxes           = boxes * image_dims
    return boxes

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.

    return colors

def draw_boxes(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], df = {}, with_score : bool = True, colors=None):
    """
    Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
    #colors      = get_colors_for_classes(len(class_names))

    
    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score   = scores.numpy()[i]
            label   = '{} {:.2f}'.format(box_class, score)
        else: label = '{}'.format(box_class)

        
        _label_ = label.split()
        if len(_label_) <= 2 : 
            if with_score : pass 
            else : label = _label_[0]
        else:
            string = ""
            for i, s in enumerate(_label_[:-1]) : string = string + s + " " if (i != len(_label_)-2) else string  + s
            _label_ = [string, _label_[-1]]

            if with_score : pass 
            else: label = string

        LABEL = _label_[0]
        if LABEL in use_classes:
            draw        = ImageDraw.Draw(image)
            label_size  = draw.textlength(text=label, font=font)
            top, left, bottom, right = box
            top         = max(0, np.floor(top + 0.5).astype('int32'))
            left        = max(0, np.floor(left + 0.5).astype('int32'))
            bottom      = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right       = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            df['top'].append(top)
            df['left'].append(left)
            df['bottom'].append(bottom)
            df['right'].append(right)
            df['score'].append(float(_label_[1]))
            df['label'].append(_label_[0])


            """
            if top -  >= 0: #label_size[1] >= 0:
                text_origin = np.array([left, top - label_size]) # label_size[1]])
            else:  text_origin = np.array([left, top + 1])
            """

            if (top - 20) >= 0 : text_origin = np.array([left, top - 20])
            else:
                idd = 0
                while (top - 20 + idd) < 0:
                    idd += 1
                text_origin = np.array([left, top - 20 + idd])
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                try:
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i], outline=colors[LABEL]
                        )
                except ValueError:
                    done = None 
                    if left + i >= right - i:
                        draw.rectangle(
                            [left + i, top + i, left + i + abs(left-right), bottom - i], outline=colors[LABEL]
                            )
                        done = True 

                    if top + i >= bottom - i :
                        if done is True : 
                            draw.rectangle(
                                [left + i, top - i, left + i + abs(left-right), top + i + abs(top-bottom)], outline=colors[LABEL]
                                )
                        else:
                            draw.rectangle(
                                [left + i, top - i, right - i, top + i + abs(top-bottom)], outline=colors[LABEL]
                                )

                        
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + (label_size, 20))],
                fill=colors[LABEL]
                )
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        else : pass 

    return np.array(image)

def draw_boxes_v8(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], colors = None,
                  df = {}, with_score : bool = True, C =None, return_sequence=False, width=2, ids = None):
    """
    Draw bounding boxes on image.

    Draw bounding boxes with class name and optional box score on image.

    Args:
        image: An `array` of shape (width, height, 3) with values in [0, 1].
        boxes: An `array` of shape (num_boxes, 4) containing box corners as
            (y_min, x_min, y_max, x_max).
        box_classes: A `list` of indicies into `class_names`.
        class_names: A `list` of `string` class names.
        `scores`: A `list` of scores for each box.

    Returns:
        A copy of `image` modified with given bounding boxes.
    """

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
    j = 0
    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score   = scores.numpy()[i]
            label   = '{} {:.2f}'.format(box_class, score)
        else: label = '{}'.format(box_class)

        _label_ = label.split()

        if len(_label_) <= 2 : 
            if with_score : pass 
            else : label = _label_[0]
        else:
            string = ""
            for i, s in enumerate(_label_[:-1]) : string = string + s + " " if (i != len(_label_)-2) else string  + s
            _label_ = [string, _label_[-1]]
            if with_score : pass 
            else: label = string

        LABEL = _label_[0]

        if LABEL in use_classes:
            if type(ids) != type(None):  label += f' id:{int(ids.numpy()[j])}'
            draw        = ImageDraw.Draw(image)
            label_size  = draw.textlength(text=label, font=font)
            left, top, right, bottom = box
            j += 1
            df['top'].append(np.round(np.float32(top), 2) ) 
            df['left'].append(np.round( np.float32(left), 2))
            df['bottom'].append(np.round(np.float32(bottom), 2))
            df['right'].append(np.round( np.float32(right), 2))
            df['score'].append(float(_label_[1]))
            df['label'].append(_label_[0])

            if (top - 20) >= 0 : text_origin = np.array([left, top - 20])
            else:
                idd = 0
                while (top - 20 + idd) < 0:
                    idd += 1
                text_origin = np.array([left, top - 20 + idd])
      
            if C : colors[LABEL] = C
            draw.rectangle(
                [left, top, right, bottom], outline=colors[LABEL], width=width, fill=None
                )
                      
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + (label_size, 20))],
                fill=colors[LABEL]
                )
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
           
            del draw
        else : pass 

    if return_sequence is False: return  np.array(image)
    else:  return image
    
def draw_boxes_v8_seg(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], colors=None,
                  df = {}, with_score : bool = True, with_names=True, alpha = 30, only_mask:bool=False):

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
   
    temp_images = []

    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score   = scores.numpy()[i]
            label   = '{} {:.2f}'.format(box_class, score)
        else: label = '{}'.format(box_class)

        _label_ = label.split()
        if len(_label_) <= 2 : 
            if with_score : pass 
            else : label = _label_[0]
        else:
            string = ""
            for i, s in enumerate(_label_[:-1]) : string = string + s + " " if (i != len(_label_)-2) else string  + s
            _label_ = [string, _label_[-1]]

            if with_score : pass 
            else: label = string
        LABEL = _label_[0]

        if LABEL in use_classes:
            temp_image  = Image.new("RGBA", image.size, (0, 0, 0, 0))
            temp_draw   = ImageDraw.Draw(temp_image) 
            label_size  = temp_draw.textlength(text=label, font=font)
            left, top, right, bottom = box
        
            df['top'].append(np.round(np.float32(top), 2) ) 
            df['left'].append(np.round( np.float32(left), 2))
            df['bottom'].append(np.round(np.float32(bottom), 2))
            df['right'].append(np.round( np.float32(right), 2))
            df['score'].append(float(_label_[1]))
            df['label'].append(_label_[0])

            if (top - 20) >= 0 : text_origin = np.array([left, top - 20])
            else:
                idd = 0
                while (top - 20 + idd) < 0:
                    idd += 1
                text_origin = np.array([left, top - 20 + idd])
            
            if only_mask is False:
                temp_draw.rectangle(
                    [left, top, right, bottom], outline=colors[LABEL]+(150,), width=2, fill=colors[LABEL]+(alpha,) 
                    )
            else:
                temp_draw.rectangle(
                    [left, top, right, bottom], outline=colors[LABEL]+(150,), width=2, fill=None 
                    )
            
            if with_names is True:        
                temp_draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + (label_size, 20))],
                    fill=colors[LABEL]+(255,) 
                    )
                temp_draw.text(text_origin, label, fill=(0, 0, 0, 255), font=font,  embedded_color=True)
            temp_images.append(temp_image)
        else : pass  

    result = image.convert("RGBA")

    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)
  
    return np.array(result)

def read_video(image):
    import imageio
    video_reader    = imageio.get_reader(image, mode='?')
    fps             = video_reader.get_meta_data()['fps']
    video_frame     = video_reader.count_frames()
    duration        = float(video_frame / fps)

    return video_reader, [fps, video_frame, duration]

def total_precess(st, prediction, estimator, video, df, details, colors, **kwargs):
    import time 

    storage             = []
    frame_count         = 0
    try: fps                 = video.get_meta_data()['fps']
    except AttributeError: fps = kwargs['fps']

    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    ct1, ct2, ct3 = st.columns(3)

    with ct1:
        typ_of_op = st.selectbox('type of operation', ('detection', 'tracking'), disabled=True)
    with ct2:
        response = st.checkbox("With score")
        st.write(response)
    with ct3:
        run = st.button('run')# button_style(st=st, name='run')

    if run:   
        # progress bar 
        s = st.empty()

        # Écrire le tableau NumPy dans une vidéo avec imageio
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            #for i, frame in enumerate(video):
            for i, frame in stqdm(enumerate(video), backend=False, frontend=True):
                if i in range(start, end, step):
                    frame                       = Image.fromarray(frame, mode='RGB')
                    frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True)        
                    frame_count                += 1
                    
                    image_predicted = prediction(yolo_model=estimator, use_classes=kwargs['class_names'],
                                        image_file=[(frame, frame_data)], anchors=kwargs['anchors'], 
                                        class_names=kwargs['Class_names'], img_size=(608, 608),
                                        max_boxes=kwargs['max_boxes'], score_threshold=kwargs['score_threshold'], 
                                        iou_threshold=kwargs['iou_threshold'], data_dict=df,shape=shape[:-1], 
                                        file_type='video', with_score = response, colors=colors
                                        )
                    
                    image_predicted = image_predicted.astype('float32')
                    writer.append_data(image_predicted)
                    s.write('banary writing in progress ...')
                else: pass

                if i == end:  break
              
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for image in storage:
                writer.append_data(image)

        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
            s.write('banary lecture in progress ...')

        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        return video_data, fps
    
    else: return None, None


def total_precess_youtuble(st, prediction, estimator, video, df, details, colors, **kwargs):
    storage             = []
    frame_count         = 0
    try: fps                 = video.get_meta_data()['fps']
    except AttributeError: fps = kwargs['fps']

    (start, end, step)  = details
    temp_video_file     = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    ct1, ct2, ct3 = st.columns(3)

    with ct1:
        typ_of_op = st.selectbox('type of operation', ('detection', 'tracking'), disabled=True)
    with ct2:
        response = st.checkbox("With score")
        st.write(response)
    with ct3:
        run = st.button('run')# button_style(st=st, name='run')

    if run:   
        # progress bar 
        s = st.empty()
        i = -1
        
        # Écrire le tableau NumPy dans une vidéo avec imageio
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            #for i, frame in enumerate(video):
            while (video.isOpened()):
                re, frame = video.read()
                
                if re:
                    i += 1
                    if i in range(start, end, step):
                        frame                       = Image.fromarray(frame, mode='RGB')
                        frame, frame_data, shape    = preprocess_image(img_path=frame, model_image_size = (608, 608), done=True)        
                        frame_count                += 1
                        
                        image_predicted = prediction(yolo_model=estimator, use_classes=kwargs['class_names'],
                                            image_file=[(frame, frame_data)], anchors=kwargs['anchors'], 
                                            class_names=kwargs['Class_names'], img_size=(608, 608),
                                            max_boxes=kwargs['max_boxes'], score_threshold=kwargs['score_threshold'], 
                                            iou_threshold=kwargs['iou_threshold'], data_dict=df,shape=shape[:-1], 
                                            file_type='video', with_score = response, colors=colors
                                            )
                        
                        image_predicted = image_predicted.astype('float32')
                        writer.append_data(image_predicted)
                        s.write('banary writing in progress ...')
                    else: pass

                    if i == end:  break
              
        with imageio.get_writer(temp_video_file.name, mode='?', fps=fps) as writer:
            for image in storage:
                writer.append_data(image)

        # Ouvrir le fichier temporaire en mode lecture binaire (rb)
        with open(temp_video_file.name, 'rb') as temp_file:
            # Lire le contenu du fichier temporaire
            video_data = temp_file.read()
            s.write('banary lecture in progress ...')
        video.release()
        shutil.rmtree(temp_video_file.name, ignore_errors=True)
        return video_data, fps
    
    else: return None, None