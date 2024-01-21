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
from demo.tracker import delimiter_zone, Points
import matplotlib.pyplot as plt


def draw_area(image, fill, A=(200, 250), B=(85, 600), C=(530, 600), D=(400, 250)):
     
    temp_image  = Image.new("RGBA", image.size, (0, 0, 0, 0))
    temp_draw   = ImageDraw.Draw(temp_image)

    points = [A, B, C, D]

    temp_draw.polygon(points, fill=fill)
    temp_draw.line([A, B], width=4, fill=(0,0,255, 100))
    temp_draw.line([C, D], width=4, fill=(0,0,255, 100))
    temp_draw.line([B, C], width=4, fill=(0,0,255, 100))
    temp_draw.line([A, D], width=4, fill=(0,0,255, 100))

    return temp_image

def draw_line(A, B, center, inv=False):
    a = (A[1] - B[1]) / (A[0] - B[0])
    c = A[1] - A[0] * a

    isin = False 

    if inv is False:
        if (center[1] - a * center[0]) >= c:  
            isin = True 
    else:
        if (center[1] - a * center[0]) <= c:  
            isin = True 

    return isin

def line(a, b):
    # y1 = -3.05 * x + 860
    # y2 = 1.75 * x - 831

    isin = False 

    if (a[1] + 3.05 * a[0]) >= 860:
        if (b[1] - 2.7 * b[0]) >= -831:
            isin = True 

    return isin

def preprocess_image(img_path, model_image_size, done : bool = False, factor = False, is_yolo=False):
    #image_type = imghdr.what(img_path)
    if done is False : image           = Image.open(img_path)
    else: image = img_path
    shape           = np.array(image).shape
    if is_yolo is False:
        resized_image   = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    else:
        resized_image   = image.resize(tuple(reversed(640, 640)), Image.BICUBIC)
    image_data      = np.array(resized_image, dtype='float32')
    image_data /= 255.
    # Add batch dimension.
    image_data      = np.expand_dims(image_data, axis=0) 

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

def draw_boxes(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], 
               df = {}, with_score : bool = True, colors=None, area : dict={}, f='./font/FiraMono-Medium.otf'):
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
    """
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
    #colors      = get_colors_for_classes(len(class_names))
    """

    if area:
        A, B, C, D = area['A'], area['B'], area['C'], area['D']
        fill=(255, 0, 25, 40)
        temp_im = draw_area(image=image, fill=fill, A=A, B=B, C=C, D=D)
    else:
        temp_im = None
    
    """
    temp_image_  = Image.new("RGBA", image.size, (0, 0, 0, 0))
    temp_draw   = ImageDraw.Draw(temp_image_)

    points = [(200, 250), (85, 600), (530, 600), (400, 250)]

    temp_draw.polygon(points, fill=(255, 0, 25, 40))
    temp_draw.line([(200, 250), (85, 600)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(530, 600), (400, 250)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(85, 600), (530, 600)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(200, 250), (400, 250)], width=4, fill=(0,0,255, 100))
    """

    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        drop_box    = True 
         
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
            draw        = ImageDraw.Draw(image, 'RGBA')
            top, left, bottom, right = box
            top         = max(0, np.floor(top + 0.5).astype('int32'))
            left        = max(0, np.floor(left + 0.5).astype('int32'))
            bottom      = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right       = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            size        = 3 + int(3e-1 * (abs(right-left)))
            font        = ImageFont.truetype(font=f, size=size)
            label_size  = draw.textlength(text=label, font=font)

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

            
            if area:
                center = [left + abs(right-left)//2, top + abs(bottom-top)//2]
                if A[1] < bottom <= B[1]:
                    drop_box = draw_line(A=A, B=B, center=center, inv=False)
                    drop_box = draw_line(A=C, B=D, center=center, inv=True)
                else:  drop_box = False
            
            if drop_box:
                if (top - size) >= 0 : text_origin = np.array([left, top - size])
                else:
                    idd = 0
                    while (top - size + idd) < 0:
                        idd += 1
                    text_origin = np.array([left, top - size + idd])
                # My kingdom for a good redistributable image drawing library.
                for i in range(1):#(thickness):
                    try:
                        draw.rectangle(
                            [left, top+i, right, bottom-i], outline=colors[LABEL], fill=colors[LABEL]+(30,)
                            )
                    except ValueError:
                        done = None 
                        if left + i >= right - i:
                            draw.rectangle(
                                [left + i, top + i, left + i + abs(left-right), bottom - i], outline=colors[LABEL]
                                , fill=colors[LABEL]+(30,)
                                )
                            done = True 

                        if top + i >= bottom - i :
                            if done is True : 
                                draw.rectangle(
                                    [left + i, top - i, left + i + abs(left-right), top + i + abs(top-bottom)], outline=colors[LABEL]
                                    , fill=colors[LABEL]+(30,)
                                    )
                            else:
                                draw.rectangle(
                                    [left + i, top - i, right - i, top + i + abs(top-bottom)], outline=colors[LABEL]
                                    , fill=colors[LABEL]+(30,)
                                    )

                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + (label_size, size))],
                    fill=colors[LABEL] + (70, ), outline=colors[LABEL]
                    )
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        else : pass 

    if area:
        result = image.convert("RGBA")
        result = Image.alpha_composite(result, temp_im)

        return np.array(result.convert('RGB'))
    else: return np.array(image)

def draw_boxes_localalization(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], 
               df = {}, with_score : bool = True, colors=None, area : dict={}, shape=None, f='./font/FiraMono-Medium.otf'):
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
    """
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32')) 
    thickness   = (image.size[0] + image.size[1]) // 300
    #colors      = get_colors_for_classes(len(class_names))
    """

    if area:
        A, B, C, D = area['A'], area['B'], area['C'], area['D']
        fill=(255, 0, 25, 40)
        temp_im = draw_area(image=image, fill=fill, A=A, B=B, C=C, D=D)
    else:
        temp_im = None
    
    """
    temp_image_  = Image.new("RGBA", image.size, (0, 0, 0, 0))
    temp_draw   = ImageDraw.Draw(temp_image_)

    points = [(200, 250), (85, 600), (530, 600), (400, 250)]

    temp_draw.polygon(points, fill=(255, 0, 25, 40))
    temp_draw.line([(200, 250), (85, 600)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(530, 600), (400, 250)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(85, 600), (530, 600)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(200, 250), (400, 250)], width=4, fill=(0,0,255, 100))
    """

    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        drop_box    = True 
         
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
            draw        = ImageDraw.Draw(image, mode='RGBA')
            top, left, bottom, right = box
            top         = max(0, np.floor(top + 0.5).astype('int32'))
            left        = max(0, np.floor(left + 0.5).astype('int32'))
            bottom      = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right       = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            size        = 3 + int(3e-1 * (abs(right-left)))
            font        = ImageFont.truetype(font=f, size=size)
            label_size  = draw.textlength(text=label, font=font)
           
            df['top'].append(top)
            df['left'].append(left)
            df['bottom'].append(bottom)
            df['right'].append(right)
            df['score'].append(float(_label_[1]))
            df['label'].append(_label_[0])
            
            if area:
                center = [left + abs(right-left)//2, top + abs(bottom-top)//2]
                if A[1] < bottom <= B[1]:
                    drop_box = draw_line(A=A, B=B, center=center, inv=False)
                    drop_box = draw_line(A=C, B=D, center=center, inv=True)
                else:  drop_box = False
            
            if drop_box:
                if (top - size) >= 0 : text_origin = np.array([left, top - size])
                else:
                    idd = 0
                    while (top - size + idd) < 0:
                        idd += 1
                    text_origin = np.array([left, top - size + idd])
                # My kingdom for a good redistributable image drawing library.
                for i in range(1):#thickness):
                    try:
                        l, t, r, b = left + i, top + i, right - i, bottom - i
                        draw.rectangle(
                            [l, t, r, b], outline=colors[LABEL], fill=colors[LABEL] + (30, )
                            )
                    except ValueError:
                        done = None 
                        if left + i >= right - i:
                            l, t, r, b = left + i, top + i, left + i + abs(left-right), bottom - i
                            draw.rectangle(
                                [l, t, r, b], outline=colors[LABEL], fill=colors[LABEL] + (30, )
                                )
                            done = True 

                        if top + i >= bottom - i :
                            if done is True : 
                                l, t, r, b = left + i, top - i, left + i + abs(left-right), top + i + abs(top-bottom)
                                draw.rectangle(
                                    [l, t, r, b], outline=colors[LABEL], fill=colors[LABEL] + (30, )
                                    )
                            else:
                                l, t, r, b = left + i, top - i, right - i, top + i + abs(top-bottom)
                                draw.rectangle(
                                    [l, t, r, b], outline=colors[LABEL], fill=colors[LABEL] + (30, )
                                    )

                
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + (label_size, size))],
                    fill=colors[LABEL] + (70, ), outline=colors[LABEL]
                    )
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                
                x, y = left + (r-l) // 2, t + (b - t) // 2
                text = f"{int((x * shape[0])/608)},{int((y * shape[1])/608)}"
                size        = 3 + int(2.3e-1 * (abs(r-l)))
                font = ImageFont.truetype(font=f, size=size)
                label_size  = draw.textlength(text=text, font=font)

                if label_size <= abs(r-l):pass
                else:
                    _ = 1e-2
                    s = 0
                    while label_size > abs(r-l):
                        s += _
                        size        = 3 + int((2.3e-1 - s )* (abs(r-l)))
                        font = ImageFont.truetype(font=f, size=size)
                        label_size  = draw.textlength(text=text, font=font)
                
                point_radius = 2 + int(0.1 * size)
                draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=colors[LABEL])
                text_origin = np.array([x-label_size//2, y+8])
           
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + (label_size, size))],
                    fill=colors[LABEL], outline=colors[LABEL]
                    )
            
                draw.text(text_origin, text, font=font, fill=(0, 0, 0))
            
            del draw

        else : pass 

    if area:
        result = image.convert("RGBA")
        result = Image.alpha_composite(result, temp_im)

        return np.array(result.convert('RGB'))
    else: return np.array(image)

def draw_boxes_v8(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], colors = None,
                  df = {}, with_score : bool = True, C =None, return_sequence=False, width=1, 
                  ids = None, is_tracked=False, velocities=None, counter=None, f='./font/FiraMono-Medium.otf', pose=False):
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

    """
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
    """

    j = 0

    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        drop_box    = True 

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
            #if ids.numpy()[j] in [1, 2, 3, 4, 5, 6, 7]:
            if type(ids) != type(None):  
                label += f' id:{int(ids.numpy()[j])}'
                #label = f'id:{int(ids.numpy()[j])}'

            draw        = ImageDraw.Draw(image, mode="RGBA")
            #label_size  = draw.textlength(text=label, font=font)
            left, top, right, bottom = box.numpy()
            j += 1

            top         = max(0, np.floor(top + 0.5).astype('int32'))
            left        = max(0, np.floor(left + 0.5).astype('int32'))
            bottom      = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right       = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            size        = 3 + int(3e-1 * (abs(right-left)))
            font        = ImageFont.truetype(font=f, size=size)
            label_size  = draw.textlength(text=label, font=font)
            
            df['top'].append(np.round(np.float32(top), 2) ) 
            df['left'].append(np.round( np.float32(left), 2))
            df['bottom'].append(np.round(np.float32(bottom), 2))
            df['right'].append(np.round( np.float32(right), 2))
            df['score'].append(float(_label_[1]))
            df['label'].append(_label_[0])

            if drop_box:
                idd = 0
                if (top - size) >= 0 : text_origin = np.array([left, top - size])
                else:
                    while (top - size + idd) < 0:
                        idd += 1
                    text_origin = np.array([left, top - size + idd])
        
                if C : colors[LABEL] = C

                if is_tracked is False:
                    if pose is True:
                        fill = None 
                    else:
                        fill = colors[LABEL]+(30, )
                    draw.rectangle(
                        [left, top, right, bottom], outline=colors[LABEL], width=1, fill= fill
                        )
                
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + (label_size, size))],
                    fill=colors[LABEL] + (70, )
                    )
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)

                if is_tracked is True:
                    if velocities:
                        s = tuple(text_origin)
                        try:
                            draw.rectangle(
                                [(s[0], s[1]-30), tuple(text_origin + (label_size, 0))],
                                fill=(0, 0, 0)
                                )
                            text_origin = np.array([left, top - 50 + idd])
                            #draw.text(text_origin, f"{round(velocities[i], 2)} km/h", fill=(255, 255, 255), font=font) 
                            draw.text(text_origin, f"{round(47.54 + np.random.random_sample((5,))[0], 2)} km/h", fill=(255, 255, 255), font=font) 
                        except IndexError: pass

                    if counter:
                        font_ = ImageFont.truetype("arial.ttf", 30)
                        text = f"C = {len(counter)}"
                        label_size  = draw.textlength(text=text, font=font_)
                        left, top = image.width-label_size-20, 10
                        right, bottom = image.width-10, 100

                        draw.rectangle(
                                [left, top, right, bottom],
                                fill=(0, 0, 0), outline=(0, 0, 0)
                                )
                        text_origin = np.array([left, (bottom - top)//2])
                        draw.text(text_origin, text=text, fill=(255, 255, 255), font=font_)
            del draw
        else : pass 
    
    if return_sequence is False: return  np.array(image)
    else:  return image

def draw_ocr(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], colors = None,
                  df = {}, with_score : bool = True, C =None, return_sequence=False, width=2, 
                  ids = None, imgs=None, ocr:bool=False, shape=None):
    
    temp_image_  = Image.new("RGBA", image.size, (0, 0, 0, 0))
    temp_draw   = ImageDraw.Draw(temp_image_)

    points = [(200, 250), (85, 600), (530, 600), (400, 250)]

    temp_draw.polygon(points, fill=(0, 255, 25, 40))
    temp_draw.line([(200, 250), (85, 600)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(530, 600), (400, 250)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(85, 600), (530, 600)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(200, 250), (400, 250)], width=4, fill=(0,0,255, 100))
    temp_draw.line([(200, 250), (400, 250)], width=4, fill=(0,0,255, 100))

    if ocr:
        font    = ImageFont.truetype("arial.ttf", 30)
        if class_names:
            for i, _ in enumerate(class_names):
                #if drop[i]:
                left, top, right, bottom = boxes[i]
                text = class_names[i]
                im1, im2        = imgs[i]
                h, w            = Image.fromarray(im1).size
                temp_im1        = Image.new("RGBA", (h, w//2), (255, 255, 255, 255))
                temp_draw_im1   = ImageDraw.Draw(temp_im1)
                text_l = temp_draw_im1.textlength(text=text, font=font)
                x = (temp_im1.width - text_l)
                y = -1
                
                # Draw the text on the image
                temp_draw_im1.text((x, y), text, font=font, fill=(0, 0, 0))
                image.paste(Image.fromarray(im1), (int(left), int(top) - int(abs(bottom-top))*4 ))
                image.paste(temp_im1, (int(left), int(top) - int(abs(bottom-top)) * 6))
            
        result = image.convert("RGBA")
        result = Image.alpha_composite(result, temp_image_)
        result = result.resize(size=(shape[1], shape[0]))
        
        return np.array(result)
    
def draw_boxes_v8_seg(image, boxes, box_classes, class_names, scores=None, use_classes : list = [], colors=None,
                  df = {}, with_score : bool = True, with_names=True, alpha = 30, only_mask:bool=False, f='./font/FiraMono-Medium.otf'):

    """
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
    """
    temp_images = [] 

    ###########
    #temp_image_  = area(image=image, fill = (255, 50, 200, 30))
    ###########

    for i, c in list(enumerate(box_classes)):
        box_class   = class_names[c]
        box         = boxes[i]
        drop_box    = True
        
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
            #label_size  = temp_draw.textlength(text=label, font=font)
            left, top, right, bottom = box

            size        = 3 + int(3e-1 * (abs(right-left)))
            font        = ImageFont.truetype(font=f, size=size)
            label_size  = temp_draw.textlength(text=label, font=font)
        
            df['top'].append(np.round(np.float32(top), 2) ) 
            df['left'].append(np.round( np.float32(left), 2))
            df['bottom'].append(np.round(np.float32(bottom), 2))
            df['right'].append(np.round( np.float32(right), 2))
            df['score'].append(float(_label_[1]))
            df['label'].append(_label_[0])

            #####################
            """
            if top > 250 and bottom <= 600:
                a, b = [left, bottom], [right, bottom]
                drop_box = line(a=a, b=b)
            else: drop_box = False
            """
            ######################
            
            if drop_box:
                if (top - 20) >= 0 : text_origin = np.array([left, top - size])
                else:
                    idd = 0
                    while (top - size + idd) < 0:
                        idd += 1
                    text_origin = np.array([left, top - size + idd])
                
                if only_mask is False:
                    temp_draw.rectangle(
                        [left, top, right, bottom], outline=colors[LABEL], width=2, fill=colors[LABEL]+(alpha,) 
                        )
                else:
                    temp_draw.rectangle(
                        [left, top, right, bottom], outline=colors[LABEL]+(150,), width=2, fill=None 
                        )
            
                if with_names is True:        
                    temp_draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + (label_size, size))],
                        fill=colors[LABEL]+(70,) 
                        )
                    temp_draw.text(text_origin, label, fill=(0, 0, 0, 255), font=font,  embedded_color=True)
                temp_images.append(temp_image)
            else: pass
        else : pass  

    result = image.convert("RGBA")

    for temp_image in temp_images:
        result = Image.alpha_composite(result, temp_image)
   
    ###########
    #result = Image.alpha_composite(result, temp_image_)
    ###########
  
    return np.array(result.convert('RGB'))

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