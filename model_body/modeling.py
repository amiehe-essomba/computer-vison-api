def modeling(st):
    import streamlit as st
    import numpy as np
    from yolo.iou import IoU
    from PIL import Image, ImageDraw
    import pandas as pd 

    st.write('<style>{}</style>'.format(styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">Data modelling and understanding</h1>', unsafe_allow_html=True)
    
    st.write(f'<h1 class="body-text">Inputs and outputs</h1>', unsafe_allow_html=True)
    p1 = """
        The input is a batch of images, and each image has the shape (608, 608, 3)
        The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented
         by 6 numbers (pc, bx, by, bh, bw, c). If you expand c
        into an 80-dimensional vector, each bounding box is then represented by 85 numbers [[pc, bx, by, bh, bw, c], [80 class probabilities]].
        """
    transform(st, p1)

    st.write(f'<h1 class="body-text">Anchor Boxes</h1>', unsafe_allow_html=True)
    p2 = """
        Anchor boxes are chosen by exploring the training data to choose reasonable height/width ratios that 
        represent the different classes. For this assignment, 5 anchor boxes have been chosen (to cover the 80 classes)
        The dimension of the encoding tensor of the second to last dimension based on the anchor boxes is  
        (𝑚, nH, nW, anchors, classes).
        The YOLO architecture is: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).
        """
    transform(st, p2)

    st.write(f'<h1 class="body-text">Class Score</h1>', unsafe_allow_html=True)

    t=r'''
    \begin{align}
        scores = pc \times 
        \begin{pmatrix}
            1. \\
            1. \\
            1. \\
            \vdots \\
            1. \\
            1. \\
            1.
        \end{pmatrix} \times
        \begin{pmatrix}
            c2 \\
            c1 \\
            c3 \\
            \vdots \\
            c78 \\
            c79 \\
            c80
        \end{pmatrix} = 
        \begin{pmatrix}
            pc \times c2 \\
            pc \times c1 \\
            pc \times c3 \\
            \vdots \\
            pc \times c78 \\
            pc \times c79 \\
            pc \times c80
        \end{pmatrix} 
    \end{align}
    '''
    st.latex(t)

    st.latex(
    r'''
    \begin{equation}
        class = argmax(scores)
    \end{equation}
    '''
    )

    p2 = """
        These two equations demonstrate how the score for each object is calculated and how the 
        object's class is determined. However, building a computer vision model is not limited 
        to simply stacking convolutional layers one after another. A much more complex process 
        lies in the ability to accurately draw bounding boxes, indicating the precise positions 
        of objects in an image or video.
        It's worth noting that modeling a system with the YOLO model yields information on 
        thousands of bounding boxes, each with its respective scores and class assignments. 
        Therefore, it is necessary to filter these results to obtain a more consistent and meaningful output. 
        To achieve this, three Python modules have been implemented: YOLO Filter Boxes, 
        Intersection over Union (IoU), and YOLO Non-Maximum Suppression (NMS). 
        Let's delve into how these three modules are indispensable for making a computer vision project more optimal.
        """
    transform(st, p2)

    st.write(f'<h1 class="body-text">YOLO Filter Boxes</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    box_anchor1,    box_anchor2 = links('box_anchor')
    with col1:
        st.markdown(f'<a href="" target="_blank"><img src="{box_anchor2}" width="300" height="200"></a>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<a href="" target="_blank"><img src="{box_anchor1}" width="200" height="200"></a>', unsafe_allow_html=True)

    p2 = """
        The YOLO Filter Boxes process involves eliminating boxes with a score below a specified threshold. 
        This is a critical step in object detection, as it serves to reduce the number of false positives, 
        ensuring that only high-quality detections are retained.
        In simpler terms, let's consider an example where there are three bounding boxes in an image, 
        each associated with a confidence score. For instance, a red box with a score of 0.7, a green 
        box with a score of 0.6, and a blue box with a score of 0.8. 
        """
    transform(st, p2)
    p2 ="""
        If you set a threshold of 0.7, only 
        the blue box will be retained, as its confidence score exceeds the specified threshold. The choice 
        of your threshold plays a pivotal role because it allows you to establish the criteria for selecting 
        boxes based on their confidence scores.
        This selection process is essential for filtering out less certain or less accurate detections, 
        ensuring that your computer vision system focuses on the most reliable results and enhances the 
        overall performance and reliability of object detection.
        """
    transform(st, p2)

    #if st.toggle('See how it works'):
    #    filter_box_test(st=st)
    
    with st.expander("HAVE A LOOK HERE."):
        filter_box_test(st=st)

    st.write(f'<h1 class="body-text">YOLO IoU(Intersection over Union)</h1>', unsafe_allow_html=True)
    iou = links('iou')
    t = r'''
    \begin{equation}
     IoU = \dfrac{A \cap_{} B}{A \cup_{} B - A \cap_{} B} 
     \end{equation}
    '''
    st.markdown(f'<a href="" target="_blank"><img src="{iou}" width="400" height="200"></a>', unsafe_allow_html=True)
    st.latex(t)

    p2 = """  
        IoU (Intersection over Union) is a fundamental metric used to evaluate the precision of object 
        detections in computer vision. It is a measure of how well a predicted bounding box aligns with 
        the ground truth, which is a manually annotated bounding box provided by a human annotator or through some other means.
        The IoU is calculated by taking the ratio of the area of intersection between the predicted 
        bounding box and the ground truth box and dividing it by the area of their union. 
        Mathematically, this is expressed as:
        """
    transform(st, p2)
    t = r'''
    \begin{equation}
     IoU = \dfrac{Intersection}{Union} 
     \end{equation}
    '''
    st.latex(t)
    transform(st, p2)
    p2 ="""
        This metric provides a quantifiable assessment of how much overlap exists between the predicted 
        and ground truth bounding boxes. A higher IoU indicates a better alignment and a more accurate 
        detection, while a lower IoU implies less agreement and a less accurate detection.
        In practice, a threshold for IoU is often defined, such as 0.5. When the IoU between a predicted 
        box and a ground truth box exceeds this threshold, the detection is considered valid. If the IoU 
        falls below the threshold, the detection is considered inaccurate, and the box is typically discarded.
        Setting an appropriate IoU threshold is crucial, as it influences the balance between sensitivity and 
        specificity in object detection. Higher thresholds result in stricter criteria for valid detections, 
        reducing false positives but potentially missing some true positives. Lower thresholds are more permissive, 
        increasing sensitivity but also potentially introducing more false positives. The choice of IoU threshold 
        should be tailored to the specific requirements and goals of the computer vision task at hand.
        """
    transform(st, p2)
    
    with st.expander("HAVE A LOOK HERE."):
        transform(st, iou=True, text=iou_schema())
        iou_test(st=st)

    st.write(f'<h1 class="body-text">YOLO Non-Max Suppression</h1>', unsafe_allow_html=True)
    non_max = links('non-max')
    st.markdown(f'<a href="" target="_blank"><img src="{non_max}" width="400" height="300"></a>', unsafe_allow_html=True)
    p2 = """
        When multiple bounding boxes exist for the same object, Non-Maximum Suppression (NMS) comes into play to remove redundant detections. 
        It operates by retaining the bounding box with the highest confidence score while eliminating others that significantly overlap with it.
        This process ensures that each object is represented by a single detection.
        """
    transform(st, p2)
    p2 ="""
        Once the bounding boxes have been filtered, and NMS is applied, you obtain a final list of valid object detections. 
        Each detection includes the predicted class (e.g., "car," "dog"), the coordinates of the bounding box, the confidence 
        score (indicating the model's confidence in the detection), and other relevant information.
        NMS is a vital post-processing step in object detection pipelines as it refines the output by reducing 
        redundancy and ensuring that only the most confident and non-overlapping detections are considered valid. 
        This helps in producing a more accurate and concise set of object detections, which is particularly 
        important in real-world applications of computer vision.
        """
    transform(st, p2)

    st.write(f'<h1 class="body-text">Unlock Python modules by solving this issue </h1>', unsafe_allow_html=True)

    with st.expander("IoU Problems"):
        reset = True
        if reset:
            np.random.seed(None)
            box1 = iuo_solution()
            box2 = iuo_solution()

        st.text(f'box1 : {box1}')
        st.text(f"box2 : {box2}")
        col1, col2 = st.columns(2)
        num = st.number_input('answer')

        with col1:
            reset = st.button('reset', key=10)
        with col2:
            run = st.button('run')
        iou = IoU(box_xy1=box1, box_xy2=box2, return_box=None)

        if run:
            if round(num, 4) == round(iou, 4):
                st.text(f'Congratulation, iou = {round(iou, 4)}')
            else:
                st.text(f'wrong, expected iou : {round(iou, 4)}')
                st.write("Try again.")

    st.write(f'<h1 class="body-text">Models</h1>', unsafe_allow_html=True)
    
    index = ['Object Detection', 'Optical Character Recognition(OCR)', 'Tracking', 'Object Segmentation', 'Classification']

    data = {
        "my model" : [True, False, True, False, False], 
        'yolov8' : [True, False, True, False, False], 
        'ocr+yolov8' : [True, True, True, False, False], 
        'yolov5' : [True, False, False, False, False], 
        'yolov8-seg'  :[True, False, False, True, False], 
        'yolov8-pose' : [True, False, False, False, False],
        'yolov8-cls' : [False, False, False, False, True]
        }
    
    df = pd.DataFrame(data=data, index=index)
    st.dataframe(data=df)

def iuo_solution():
    import numpy as np 
    ix1, iy1 = np.random.randint((100, 100))
    ix2, iy2 = np.random.randint((100, 100))
    
    if ix1 > ix2 :  x1, x2 = ix2, ix1 
    else: x1, x2 = ix1, ix2

    if iy1 > iy2 : y1, y2 = iy2, iy1
    else: y1, y2 = iy1, iy2

    return (x1, y1, x2, y2)

def transform(st, text, iou: bool = False):
    s = text
    if iou: st.markdown(f'<p class="iou_schema">{s}</p>', unsafe_allow_html=True)
    else : st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

def links(name : str):
    if name == 'box_anchor':
        s1 = "https://datadrivenscience.com/wp-content/uploads/elementor/thumbs/anchor-boxes-q3h8ftp0qfhcizr6y9n9eln2x0caocm4e5cfysklxc.png"
        s2 = "https://blog.roboflow.com/content/images/2020/08/anchor_boxes.jpg"
        return (s1, s2)
    if name == 'iou':
        s = "https://assets-global.website-files.com/5be2a3cf7a0067207a10fd2d/5f5a78ea7e6bb843baeb0fff_iou_formula.png"
        return s
    if name == 'non-max':
        s = "https://th.bing.com/th/id/R.fa99141b5d1ade2fb1ef2fc0197aea1e?rik=tZYSy7EaouU7OQ&riu=http%3a%2f%2fmedia5.datahacker.rs%2f2018%2f11%2fnon_max_b_boxes.png&ehk=odgL7S4nWOigVD4lnpd9ViXNOMMU0XeYkP60%2f9tRmSA%3d&risl=&pid=ImgRaw&r=0"
        return s
    else: pass 

    """
    IoU(Intersection over Union) est une métrique essentielle pour évaluer la précision des détections. 
    Elle est calculée en comparant le chevauchement entre une boîte prédite et une boîte 
    de vérité terrain (une boîte annotée par un humain). L'IoU est calculée comme : Intersection / Union. 
    Un seuil IoU est généralement défini (par exemple, 0,5) pour déterminer si une détection est valide.
    """

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

        .body-text {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 22px; /* Taille de police */
            font-weight: bolder; /* Gras */
            margin: 5px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
        }

        .header-text-under {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 15px; /* Taille de police */
            text-decoration: None; /* Souligné underline overline */
            margin: 10px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
        }

        .iou_schema {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 15px; /* Taille de police */
            text-decoration: None; /* Souligné underline overline */
            margin: 10px; /* Marge extérieure */
            border-radius: 5px; /* Coins arrondis */
            font-family: Arial, sans-serif; /* font family*/
        }

    """
     
    return custom_css_title

def boxes(st, name : str, box: tuple = (0.0, 0.5, 5.0, 5.0)):
    import streamlit as st

    X1, Y1, X2, Y2 = st.columns(4)
    with X1:
        x1 = st.slider(f"{name} x1", min_value=0.0, max_value=100.0, value=box[0])
    with Y1:
        y1 = st.slider(f"{name} y1", min_value=0.0, max_value=100.0, value=box[1])
    with X2:
        x2 = st.slider(f"{name} x2", min_value=0.0, max_value=100.0, value=box[2])
    with Y2:
        y2 = st.slider(f"{name} y2", min_value=0.0, max_value=100.0, value=box[3])

    return (x1, x2, y1, y2)

def check(*args):
    error = None 
    names = ['x1', 'x2', 'y1', 'y2']
    if args[0] < args[1]:
        if args[2] < args[3]:
            pass 
        else: error = f"{names[3]} is lower than {names[2]}"
    else: error = f"{names[1]} is lower than {names[0]}"

    return error 

def iou_test(st):
    import streamlit as st 
    import pandas as pd 
    from yolo.iou import IoU
    from PIL import Image, ImageDraw
    import numpy as np 

    (x1, x2, y1, y2) = boxes(st=st, name='box1')

    error = check(*(x1, x2, y1, y2))
    if error :  st.warning(error ,icon="⚠️")
    else:
        (x1_, x2_, y1_, y2_) = boxes(st=st, name='box2',box=(2.5, 3.5, 7.5, 8.9))
        error = check(*(x1_, x2_, y1_, y2_))
        if error :  st.warning(error ,icon="⚠️")
        else: 
            data    = {'x1' : [x1, x1_], 'x2' : [x2, x2_], 'y1' : [y1, y1_], 'y2' : [y2, y2_]}
            box1    = [x1, y1, x2, y2]
            box2    = [x1_, y1_, x2_, y2_]

            f_, w_      = st.columns(2)
            with f_:
                factor  = st.slider("scale factor", max_value=100, min_value=1, step=2, value=10)
            with w_:
                width   = st.slider('line width', max_value=8, min_value=1, step=1, value=2)

            fill_box    = st.checkbox('fill', value=False)
            (w, h), box1, box2      = w_h(box1=box1, box2=box2, factor=factor)
            image                   = draw_boxes(box1=box1, box2=box2, w=w, h=h, width=width, bg=(0, 0, 0), fill_box=fill_box)
            
            try:
                iou, box_iou  = IoU(box_xy1=box1, box_xy2=box2, return_box=True)
                
                if iou == 0.0 : st.image(image)
                else:
                    if box_iou[0] > box_iou[2] : 
                        x1, x2 = box_iou[0], box_iou[2]
                        box_iou[0] = x2 
                        box_iou[2] = x1 
                    if box_iou[1] > box_iou[3]:
                        y1, y2 = box_iou[1], box_iou[3]
                        box_iou[1] = y2 
                        box_iou[3] = y1
                    img = Image.fromarray(image)
                    draw = ImageDraw.Draw(img)
                    if fill_box:  fill = (0, 0, 255)
                    else : fill = (255, 255, 255)
                    draw.rectangle(box_iou, outline=(0, 0, 255), width=width, fill=fill)
                    image  =  np.array(img)
                   
                    if st.checkbox('Show image') : st.image(image)
                
                run = st.button('run IoU')
                if run : st.write('IoU :', round(iou, 5))
            except ZeroDivisionError:
                st.warning("Division by zero", icon="⚠️") 
          
def draw_boxes(box1, box2, w: float, h : float, color = [(255, 0, 0), (0, 255, 0)], 
               width = 2, bg: str = "", fill_box : bool = False):
    from PIL import Image, ImageDraw
    import numpy as np 

    image = Image.new('RGB', (w, h), color=bg)  

    draw = ImageDraw.Draw(image)
    if fill_box:  fill = color
    else:  fill = (None, None)
    draw.rectangle(box1, outline=color[0], width=width, fill=fill[0])
    draw.rectangle(box2, outline=color[1], width=width, fill=fill[1])

    image_array = np.array(image)

    return image_array

def w_h(box1, box2, factor):
    if box1[0] <= box2[0]:
        if box1[2] <= box2[2]:
            w = int(abs(box1[0] - box2[2])) + 1
        else:
            w = int(abs(box1[0] - box1[2])) + 1
    else:
        if box1[2] <= box2[2]:
            w = int(abs(box2[0] - box2[2])) + 1
        else:
            w = int(abs(box1[2] - box2[0])) + 1


    if box1[1] <= box2[1]:
        if box1[3] <= box2[3]:
            h = int(abs(box1[1] - box2[3])) + 1
        else:
            h = int(abs(box1[1] - box1[3])) + 1
    else:
        if box1[3] <= box2[3]:
            h = int(abs(box2[1] - box2[3])) + 1
        else:
            h = int(abs(box1[3] - box2[1])) + 1
    
    box1 = [x * factor for x in box1]
    box2 = [x * factor for x in box2]

    return (w * factor, h * factor), box1, box2

def iou_schema():
    s="""
    IoU Schema\n\n
    (box x1, box1 y1)(min)
    +-------------------------------------+
    |                 box 1               |
    |                                     |
    |      (box2 x1, (box2 y1)(min)       |
    |        +----------------------------+-------+
    |        |/////////  IoU AREA  ////// |       | 
    +--------+----------------------------+-------+ (box1 x2, box1 y2) (max) 
             |                                    |
             |        box 2                       |
             |                                    |
             +------------------------------------+ (box2 x2, box2 y2) (max)
    """

    return s 

def filter_box_test(st):
    import streamlit as st 
    import numpy as np 
    from PIL import Image
    from yolo.filter_boxes import yolo_filter_boxes
    from model_body.filter_box_draw import draw_boxes as db
    import tensorflow as tf 

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        img_size  = st.slider('image size', max_value=10, min_value=5, step=1, value=5)
    with col2:
        n_box = st.slider('number of boxes', min_value=1, max_value=10, step=1, value=1)
    with col3:
        n_classes = st.slider('classes', min_value=1, max_value=100, step=5, value=5)
    with col4:
        threshold = st.slider('threshold', min_value=0.1, max_value=1.0, step=0.05, value=0.5)
    
    f_, w_, c_ = st.columns(3)
    with f_:
        factor    = st.slider('scale factor', min_value=1, max_value=100, step=5, value=10) 
    with w_:
        width     = st.slider('line width', min_value=1, max_value=10, step=1, value=2)
    with c_:
        seed      = st.slider('random state', min_value=1, max_value=100, step=1, value=3)

    (w, h), scores, boxes, classes, image, _ = genarate_data(img_size, n_box, n_classes, threshold, factor, seed)

    image = db(boxes=boxes, w=w, h=h, width=width)
    if st.checkbox("show image") : st.image(image)

def genarate_data(img_size, n_box, n_classes, threshold, factor, seed):
    import numpy as np 
    import streamlit as st 
    from PIL import Image
    from yolo.filter_boxes import yolo_filter_boxes
    from model_body.filter_box_draw import draw_boxes as db
    import tensorflow as tf

    np.random.seed(seed=seed)
    boxes           = np.random.randn(img_size, img_size, n_box, 4)
    box_confidence  = np.random.randn(img_size, img_size, n_box, 1)
    box_class_prob  = np.random.randn(img_size, img_size, n_box, n_classes)

    image                   = Image.new('RGB', (img_size, img_size), color=(255, 255, 255)) 
    scores, boxes_, classes = yolo_filter_boxes(boxes=boxes, box_confidence=box_confidence, 
                                    box_class_prob=box_class_prob, threshold=threshold)
    boxes                   = boxes_.numpy().copy()

    for i in range(boxes.shape[0]):
        x1, y1 = boxes[i][0], boxes[i][1]
        bw, bh = boxes[i][2], boxes[i][3]

        if x1 < 0 : x1 += img_size
        if y1 < 0 : y1 += img_size
        if bh < 0 : bh += img_size
        if bw < 0 : bw += img_size

        boxes[i][2], boxes[i][3] = bw, bh 
        boxes[i][0], boxes[i][1] = x1, y1

        if x1 > bw :
            boxes[i][2] = boxes[i][0] + 1.0 + (x1-bw)

        if y1 > bh:
            boxes[i][3] = boxes[i][1] + 1.0 + (y1-bh)

    box1 = [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 0].max(), boxes[:, 1].max()]
    box2 = [boxes[:, 2].min(), boxes[:, 3].min(), boxes[:, 2].max(), boxes[:, 3].max()]
    boxes = boxes * factor

    (w, h), box1, box2 = w_h(box1, box2, factor)

    return (w, h), scores, boxes, classes, image, boxes_

