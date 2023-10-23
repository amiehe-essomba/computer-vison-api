def project(st):
    st.write('<style>{}</style>'.format(styles()), unsafe_allow_html=True)
    st.write(f'<h1 class="header-text">{title()}</h1>', unsafe_allow_html=True)
    
    objectives(st=st)
    data_colllection(st=st)
    train(st=st)

    ressources(st=st)

def title():
    t = """
        Project Title: Enhancing Road Safety through Computer Vision
        """
    return t 

def objectives(st):
    s = """
    The main goal of this project is to utilize Optical Character Recognition (OCR) 
    technology alongside smart cameras and the YOLO (You Only Look Once) model to enhance
    traffic signal management and ultimately decrease the occurrence of traffic accidents. 
    This project seeks to establish an intelligent traffic control system capable of identifying 
    vehicles and pedestrians, analyzing their actions, and dynamically adjusting traffic flow to proactively prevent collisions.
    """
    st.write(f'<h1 class="body-text">Objective:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)
   

def data_colllection(st):
    s = """
    I began by collecting a diverse and representative dataset, 
    including videos of road traffic scenes, images of traffic lights, 
    and annotations for vehicles, pedestrians, traffic lights, etc..
    """
    st.write(f'<h1 class="body-text">Data Collection:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

def process(st):
    s = """
        Videos were divided into individual images, 
        and annotations were prepared for training the YOLO model.
        """ 
    st.write(f'<h1 class="body-text">Data Preprocessing:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

def train(st):
    s = """
        I used the YOLO model to train a convolutional neural network to detect objects and 
        key elements of traffic, such as vehicles, pedestrians, 
        traffic lights etc. Training involved multiple iterations to optimize detection accuracy.
        """ 
  
    st.write(f'<h1 class="body-text">Training the YOLO Model:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

    s = """
        The YOLO model integrated into the smart cameras analyzes real-time images and 
        videos captured to detect vehicles, pedestrians, and traffic lights and so on. It also 
        identifies potentially dangerous movements and behaviors.
        """
    st.write(f'<h1 class="body-text">Real-Time Detection:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)


    s = """Based on real-time data, the system makes decisions to regulate traffic to prevent collisions. 
        For example, it can extend a green light to allow a pedestrian to cross safely or 
        detect vehicles that do not stop at a red light.
        """
    
    st.write(f'<h1 class="body-text">Decision Making and Traffic Light Control:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

    s = """I established an ongoing evaluation process to monitor the system's 
            effectiveness and make improvements based on real-time data and feedback.
        """
    
    st.write(f'<h1 class="body-text">Evaluation and Optimization:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

    s = """
        The ultimate goal of this project is to reduce the number of traffic accidents, 
        improve traffic flow, and make roads safer for users. By optimizing real-time traffic 
        light management through computer vision, this project contributes to a safer 
        and more efficient driving environment.
        """
    st.write(f'<h1 class="body-text">Expected Results:</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="header-text-under">{s}</p>', unsafe_allow_html=True)

    st.write(f'<h1 class="body-text">Potential Impacts:</h1>', unsafe_allow_html=True)
    st.markdown(
        '''
        <ol>
        <li class="header-text-under">Road safety.</li>
        <li class="header-text-under">Reduction in the number of traffic accidents.</li>
        <li class="header-text-under">Decreased traffic congestion.</li>
        <li class="header-text-under">Improved safety for pedestrians and drivers.</li>
        <li class="header-text-under">Contribution to intelligent traffic management in cities.</li>
        </ol>
        '''
        , unsafe_allow_html=True)


def ressources(st):
    st.write(f'<h1 class="body-text">Ressources:</h1>', unsafe_allow_html=True)
    st.markdown(f'<a class="header-text-under" href="https://pjreddie.com/darknet/yolo/">The official YOLO website</a>', unsafe_allow_html=True)
    st.markdown(f'<a class="header-text-under" href="https://arxiv.org/abs/1612.08242">Joseph Redmon, Ali Farhadi</a>', unsafe_allow_html=True)
    st.markdown(f'<a class="header-text-under" href="https://arxiv.org/abs/1612.08242">YOLO9000: Better, Faster, Stronger</a>', unsafe_allow_html=True)
    st.markdown(f'<a class="header-text-under" href="https://learnopencv.com/ultralytics-yolov8/#How-to-Use-YOLOv8?">YOLOv8 Ultralytics: State-of-the-Art YOLO Models</a>', unsafe_allow_html=True)
    st.markdown(f'<a class="header-text-under" href="https://docs.ultralytics.com/datasets/">YOLOv8  Datasets</a>', unsafe_allow_html=True)

def styles():
     custom_css_title = """
        .header-text {
            color: black; /* Couleur du texte */
            background-color: white; /* Couleur de l'arrière-plan */
            font-size: 25px; /* Taille de police */
            font-weight: bolder; /* Gras */
            text-decoration: underline; /* Souligné underline overline */
            font-family: Arial, sans-serif; /* font family*/
            text-align: justify;
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

    """
     
     return custom_css_title






