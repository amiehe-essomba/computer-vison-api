from streamlit_modules.links import links
from streamlit_modules.sidebar_styles import sidebar_styles as ss
import streamlit as st
from streamlit_modules.info import info

def example(st, file : str = "./video/yolo_pred.mp4"):
    """
    import tempfile
    import moviepy.editor as mb

    file = open(file=file, mode='rb')
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(file.read())
    video = mb.VideoFileClip(temp_file.name)

    # Utilisez un répertoire temporaire pour stocker la vidéo encodée
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video.write_videofile(temp_video_file.name, codec='libx264')
    st.sidebar.video(temp_video_file.name, format="video/mp4")

    # Assurez-vous de supprimer les fichiers temporaires après utilisation
    if 'temp_file' in locals():
        temp_file.close()

    if 'temp_video_file' in locals():
        temp_video_file.close()

    """
    """
    import cv2
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        st.error("Impossible de lire la vidéo.")
    else:
        video_file = open(file, 'rb')
        video_bytes = video_file.read()
        st.sidebar.video(video_bytes, format="video/mp4")

        # Libérez les ressources
        cap.release()
        video_file.close()

    """

    video_file = open(file, 'rb')
    video_bytes = video_file.read()
    st.sidebar.video(video_bytes, format="video/mp4")
    video_file.close()

def sidebar(streamlit = st):
    yolo_feedback_contrain, contain_feedback = None, None 

    all_names = ['logo_git', 'logo_linkidin', 'git_page','linkinding_page', 'loyo_logo','my_picture', 'computer-vis']

    # get sideber style 
    custom_sidebar_style = ss()

    # initialize the style
    streamlit.write('<style>{}</style>'.format(custom_sidebar_style), unsafe_allow_html=True)
   
    # put the first image in the sidebar 
    cm          = links(name='computer-vis')
    # git hub link page 
    git_page    = links('git_page')
    # create image with associtated link 
    streamlit.sidebar.markdown(f'<a href="{git_page}" target="_blank"><img src="{cm}" width="250" height="200"></a>', unsafe_allow_html=True)
    
    
    # contains section : create the table of constains 
    streamlit.sidebar.write('<h3 class="sidebar-text">Table of contains</h3>', unsafe_allow_html=True)
    # list of contains 
    contains = (
        "Project description", 
        "Introduction",
        "Modelling",
        "Model performance",
        "prediction",
        "Conclusion"
        )
    
    
    # get feedback storage in contain_feedback
    index            = None 
    contain_feedback = streamlit.sidebar.radio('all contains', options=contains, disabled=False, index=index)
    
    if streamlit.sidebar.button('reset'):
        contain_feedback = None 
        index = None   
    # Scripts python section : 
    streamlit.sidebar.write('<h3 class="sidebar-text">Learn More About Scripts Project</h3>', unsafe_allow_html=True)
    # liste of python scripts to create yolo model
    yolo_contrains = ("IoU", "yolo filter boxes", "yolo-non-max suppression", 
                "yolo boxes to corners", "yolo evaluation", "yolo model")
    # get the feedback
    if contain_feedback:  disable = True
    else: disable = False
    
    yolo_feedback_contrain = streamlit.sidebar.selectbox('Computer Vision modules', options=yolo_contrains, disabled=disable, index=None)
    
    example(st=streamlit)

    streamlit.write('<style>{}</style>'.format(custom_sidebar_style), unsafe_allow_html=True)
    # create 10 line of space 
    for i in range(5):
        streamlit.sidebar.write('<h5 class="author"> </h5>', unsafe_allow_html=True)
    
    # section about author
    streamlit.sidebar.write('<h3 class="sidebar-text">About Author</h3>', unsafe_allow_html=True)
    # my name 
    streamlit.sidebar.write('<h5 class="author">Dr. Iréné Amiehe Essomba </h5>', unsafe_allow_html=True)
    # git and linkidin links apge 
    linkidin_page   = links('linkinding_page')
    # my picture took in my linkidin page 
    my_photo        = links('my_picture')
    col1_, col2_      = streamlit.sidebar.columns(2)
    with col1_:
        streamlit.sidebar.markdown(f'<a href="{linkidin_page}" target="_blank"><img src="{my_photo}" width="125" height="125"></a>', unsafe_allow_html=True)
    
    with col2_:
        # Bibliograpy section 
        streamlit.sidebar.markdown(f'<p class="author-info ">{info()}</p>', unsafe_allow_html=True)

    # github and likidin logo 
    for i in range(1):
        streamlit.sidebar.write('<h5 class="author"> </h5>', unsafe_allow_html=True)

    logo_git        = links('logo_git')
    logo_linkidin   = links('logo_linkidin')
    streamlit.sidebar.markdown(
        f'<div style="text-align: left;">'
        f'<a href="{linkidin_page}" target="_blank"><img src="{logo_linkidin}" width="20"></a>'
        f'<a href="{git_page}" target="_blank"><img src="{logo_git}" width="20"></a>'
        f'</div>', 
        unsafe_allow_html=True
        )
    
    return [contain_feedback, yolo_feedback_contrain]