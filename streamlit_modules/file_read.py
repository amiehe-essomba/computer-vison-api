#import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tempfile
import shutil
from skimage.transform import resize
from yolo.utils.tools import preprocess_image, read_video


def img_resize(image, factor: float = None):
    if factor : image   = image.reduce(int(factor))
    else: pass

    return np.array(image).astype(np.float32) / 255.

def file_read(st, uploaded_file : any, show: bool = True, factor : float = None):
    import numpy as np 

    types       = ["jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi", "mkv"]
    files       = {'image' : [], "video" : [], 'video_reader' : [], 'image_shape' : [], "details" : []}
    process     = True
    list_types  = []
    video_id    = 0
    image_id    = 0
    _files      = []

    for file in uploaded_file:
        file_extension = file.name.split(".")[-1].lower()
        if file_extension in types:

            if file_extension in ["jpg", "jpeg", "png", "gif", "webp"]:
                try:
                    
                    image, image_data, shape = preprocess_image(img_path=file, model_image_size = (608, 608), factor=factor)
                    
                    files['image'].append((image, image_data))
                    _files.append('image')
                    list_types.append(f'image {image_id}')
                    files['image_shape'].append(shape)
                    image_id += 1
                except (FileNotFoundError, FileExistsError) : 
                    st.white("File loading error")
                    process = False
                    break
            elif file_extension in ["mp4", "mov", "avi", "mkv"]:
                import tempfile
                
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_file.write(file.read())
                video_path = temp_video_file.name

                list_types.append(f'video {video_id}')
                _files.append('video')
                video_id += 1
                files['video'].append(video_path)
                files['video_reader'].append(video_path)
                files['details'].append(None)
            else:
                process = False 
                st.write("Fichier non pris en charge. Veuillez charger une image ou une vidéo.")
                break
        else: 
            process = False
            st.white("The input is not take into account here")
            break
    
    if process:
        video_id    = 0
        image_id    = 0

        tabs = st.tabs(list_types)
        
        if show:
            if list_types : 
                for i, types in enumerate(list_types):
                    if _files[i] == 'image':
                        with tabs[i]:
                            shape = files['image_shape'][image_id]
                            st.header(f"{list_types[i]}, shape = {shape}")
                            img = files['image'][image_id][0].copy()
                            img_array = resize(np.array(img), output_shape=shape[:-1])
                            st.image(img_array, use_column_width=True)
                        image_id += 1
                    else:
                        with tabs[i]:
                            st.header(list_types[i])
                            #shutil.copy(files['video'][video_id], "temp.mp4")
                            st.video(files['video'][video_id], format="video/mp4")
                            video_reader, *details = read_video(files['video'][video_id])
                            files['video_reader'][video_id] = video_reader
                            files['details'][video_id] = details[0]
                            #shutil.rmtree(files['video'][video_id], ignore_errors=True)
                           
                        video_id += 1
        else :
            if list_types:
                for i, types in enumerate(list_types): 
                    if _files[i] != 'image':
                        video_reader,  *details  = read_video(files['video'][video_id])
                        files['video_reader'][video_id] = video_reader
                        files['details'][video_id] = details[0]
                        video_id += 1
                        
    else: pass 

    return files

def online_link(st, url : str = "", show_image : bool = True):
    image, image_data, shape, error = None, None, None, None
    # Vérifie si le champ de saisie n'est pas vide
    if url:
        # Affiche le lien hypertexte
        st.markdown(f"You enter this link : [{url}]({url})")
        image, image_data, shape, error = url_img_read(url=url)

        if error is None:
            if show_image is True:
                st.header(f"image 0, shape = {shape}")
                img_array = resize(np.array(image), output_shape=shape[:-1])
                st.image(img_array, use_column_width=True)
                st.markdown(f"file successfully upladed...")
            else: pass
        else: st.markdown(f"{error}")
    else: pass 

    return (image, image_data, shape, error)

def url_img_read( url : str, factor = False):
    from PIL import Image
    import requests
    from io import BytesIO
    import numpy as np 

    image, image_data, error, shape = None, None, None, None
    # Replace 'url' with the URL of the image you want to read

    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Read the image from the response content
            image_data = BytesIO(response.content)
            #Image.open(image_data)
            image = Image.open(image_data)

            image, image_data, shape = preprocess_image(img_path=image, model_image_size = (608, 608), done=True, factor=factor)
        else:
            error = f"Failed to retrieve image. Status code: {response.status_code}"
    except Exception as e:
        error = f"An error occurred: {str(e)}"

    return image, image_data, shape,  error

def camera():
    import streamlit as st
    import cv2
    import numpy as np

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        image = Image.open(img_file_buffer)

        # Check the shape of cv2_img:
        # Should output shape: (height, width, channels)
        st.write(image.size)

        image, image_data, shape = preprocess_image(img_path=image, model_image_size = (608, 608), done=True, factor=False)

        
        return image, image_data, shape
    else: return None, None, None