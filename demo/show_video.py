def video(video_path :str = './video/yolo_video2.mp4'):
    from IPython.display import HTML
    from base64 import b64encode

    video_file = open(video_path, "rb").read()

    # Encodez la vidéo en base64
    video_encoded = b64encode(video_file).decode('utf-8')

    # Affichez la vidéo dans le notebook
    video_tag = f'<video controls alt="test" src="data:video/mp4;base64,{video_encoded}" type="video/mp4">'
   
    return  HTML(video_tag)

