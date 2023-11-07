import urllib
import cv2
import numpy as np 

def url_to_image(url, size=(224, 224)):
    resp = urllib.request.urlopen(url) 
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, -1)
    img = cv2.resize(img[:,:,[2,1,0]], size)
    return img

def url_img_read_and_save(url: str, path : str = "./data/data/" ):
    import requests

    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Read the image from the response content

            """
            image_data = BytesIO(response.content)
            image = Image.open(image_data) #Image.open(image_data)

            # You can now work with the 'image' object (e.g., display or process it)
            # For example, you can display the image:
            image = np.array(image).astype(np.float32) / 255 
            """ 

            with open(path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to retrieve image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
