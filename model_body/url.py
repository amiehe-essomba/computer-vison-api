def url_img_read( url : str, show: bool = False):
    from PIL import Image
    import requests
    from io import BytesIO
    import matplotlib.pyplot as plt

    image = None
    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Read the image from the response content
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            if show:
                f, x = plt.subplots(figsize=(8, 8))
                x.axis('off')
                x.imshow(image)
                plt.show()

        else:  print(f"Failed to retrieve image. Status code: {response.status_code}")
    except Exception as e:  print(f"An error occurred: {str(e)}")

    return image