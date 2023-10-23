import string
import easyocr
import numpy as np
from PIL import ImageDraw, ImageFont
from skimage.transform import resize
from keras import backend as K
from functools import reduce



def read_license_plate(license_plate_crop):
    # Initialize the OCR reader
    reader = easyocr.Reader(['en'], gpu=False)
    detections = reader.readtext(license_plate_crop)
    
    for detection in detections:
        bbox, text, score   = detection
        text                = text.upper()

        return text, score
        
    return None, None