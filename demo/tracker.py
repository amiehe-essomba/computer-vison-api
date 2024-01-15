from PIL import Image
import cv2

def ROI(frame):
    x, y, w, h = cv2.selectROI("Windows Shape", frame, fromCenter=False, showCrosshair=True)
    roi = frame[y : y + h, x : x + w]

    return roi

def roi_initialize(roi):
    roi_hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist    = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
    roi_hist    = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    return term_crit, roi_hist 

def run_roi(frame, roi_hist, term_crit):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_crit)

    x, y, w, h = track_window
    image = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

    return image


def Points(
        a : list[int, int], 
        b : list[int, int], 
        c : list[int, int], 
        d : list[int, int]
        ):
    import numpy as np
    # Définir les quatre points du quadrilatère
    points = np.array([a, b, c, d], np.int32)
    points = points.reshape((-1, 1, 2))

    return points

def delimiter_zone( 
        shape,
        points  : any, 
        fill    : tuple[int, int, int] = (0, 255, 0), 
        boder   : tuple[int, int, int] = (255, 255, 255),
        thickness : int = 2
        ):
    import cv2
    import numpy as np 

    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Tracer les lignes du quadrilatère (ne remplit pas encore)
    image = cv2.polylines(image, [points], isClosed=True, color=boder, thickness=thickness)

    # Remplir la région intérieure du quadrilatère
    image = cv2.fillPoly(image, [points], color=fill)

    return image




    
  
