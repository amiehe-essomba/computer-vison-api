import json 
import codecs
import pandas as pd 
from tensorflow_process.img_read_url import url_img_read_and_save


def build():
    # Spécifiez le chemin de votre fichier JSON
    address = "./data/face_detection.json"

    jsonData = []

    with codecs.open(address, 'rb', 'utf-8') as js:
        for line in js:
            jsonData.append(json.loads(line))

    print(f"{len(jsonData)} image found!")

    print("Sample row:")

    PATHS = []
    Xmin, Ymin, Xmax, Ymax = [[], [], [], []]
    LABELS      = []
    W, H    = [[], []]

    DATA = {"PATH" : PATHS, "xmin" : Xmin, "ymin" : Ymin, "xmax": Xmax, "ymax":Ymax, "width":W, 'height':H, "LABELS" : LABELS}
    index = 0 

    for i, data in enumerate(jsonData):
        if len(data['annotation']) == 1:
            index += 1
            _path_ = f"./data/data/image-{index}.jpg"
            url = data['content']
            PATHS.append( _path_ )
            
            url_img_read_and_save(url=url, path=_path_)

            for items in data['annotation']:
                W.append(items['imageWidth'])
                H.append(items['imageHeight'])

                LABELS.append(items['label'][0]) 
                Xmin.append(items['points'][0]['x'])
                Ymin.append(items['points'][0]['y'])
                Xmax.append(items['points'][1]['x'])
                Ymax.append(items['points'][1]['y'])


    df = pd.DataFrame(DATA)
    df['xmean'] = (df['xmin'] + df['xmax']) / 2.0
    df['ymean'] = (df['ymin'] + df['ymax']) / 2.0
    df['w'] = (df['xmax'] - df['xmin']) 
    df['h'] = (df['ymax'] - df['xmin'])

    df.to_csv('./data/face_detection.csv', sep='\t', header=True, index_label=False)
