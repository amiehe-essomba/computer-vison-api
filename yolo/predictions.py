from PIL import Image
import os
from yolo.yolo_head import yolo_head
from yolo.eval import yolo_eval
from yolo.utils.tools import  draw_boxes
from skimage.transform import resize
from grad_cam.grad_cam import CompGradcam

def area_conv(area, shape):
    for key, value in area.items():
        area[key] = ( ((value[0] * 608) / shape[1]), ((value[1] * 608) / shape[0]) )
    
    return area

def prediction(
        yolo_model  : any, 
        image_file  : list, 
        anchors     : int, 
        class_names : list, 
        img_size    : tuple     = (1, 1),
        iou_threshold   : float = 0.5, 
        score_threshold : float = 0.6, 
        max_boxes       : int   = 10, 
        use_classes     : list  = [],
        data_dict       : dict  = {}, 
        save_img        : bool  = False,
        image_path      : str   = '',
        action          : bool  = False,
        shape           : tuple = (),
        file_type       : str   = 'image',
        with_score      : bool  = True,
        colors          : dict  = {},
        grad_cam        : bool  = False,
        area            : dict  = {}
        ):

    out_scores, out_boxes, out_classes  = [[], [], []]
                                           
    draw_image = None 
    import numpy as np
    if image_file:
        for i, _ in enumerate(image_file):
            try:
                # Preprocess your image
                IMG, image_data = image_file[i]
                image = IMG.copy()
                
                if file_type == 'image':
                    if grad_cam:
                        grad, guided_grad = CompGradcam(model=yolo_model, img=image_data.reshape((608, 608, 3)))
                        guided_grad = resize(np.array(guided_grad), output_shape=shape)
                        grad = resize(np.array(grad), output_shape=shape)
                
                image_data = image_data.reshape((1, img_size[0], img_size[0], 3))
            
                yolo_model_outputs = yolo_model(image_data)
                
                yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
                
                out_score, out_boxe, out_classe = yolo_eval(yolo_outputs, [image.size[1],  
                                        image.size[0]], max_boxes, score_threshold, iou_threshold)

                # Draw bounding boxes on the image file
                if area:
                    area = area_conv(area=area, shape=shape)
                draw_image = draw_boxes(image=image, boxes=out_boxe, box_classes=out_classe, with_score=with_score, 
                                        class_names=class_names, scores=out_score, use_classes=use_classes, 
                                        df=data_dict, colors=colors, area=area)
                
                r = shape[0] / shape[1] 

                if file_type == 'image':
                    draw_image = resize(draw_image, output_shape=shape)
                else:
                    if draw_image.shape[0] <= shape[0]:
                        if draw_image.shape[1] <= shape[0]:
                            shape = draw_image.shape[:-1]
                        else:
                            shape = (draw_image.shape[0], shape[1])
                    else:
                        if draw_image.shape[1] <= shape[0]:
                            shape = (shape[0], draw_image.shape[1])
                        else: pass 
                    
                    shape = (int( shape[1] * r), shape[1])
                    draw_image = resize(draw_image, output_shape=shape).astype("float32")
                                      
                draw_image = resize(draw_image, output_shape=shape)

                
                # saving section
                if save_img:
                    if action:
                        # Save the predicted bounding box on the image
                        image.save(image_path)

                out_scores.append(out_score), out_boxes.append(out_boxe), out_classes.append(out_classe)

                if grad_cam:
                    return draw_image, grad, guided_grad
                
            except (FileNotFoundError, FileExistsError) : 
                print('ERRROR')
                break
    else : print('ERROR')

    return draw_image

