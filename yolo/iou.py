def IoU(
        box_xy1 : tuple = (), 
        box_xy2 : tuple = ()
        ):

    """
    * ----------------------------------------------------------------------------- 

    >>> AUTOR : < Iréné Amiehe-Essomba > 
    >>> Copyright (c) 2023

    * -----------------------------------------------------------------------------

    (b1_x1, b1_y1)(min)
    +---------------------------------------------+
    |                 box 2                       |
    |                                             |
    |      (b2_x1, b2_y1)(min)                    |
    |        +----------------------------+-------+
    |        |/////// bonding box /////// |       |
    +--------+------------------------------------+ (b1_x2, b1_y2) (max) 
             |                            |
             |        box 1               |
             |                            |
             +----------------------------+ (b2_x2, b2_y2) (max)
    """
    
    # boxes coordinates 
    (b1_x1, b1_y1, b1_x2, b1_y2) = box_xy1
    (b2_x1, b2_y1, b2_x2, b2_y2) = box_xy2
    
    # bonding box coordinates
    xi1 = max(b1_x1, b1_x2)
    yi1 = max(b1_y1, b1_y2)

    xi2 = min(b2_x2, b2_x1)
    yi2 = min(b2_y1, b2_y2)

    inner_width     = (xi1 - xi2)
    inner_height    = (yi1 - yi2)

    # bonding box surface calculation
    bonding_box_surface = max(inner_width, 0) * max(inner_height, 0)

    # surface area of each box 
    surface_box1    = b1_x2 * b1_y2
    surface_box2    = b2_x2 * b2_y2 

    # global surface = union(box1, box2) - inter(box1, box2)
    global_surface  = surface_box1 + surface_box2 - bonding_box_surface 


    # compute iou values 
    iou             = (bonding_box_surface) / global_surface 


    return iou