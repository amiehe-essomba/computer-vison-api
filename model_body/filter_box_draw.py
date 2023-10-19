import colorsys
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

def get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    get_colors_for_classes.colors = colors  # Save colors for future calls.

    return colors

def draw_boxes(boxes, w, h, width):
    image = Image.new('RGB', (w, h), color=(0, 0, 0))  
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = (image.size[0] + image.size[1]) // 300
    colors      = get_colors_for_classes(boxes.shape[0])

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2  = boxes[i]
        draw.rectangle([x1, y1, x2, y2], outline=colors[i], width=width)

    image_array = np.array(image)

    return image_array


    