import os
import glob
from pathlib import Path
from PIL import Image
from skimage import io
from face_detection import RetinaFace
import cv2
import numpy as np


def crop(cropper, image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB' and image.mode != 'RGBA':
        new_image = Image.new('RGB', image.size)
        new_image.paste(image)
        image = new_image

    
    result = cropper.crop(image, width=500, height=500)    
    box = (
        result['top_crop']['x'],
        result['top_crop']['y'],
        result['top_crop']['width'] + result['top_crop']['x'],
        result['top_crop']['height'] + result['top_crop']['y']
    )
    cropped_image = image.crop(box)
    cropped_image.thumbnail((500,500), Image.LANCZOS)
    return cropped_image

def face_detector(detector, img_path):
    img= io.imread(img_path)
    faces = detector(img)
    return faces[0]


def create_square_box_expanded(img, x, y, w, h, size=512):
    # Calculate center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Determine the largest dimension and extend the shorter side to match
    max_dim = max(w, h)
    half_dim = max_dim // 2

    # Compute the new square bounding box coordinates
    new_x = center_x - half_dim
    new_y = center_y - half_dim
    new_w = max_dim
    new_h = max_dim

    # Expand the bounding box to specified size, handling corner cases
    expand = (size - max_dim) // 2
    new_x -= expand
    new_y -= expand
    new_w += 2 * expand
    new_h += 2 * expand

    img_height, img_width = img.shape[:2]

    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0

    if new_x + new_w > img_width:
        new_w = img_width - new_x
    if new_y + new_h > img_height:
        new_h = img_height - new_y

    return new_x, new_y, new_w, new_h

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def smart_crop(input_folder, output_folder, target_size=738):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob.glob(input_folder + '/*.*')
    detector = RetinaFace()
    for img_path in image_paths:
        print(img_path)
        img = cv2.imread(img_path)
        filename = Path(img_path).name
        out_img = os.path.join(output_folder, filename)
        box, landmarks, score=face_detector(detector, img_path)
        point1 = (int(box[0]), int(box[1]))
        point2 = (int(box[2]), int(box[3]))
        # thickness = 2
        new_x, new_y, new_w, new_h= create_square_box_expanded(img, point1[0], point1[1], point2[1]-point1[1],point2[0]-point1[0], target_size)
        cropped_image = img[new_y:new_y+new_h, new_x:new_x+new_w]

        # print(new_w, new_h)
        # cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), thickness)
        cv2.imwrite(out_img, cropped_image)
        # print(box, score)

def main():
    input_folder = 'liuyifei'
    output_folder = 'output'
    target_size = 768

    smart_crop(input_folder, output_folder, target_size)

if __name__ == '__main__':
    main()