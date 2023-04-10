import os
import glob
from pathlib import Path
from PIL import Image
from smartcrop import SmartCrop

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
    

def smart_crop(input_folder, output_folder, target_size=500):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob.glob(input_folder + '/*.*')
    cropper = SmartCrop()
    for img_path in image_paths:
        filename = Path(img_path).name
        out_img = os.path.join(output_folder, filename)
        cropped_image=crop(cropper, img_path)
        cropped_image.save(out_img, 'JPEG', quality=100)

def main():
    input_folder = 'hanabunny_school'
    output_folder = 'output'
    target_size = 500

    smart_crop(input_folder, output_folder, target_size)

if __name__ == '__main__':
    main()