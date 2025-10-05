from PIL import Image
import os
from PIL import Image

def convert_gif_to_jpg(directory):
    print("Converting .gif files to .jpg...")
    for root, dirs, files in os.walk(directory):
        for filename in files:
            print(filename)
            print(dirs)
            gif_path = os.path.join(root, filename)
            jpg_path = os.path.join(root, os.path.splitext(filename)[0] + ".jpg")
            
            # Open the .gif file
            with Image.open(gif_path) as gif_image:
                # Convert and save as .jpg
                gif_image.convert("RGB").save(jpg_path, "JPEG")
            
            # Remove the original .gif file
            os.remove(gif_path)
    print("Conversion complete!")

convert_gif_to_jpg("./Pipeline/Data/Datasets/MPEG7")