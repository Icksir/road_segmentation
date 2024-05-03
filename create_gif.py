from PIL import Image
import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def make_gif(frame_folder):

    # Take list of paths for images
    image_path_list = [img for img in os.listdir(frame_folder) if img.endswith(".png")]
    image_path_list = sorted_alphanumeric(image_path_list)

    # Create a list of image objects
    image_list = [Image.open(f"{frame_folder}/{file}") for file in image_path_list]

    # Save the first image as a GIF file
    image_list[0].save(
                'animation.gif',
                save_all=True,
                append_images=image_list[1:], # append rest of the images
                duration=100, # in milliseconds
                loop=0)
    
if __name__ == "__main__":
    make_gif("output")