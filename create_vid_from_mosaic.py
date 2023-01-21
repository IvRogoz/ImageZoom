
import os
import cv2
import numpy as np
import moviepy.editor as mp
from colorama import init, Fore; init(autoreset=True)
from PIL import Image; Image.MAX_IMAGE_PIXELS = 933120000

from termcolor import colored

from tkinter.filedialog import askopenfilename, askdirectory
import tkinter as tk

root = tk.Tk()
root.overrideredirect(1)
root.withdraw()

original_picture =  askopenfilename()
print("[" ,colored("Opening Original picture", 'green', attrs=['bold']) , "] 	",original_picture)
original_img = cv2.imread(original_picture, cv2.IMREAD_UNCHANGED)
if not original_img.any():      # always check for None
		raise ValueError("unable to load Image")
cv2.imshow('Original', original_img)

def get_file_size(path):
    return os.path.getsize(path)/2**20

def create_zoom_img(img_arr, full_shape, main_img_shape, zoom, max_res):
    max_res = int(np.ceil(max_res/2) * 2) # Make it even (res can't be odd)
    center_x = full_shape[0]//2
    center_y = full_shape[1]//2
    left = center_y - int(full_shape[1]/(zoom*2))
    right = center_y + int(full_shape[1]/(zoom*2))
    top = center_x - int(full_shape[0]/(zoom*2))
    bottom = center_x + int(full_shape[0]/(zoom*2))
    img = img_arr[top:bottom, left:right]
    if main_img_shape[0] > main_img_shape[1]:
        height = int(np.ceil(max_res*main_img_shape[0]//main_img_shape[1]/2)*2) # To make it even (res can't be odd)
        img_res = cv2.resize(img, (max_res, height), interpolation=cv2.INTER_AREA)
    else:
        width = int(np.ceil(max_res*main_img_shape[1]//main_img_shape[0]/2)*2) # To make it even (res can't be odd)
        img_res = cv2.resize(img, (width, max_res), interpolation=cv2.INTER_AREA)
    return img_res

def save_gif_func(images, path, quality=95, frame_duration=30):
    images[0].save(path, format="GIF", append_images=images, save_all=True, duration=frame_duration, loop=0, quality=quality)
    print(f"{Fore.GREEN}GIF saved{Fore.RESET} ({get_file_size(path):.2f} MB)")

def save_vid_gif(gif_path, new_path):
    mp.VideoFileClip(gif_path).write_videofile(new_path, logger=None)
    print(f"{Fore.GREEN}Video saved{Fore.RESET} ({get_file_size(new_path):.2f} MB)")

def save_zooms_gif(img_arr, new_name, main_img_shape, images_size, save_vid, quality, max_zoomed_images, zoom_incr, frame_duration, max_res=1080):
    full_shape = img_arr.shape
    gif_images = []
    zoom = 1

    zoom_img = []
    start = True

    while min(full_shape[0], full_shape[1])/zoom > images_size*max_zoomed_images:
        zoom_img = cv2.cvtColor(create_zoom_img(img_arr, full_shape, main_img_shape, zoom, max_res), cv2.COLOR_BGR2RGB)
        if start:
            for i in range(60):
                gif_images.append(Image.fromarray(zoom_img))
            start = False
        gif_images.append(Image.fromarray(zoom_img))
        zoom *= zoom_incr
        
    for i in range(60):
        gif_images.append(Image.fromarray(zoom_img))
    
    gif_images = gif_images[::-1]
    print("array length:", len(gif_images))
    tmp_img=gif_images[::-1].copy()
    gif_images.extend(tmp_img)
    print("array length:", len(gif_images))

    if save_vid:
        gif_path = f"./{new_name}.gif"
        save_gif_func(gif_images, gif_path, quality, frame_duration)
        save_vid_gif(gif_path, f"./{new_name}.mp4")


_, tail = os.path.split(original_picture)
tail = os.path.splitext(tail)[0]

i = 0
while os.path.exists(f"mosaic_{tail}__{i}.jpg"):
	i += 1
new_name= f"mosaic_{tail}__{i}"

new_folder = f"./{new_name}"
save_vid= True
quality= 95
max_zoomed_images= 5
zoom_incr= 1.02
frame_duration= 30
tile_size = 100

print("[" ,colored("Creating Gifs and Videos", 'red', attrs=['bold']) , "]")
save_zooms_gif(original_img, new_name, original_img.shape, tile_size, save_vid, quality, max_zoomed_images, zoom_incr, frame_duration)
