import os
from os.path import isfile, join
import sys
import cv2
import time
import art
import numpy as np
import datetime as dt
from termcolor import colored
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from skimage.metrics import mean_squared_error
from tkinter.filedialog import askopenfilename, askdirectory
import tkinter as tk

import moviepy.editor as mp
from colorama import init, Fore; init(autoreset=True)
from PIL import Image; Image.MAX_IMAGE_PIXELS = 933120000


# DEFAULT IS MSE COLOR METHOD
view_progress 		= False
using_mse 			= True
use_background 		= False
auto_search_bg 		= False
crop_center 		= True
add_noncolors 		= True
sharpen				= False
saturate 			= False
background_color 	= "769fb5"
mirror 				= False

NUM_CLUSTERS 	  = 10			#NUMBER OF CLUSTER WHEN SEARCHING FOR BACKGROUND COLOR
tile_size		  = [100,100]	#SIZE OF TILE (w,h)]
Enlargment		  = 6			#HOW MUCH TO ENLARGE THE IMAGE


def imageInfo(image):
		dimensions = image.shape
		height, width, channels = image.shape[:3]
		print(" [" , colored("Image Dimension", 'blue') , "] 		" , dimensions , "px")
		print(" [" , colored("Image Height", 'blue') , "] 		" , width , "px")
		print(" [" , colored("Image Width ", 'blue') , "] 		" , height , "px")
		print(" [" , colored("Color Chanels", 'blue') , "] 		" , channels)

def image_resize(image2):
	h, w = np.shape(image2)[:2]
	ar = tile_size[0] / tile_size[1]
	aspect_ratio = w / h

	if aspect_ratio >= ar:
		scaledHeight = tile_size[1]
		scaledWidth = int(scaledHeight * aspect_ratio)
	else:
		scaledWidth = tile_size[0]
		scaledHeight = int(scaledWidth / aspect_ratio)

	dim = (scaledWidth, scaledHeight)
	im = cv2.resize(image2, dim, interpolation=cv2.INTER_AREA)

	h, w = im.shape[:2]
	h_start = 0
	h_end = h_start + tile_size[1]
	w_start = max(0, (w - tile_size[0]) // 2)
	w_end = w_start + tile_size[0]
	cropped = im[h_start:h_end, w_start:w_end]

	return cropped
	
def main_image(main_img):
	print("[" , colored("Processing main image...", 'green'), "] ") 

	if len(main_img.shape) > 2 and main_img.shape[2] == 4:
		main_img = cv2.cvtColor(main_img, cv2.COLOR_BGRA2BGR)
		print(" > Converted to 3 Color Chanels")

	#increase color saturation
	if saturate:
		hsv = cv2.cvtColor(main_img, cv2.COLOR_BGR2HSV) # convert image to HSV color space
		hsv = np.array(hsv, dtype = np.float64)
		hsv[:,:,1] = hsv[:,:,1]*1.5 # scale pixel values up for channel 1
		hsv[:,:,1][hsv[:,:,1]>255]  = 255
		hsv[:,:,2] = hsv[:,:,2]*1.5 # scale pixel values up for channel 2
		hsv[:,:,2][hsv[:,:,2]>255]  = 255
		hsv = np.array(hsv, dtype = np.uint8)
		main_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	h, w = main_img.shape[:2]

	if Enlargment > 1:
		mpix = w*h//1000000
		print(" [" , colored("Enlarging by", 'blue'), "] 		", Enlargment) 
		tmp_w = main_img.shape[1]*Enlargment
		tmp_h = main_img.shape[0]*Enlargment
		enlargment_tmp = (tmp_w, tmp_h)
		main_img = cv2.resize(main_img, enlargment_tmp, interpolation= cv2.INTER_LINEAR)
	try:
		print(" [" , colored("Resized to", 'blue'), "] 		", main_img.shape, colored("(!numpy array)", "red")) 
		h, w = main_img.shape[:2]
		no_tiles = (w//tile_size[0])*(h//tile_size[1])
		mpix = w*h//1000000
		mem_size = round(main_img.nbytes/1024/1024, 2)
		estimated = round(mem_size/5.27,2)
		print(" [" , colored("Size in Memory", 'blue'), "] 		", mem_size, "mb")
		print(" [" , colored("Estimated Size on disk", 'blue'), "] 	", estimated, "mb")
		if mpix>200:
			print(" [" , colored("Mega Pixels", 'red'), "] 		", mpix)
		else:
			print(" [" , colored("Mega Pixels", 'blue'), "] 		", mpix)
		print(" [" , colored("Number of Tiles to process", 'blue'), "]	", no_tiles) 
		print("[" , colored("Main image processed", 'green'), "] ") 
	except Exception as e:
		print("<<>>",e)
		quit()


	h, w = main_img.shape[:2]
	w_diff = (w // tile_size[0])*tile_size[0]
	h_diff = (h // tile_size[1])*tile_size[1]
	print(" [" , colored("Tile Size", 'blue'), "] 			", tile_size, "px") 
	print(" [" , colored("Resizing to fit Tiles", 'blue'), "] 	", w_diff,"x",h_diff) 

	# if necessary, crop the image slightly so we use a whole number of tiles horizontally and vertically
	down_points = (w_diff, h_diff)
	main_img = cv2.resize(main_img, down_points, interpolation= cv2.INTER_LINEAR)

	return main_img

### NEW Progressbar ?
def progress_bar(percent, text="", bar_len=30):
		SYMBOL = "━"
		done = round(percent*bar_len)
		left = bar_len - done

		print(f"   {Fore.GREEN}{SYMBOL*done}{Fore.RESET}{SYMBOL*left} {f'[{round(percent*100,2)}%]'.ljust(8)} {Fore.MAGENTA}{text}{Fore.RESET}", end='\r')

		if percent == 1: print("✅")

def update_progress(progress, total):
	filled_length = int(round(100 * progress / float(total)))
	sys.stdout.write('\r [\033[1;34mPROGRESS\033[0;0m] [\033[0;32m{0}\033[0;0m]:{1}%'.format('#' * int(filled_length/5), filled_length))
	if progress == total:sys.stdout.write('\n')
	sys.stdout.flush()

def update_progress_time(progress, total, start):
	telapsed = time.time() - start
	testimated = (telapsed/progress)*(total)
	finishtime = start + testimated
	finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time
	lefttime = testimated-telapsed  # in seconds
	remain_m = lefttime // 60
	remain_s = lefttime % 60
	filled_length = int(round(100 * progress / float(total)))
	sys.stdout.write('\r[\033[1;34mCOMPLETION IN {2}:{3}\033[0;0m] [\033[0;32m{0}\033[0;0m]:{1}%'.format('#' * int(filled_length/5), filled_length, int(remain_m), int(remain_s)))
	if progress == total: sys.stdout.write('\n')
	sys.stdout.flush()

def buffer_img():
	print("[" , colored("Buffering Images", 'yellow', attrs=['bold']) , "] ")
	onlyfiles = [f for f in os.listdir(small_pix) if isfile(join(small_pix, f)) & f.endswith(".jpg")]
	
	orig_len = len(onlyfiles)

	if mirror:
		no_files = len(onlyfiles)*2
	else:
		no_files = len(onlyfiles)
	if use_background:no_files = no_files + 1
	if add_noncolors:no_files = no_files + 2
	
	images = np.empty(no_files, dtype=object)
	
	for n in range(0, len(onlyfiles)):
		update_progress(n, len(onlyfiles))
		try:
			im = cv2.imread(join(small_pix, onlyfiles[n]))
			thumbnail = image_resize(im)	
			
			if len(thumbnail.shape) > 2 and thumbnail.shape[2] == 4:
				thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGRA2BGR)
				print(join(small_pix, onlyfiles[n]), " > Converted")
			
			images[n] = thumbnail

			if mirror:
				flipHorizontal = cv2.flip(thumbnail, 1)
				images[orig_len+n] = flipHorizontal

		except Exception as e:
			print()
			print("[" , colored("::Warning >> Load:3:", 'yellow', attrs=['bold', 'blink']) , "]" , " Resizing:"+join(small_pix, onlyfiles[n]))
			print(e)

	print()
	if use_background:
		background_img = np.zeros((tile_size[1], tile_size[0], 3), np.uint8)
		background_color2 = tuple(int(background_color[i:i+2], 16) for i in (0, 2, 4))
		background_color2= (background_color2[2], background_color2[1], background_color2[0])
		background_img[:] = background_color2
		images[no_files-1] = background_img
	
	if add_noncolors:
		background_img = np.zeros((tile_size[1], tile_size[0], 3), np.uint8)
		background_img[:] = 0
		images[no_files-1] = background_img

		background_img = np.zeros((tile_size[1], tile_size[0], 3), np.uint8)
		background_img[:] = 255
		images[no_files-2] = background_img

	print()
	if use_background: print("[" , colored("Added background tile", 'green', attrs=['bold']) , "]", background_color)
	if add_noncolors: print("[" , colored("Added non color  tile", 'green', attrs=['bold']) , "]", images[no_files-1][0,0][:], "<>",images[no_files-2][0,0][:])
	print(" [" , colored("Buffered Images", 'yellow') , "] 		",len(images))
	return images #, blur_images

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
		print(gif_path, new_path)

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

def process_section(args):
	section, images, tile_size = args
	h, w, _ = section.shape
	mosaic_section = np.zeros_like(section)
	section_no_tiles = (w//tile_size[0])*(h//tile_size[1])
	processed_tiles = 0
	section_start_time = time.time()

	for y in range(0, h, tile_size[1]):
		for x in range(0, w, tile_size[0]):
			roi = section[y:y + tile_size[1], x:x + tile_size[0]]

			best_match_index = -1
			best_match_score = float('inf')

			for i, img in enumerate(images):
				score = mean_squared_error(roi, img)
				if score < best_match_score:
					best_match_score = score
					best_match_index = i

			if best_match_index != -1:
				mosaic_section[y:y + tile_size[1], x:x + tile_size[0]] = images[best_match_index]
			
			processed_tiles += 1
			progress = (processed_tiles / section_no_tiles) * 100
			if progress % 10 == 0:  # Check if progress is at a 10% increment
				section_elapsed_time = time.time() - section_start_time  # Calculate elapsed time
				estimated_total_time = section_elapsed_time / (progress / 100)  # Estimate total time based on current progress
				remaining_time = estimated_total_time - section_elapsed_time  # Estimate remaining time

				# Convert remaining time to a readable format (e.g., minutes and seconds)
				remaining_minutes, remaining_seconds = divmod(remaining_time, 60)

				print(f": {progress:.0f}% processed. Estimated time remaining: {int(remaining_minutes)}m {int(remaining_seconds)}s.")


	return mosaic_section

def create_mosaic(image, images, tile_size=(100, 100)):
	h, w, _ = image.shape
	num_cores = cpu_count()

	# Calculate the exact height of each section to fit whole tiles
	tiles_per_section = (h // tile_size[1]) // num_cores
	section_height = tiles_per_section * tile_size[1]

	args_list = []
	for i in range(num_cores):
		start_y = i * section_height
		end_y = start_y + section_height

		# Ensure the last section goes to the end of the image and includes any remaining rows
		if i == num_cores - 1:
			end_y = h

		section = image[start_y:end_y, :, :]
		args_list.append((section, images, tile_size))

	with Pool(num_cores) as pool:
		results = list(tqdm(pool.imap(process_section, args_list), total=num_cores))

	# Combine the processed sections back into the final mosaic image
	mosaic = np.vstack(results)

	return mosaic


if __name__ == "__main__":
	root = tk.Tk()
	root.overrideredirect(1)
	root.withdraw()

	os.system('cls')
	print()
	art.tprint("MultiProcess Mosaic", font="nancyj-fancy") 
	print("by Ivan Rogoz")
	print()

	picture =  askopenfilename()
	print("[" ,colored("Opening picture", 'green', attrs=['bold']) , "] 		",picture)
	img = cv2.imread(picture, cv2.IMREAD_UNCHANGED)
	if not img.any():      # always check for None
			raise ValueError("unable to load Image")

	if sharpen:
		blur_img = cv2.GaussianBlur(img, (0, 0), 15)
		img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

	cv2.imshow('Original', img)

	imageInfo(img)
	original = img.copy()
	img=main_image(img)

	print("[" , colored("View Progress", 'green') , "] 		", view_progress)

	small_pix = askdirectory()
	print("[" , colored("Using Structural Similarity Index", 'yellow', attrs=['bold']) , "] ")
	print("  [" , colored("Computationaly heavy", 'red', attrs=['bold']) , "] ")
	images = buffer_img()

	start_time = time.time()

	mosaic = create_mosaic(img, images, tile_size)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"Total execution time: {elapsed_time:.2f} seconds.")

	# cv2.imwrite('111mosaic.jpg', mosaic)
	# cv2.destroyAllWindows()

	w, h = mosaic.shape[:2]
	used=0

	_, tail = os.path.split(picture)
	tail = os.path.splitext(tail)[0]
	try:
		fullIndex = tail.index('full')+4
	except:
		fullIndex = 0
	tail = tail[fullIndex:]
	count = len(images)
	no_tiles = (w//tile_size[0])*(h//tile_size[1])

	save_vid= True
	quality= 95
	max_zoomed_images= 5
	zoom_incr= 1.02
	frame_duration= 30

	print("[" , colored(" DONE ", 'green', attrs=['bold']) , "]")
	print("[" , colored(" WRITING TO DISK ", 'red', attrs=['bold']) , "]")
	i = 0

	while os.path.exists(f"mosaic_{tail}_{i}_{count}_{no_tiles}_o_{used}.jpg"):
		i += 1
	filename = f"mosaic_{tail}_{i}_{count}_{no_tiles}_o_{used}.jpg"
	filename_gif = f"mosaic_{tail}_{i}_{count}_{no_tiles}_o_{used}"

	isWritten  = cv2.imwrite(filename, mosaic)
	if isWritten:
		print("[" , colored(" Mosaic Writen to disk as " + filename, 'green', attrs=['bold']) , "]")
	else:
		print("[" , colored(" Error writing Mosaic to disk ", 'red', attrs=['bold']) , "]")

	print("[" ,colored("Creating Gifs and Videos", 'red', attrs=['bold']) , "]")
	save_zooms_gif(mosaic, filename_gif, original.shape, tile_size[1], save_vid, quality, max_zoomed_images, zoom_incr, frame_duration)
