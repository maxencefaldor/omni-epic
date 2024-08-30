import cv2
import base64
import numpy as np
import os
import re
import json
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from textwrap import dedent

from embodied.envs.pybullet import PyBullet
from omni_epic.robots import robot_dict
from omni_epic.core.fm import FM


# Function to get images at specified intervals from a video file
def get_images_from_video(video_file, interval=62):
	# Open the video file
	cap = cv2.VideoCapture(video_file)
	if not cap.isOpened():
		print("Error: Could not open video.")
		return None
	# Calculate the interval between each image to be captured
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# Skip the first 10 frames to get to the interesting parts
	frames_to_capture = range(10, total_frames, interval)
	images = []
	for frame_id in frames_to_capture:
		# Read the current frame position of the video file
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
		ret, frame = cap.read()
		if ret:
			images.append(frame)
		else:
			print(f"Error: Could not read frame {frame_id}")
			break
	# Release the video capture object
	cap.release()
	return images

# Save images into output directory
def save_images(images, output_dir):
	os.makedirs(output_dir, exist_ok=True)
	# Save individual images
	for i, image in enumerate(images):
		cv2.imwrite(f'{output_dir}/image_{i}.png', image)

	# Label image number on the top left corner of each image
	for i, image in enumerate(images):
		cv2.putText(image, f'{i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

	n_per_row = 8
	# Calculate the number of images to be padded
	padded_images = images.copy()
	remainder = len(padded_images) % n_per_row
	if remainder != 0:
		padding = n_per_row - remainder
		# Create a dummy image with the same shape as the last image in the list
		dummy_image = np.zeros_like(padded_images[-1])
		# Add the dummy image to the list of images
		padded_images.extend([dummy_image] * padding)

	# Save concated images, only have N images per row
	concat_image = np.concatenate([
		 np.concatenate(padded_images[i:i+n_per_row], axis=1) \
			for i in range(0, len(padded_images), n_per_row)], axis=0)
	cv2.imwrite(f'{output_dir}/concat_image.png', concat_image)

	return concat_image

# Function to encode the image
def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')

def get_envcode_path(run_folder):
	input_config = OmegaConf.load(os.path.join(run_folder, "./.hydra/config.yaml"))
	return input_config['env']['path']

def parse_task_desc_from_env_code(env_code):
	# Only search after class definition
	task_desc = re.search(r'class Env.*?:\s*\"\"\"(.+?)\"\"\"', env_code, re.DOTALL).group(1)
	# For each line in taskdesc, remove leading and trailing whitespaces
	task_desc = '\n'.join([line.strip() for line in task_desc.split('\n')]).strip()
	return task_desc

def get_task_desc_from_env_path(env_path):
	env = PyBullet(env_path=env_path, vision=False)._env
	task_desc = dedent(env.__doc__).strip()
	return task_desc

def get_task_success_from_file(success_file):
	# Read file
	with open(success_file, "r") as f:
		text = f.read().strip()
		step_successes = text.split('\n')
		step_successes = [x == 'True' for x in step_successes]
	# Determine final task success
	success = any(step_successes)
	return success

def get_task_success_from_folder(run_folder, voting='majority'):
	# Get task success from saved files
	success_files = [f for f in os.listdir(run_folder) if f.endswith('.txt') and f.startswith('success')]
	success_files = [os.path.join(run_folder, f) for f in success_files]
	# Process overall task success
	task_successes = [get_task_success_from_file(f) for f in success_files]
	if voting == 'majority':
		task_success = sum(task_successes) >= len(task_successes) / 2
	elif voting == 'all':
		task_success = all(task_successes)
	else:
		task_success = any(task_successes)
	return task_success

def get_task_success_file_from_folder(run_folder):
	# Get task success from saved files
	success_files = [f for f in os.listdir(run_folder) if f.endswith('.txt') and f.startswith('success')]
	success_files = [os.path.join(run_folder, f) for f in success_files]
	# Process overall task success
	task_successes = [get_task_success_from_file(f) for f in success_files]
	# Return the first successful file
	for i, task_success in enumerate(task_successes):
		if task_success:
			return success_files[i]
	return None
