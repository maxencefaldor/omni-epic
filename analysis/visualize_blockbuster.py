from manim import (
    Scene, config, MarkupText, ImageMobject,
    LEFT, RIGHT, UP, DOWN,
)
import argparse
import re
import os
import json
import uuid
import cv2
import tempfile
import numpy as np

from analysis.visualize_taskgen import read_last_json_entry, extract_task_id
from run_utils import get_task_desc_from_env_path, get_task_success_file_from_folder

def write_frames_to_video(frames, video_path, fps=20):
    # Assuming all frames are of the same shape and dtype
    height, width, layers = frames[0].shape
    size = (width, height)
    # Use the mp4v codec for MP4 format
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)  # Write out frame to video
    out.release()  # Release the video writer

def process_recording_file(file_path, output_dir=None):
    """ Read the recorded_actions.jsonl file and save the recorded frames as a video. """
    base_folder = output_dir if output_dir else os.path.dirname(file_path)
    video_folder = os.path.join(base_folder, 'videos')
    os.makedirs(video_folder, exist_ok=True)

    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                env_path = data['env_filepath']
                env_dirname = os.path.basename(os.path.dirname(env_path))
                video_path = os.path.join(video_folder, f"{env_dirname}_{str(uuid.uuid4())[:8]}.mp4")
                recorded_frames = [np.array(d, dtype=np.uint8) for d in data['recorded_frames'] if d is not None]
                write_frames_to_video(recorded_frames, video_path)
                yield env_path, video_path
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def process_dir(dir_path, recorded_actions=False):
    video_json = {}  # Keep track of the videos generated for each environment
    if recorded_actions:
        """ Process all the recorded_actions.jsonl files in the directory. """
        for root, _, files in os.walk(dir_path):
            print('Processing:', root)
            for file in files:
                if file.endswith('recorded_actions.jsonl'):
                    for env_path, video_path in process_recording_file(os.path.join(root, file), output_dir=dir_path):
                        video_json[env_path] = video_json.get(env_path, []) + [video_path]
                        # if len(video_json[env_path]) > 0:  # for debugging purposes
                        #     return video_json
    else:
        """ Process all the videos in the directory. """
        archive_data = read_last_json_entry(os.path.join(dir_path, 'archive.jsonl'))
        env_paths = archive_data['codepaths'] + archive_data['failedtrain']
        for env_path in env_paths:
            taskname = os.path.basename(os.path.dirname(env_path))
            print('Processing:', taskname)
            video_folder = os.path.join(dir_path, f'{taskname}/dreamer/eval')
            success_path = get_task_success_file_from_folder(video_folder)
            if success_path:  # Get the successful video file
                success_dir, success_file = os.path.split(success_path)
                video_file = success_file.replace('success', 'render').replace('.txt', '.mp4')
                video_path = os.path.join(success_dir, video_file)
                video_json[env_path] = video_json.get(env_path, []) + [video_path]
            else:  # Get the first video file found in the folder
                for file in os.listdir(video_folder):
                    if file.endswith('.mp4') and file.startswith('render_'):
                        video_path = os.path.join(video_folder, file)
                        video_json[env_path] = video_json.get(env_path, []) + [video_path]
                        break
    return video_json

def wrap_text(text, character_length=50, trunc_lines=5):
    """ Dynamically wrap text to fit into the character length. """
    wrapped_text = []
    lines = text.split('\n')
    lines = [lines[0] + '\n', ' '.join(lines[1:])]
    for i, line in enumerate(lines):
        words = line.split(' ')
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= character_length:
                line += word + " "
            else:
                if len(wrapped_text) >= trunc_lines:
                    break
                wrapped_text.append(line)
                line = word + " "
        if len(wrapped_text) < trunc_lines:
            wrapped_text.append(line)
        if len(wrapped_text) >= trunc_lines:
            wrapped_text[-1] = wrapped_text[-1].rstrip() + "..."
            break
    return '\n'.join(wrapped_text)

class SimpleAnimation(Scene):
    def __init__(self, dirpath, video_json, n_examples=0, **kwargs):
        super().__init__(**kwargs)
        self.dirpath = dirpath
        self.video_json = video_json
        self.n_examples = n_examples

    def construct(self):
        archive_data = read_last_json_entry(os.path.join(self.dirpath, 'archive.jsonl'))
        archive_comments_path = os.path.join(self.dirpath, 'archive_comments.jsonl')
        archive_comments = read_last_json_entry(archive_comments_path) if os.path.exists(archive_comments_path) else {}
        env_paths = archive_data['codepaths'] + archive_data['failedint'] + archive_data['failedtrain']
        env_paths = sorted(env_paths, key=lambda x: int(extract_task_id(x).split('_')[1]))
        example_counter = 0
        task_counter = 0

        for env_path in env_paths:
            video_paths = self.video_json.get(env_path, [])
            # Skip if no video paths are found
            if not video_paths:
                task_counter += 1
                continue
            # Get task information
            task_id = extract_task_id(env_path)
            if example_counter < self.n_examples:
                example_counter += 1
                task_id = f'Seed {example_counter}'
            else:
                task_counter += 1
                task_id = f"Task {task_counter}"
            task_id = f'<span foreground="#FFFF00">{task_id}</span>'
            task_desc = get_task_desc_from_env_path(env_path)
            task_desc = re.split('[.\n]', task_desc)[0]  # Only take the first line of the task description
            task_desc = wrap_text(task_desc, character_length=30, trunc_lines=8)
            task_success = env_path in archive_data['codepaths']
            success_color = '#2FFF2F' if task_success else '#FF4911'
            task_success_text = f'<span foreground="{success_color}" font_size="smaller">{"success detector: true" if task_success else "success detector: false"}</span>'
            task_comments = wrap_text(archive_comments.get(env_path, ''), character_length=35, trunc_lines=5).lower()
            task_comments = f'<span font_size="smaller">{task_comments}</span>'

            # Render task information
            task_info_text = MarkupText(
                f'{task_id}\n\n{task_desc}\n{task_success_text}\n{task_comments}',
                color='white',
                font_size=25,
            )
            task_info_text.to_edge(LEFT)
            self.add(task_info_text)
            self.wait(1.0)

            # Get videos for the task
            for video_file in video_paths:
                cap = cv2.VideoCapture(video_file)
                # tmp_i = 0  # for debugging purposes
                while cap.isOpened():
                    ret, frame = cap.read()
                    # tmp_i += 1
                    if not ret:
                        break
                    # Write frame to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
                        cv2.imwrite(tmpfile.name, frame)
                        img_mobject = ImageMobject(tmpfile.name)
                        img_mobject.width = config['frame_width'] / 2
                        img_mobject.to_edge(RIGHT)
                        self.add(img_mobject)
                        self.wait(0.05)
                        self.remove(img_mobject)
                    # if tmp_i > 1:
                    #     break
                cap.release()

            # Remove current task information rendering
            self.remove(task_info_text)

def main():
    # Setup argparse for command line arguments
    parser = argparse.ArgumentParser(description="Create a blockbuster using Manim")
    parser.add_argument("--dirpath", type=str, required=True, help="Path to the directory")
    parser.add_argument("--recorded_actions", action='store_true', help="Read recorded_actions.jsonl instead of the already rendered videos.")
    parser.add_argument("--high-quality", action='store_true', help="Render the video in high quality.")
    parser.add_argument("--n-examples", "-e", type=int, default=0, help="Number of tasks that are examples.")
    args = parser.parse_args()

    # Process the directory containing multiple recorded_actions.jsonl files
    video_json = process_dir(args.dirpath, recorded_actions=args.recorded_actions)

    # Additional configuration for rendering
    if args.high_quality:
        config.pixel_height = 720
        config.pixel_width = 1280
        config.frame_rate = 60
    else:
        config.pixel_height = 360
        config.pixel_width = 640
        config.frame_rate = 30
    config.media_dir = os.path.join(args.dirpath, 'media')
    config.disable_caching = True

    # Create manim video
    scene = SimpleAnimation(args.dirpath, video_json, n_examples=args.n_examples)
    scene.render()

if __name__ == "__main__":
    main()
