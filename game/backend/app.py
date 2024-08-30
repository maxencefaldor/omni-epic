import eventlet
eventlet.monkey_patch()

from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from flask import Flask, render_template, Response, request, jsonify

import requests
import threading
import time

from textwrap import dedent
import re
import json
import os
from datetime import datetime
import cv2
import numpy as np
from omegaconf import OmegaConf
from absl import logging
from absl import app as absl_app
import shutil

from embodied.envs.pybullet import PyBullet
from main import main


app = Flask(__name__)

# Configure logging
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGDIR = os.path.abspath(f"/workspace/src/game/backend/omni_epic_logs/{current_time}")
os.makedirs(LOGDIR, exist_ok=True)
logging.get_absl_handler().use_absl_log_file('application', LOGDIR)
logging.set_verbosity(logging.INFO)

# Adding CORS policies here
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")


def load_archive(archive_filepath):
    # Load the archive file and return the codepaths
    with open(archive_filepath, 'r') as f:
        content = f.read()
        json_str = re.split('(?<=})\n(?={)', content)[-1]
        json_obj = json.loads(json_str)
    return json_obj["codepaths"]


# Initialize the game environment
if not os.path.exists(os.path.join(LOGDIR, "archive.jsonl")):
    shutil.copy("./game/backend/env_codes/archive.jsonl", LOGDIR)
ARCHIVE = load_archive(os.path.join(LOGDIR, "archive.jsonl"))
ARCHIVE_INDEX = 0
LOOP_ARCHIVE = False  # Loop through the archive, instead of generating new levels
ENV_PATH = ARCHIVE[ARCHIVE_INDEX]
ENV = PyBullet(env_path=ENV_PATH, vision=False)._env
TASK_DESC = dedent(ENV.__doc__).strip().split('\n')[0]
ENV.reset()
SUCCESS = False

# Recording actions
IS_RECORDING = False
RECORDED_ACTIONS = []
RECORDED_FRAMES = []
MAX_RECORD_STEPS = 10000

# Generation variables
TASKGEN_CHOOSE_PROBS = np.ones(len(ARCHIVE))
ITERATE_SAME_TASK_COUNT = 0
GENERATING_NEXT_LEVEL = False

# Access variables
ACCESS_USER_ID = None
CURR_ACCESS_CODE = None
SECRET_ACESS_CODE = "omni_epic_is_awesome_0123"


@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global ENV, SUCCESS, GENERATING_NEXT_LEVEL
    while True:
        socketio.sleep(0.01)  # Let SocketIO manage app sleeping
        for i in range(5):
            action_do_nothing = 0
            observation, reward, terminated, truncated, info = ENV.step(action_do_nothing)
            # Skip recording the last empty action
            if i < 4:
                record_action_logic(action_do_nothing, None)
            # # Check if the game is terminated
            # if terminated:
            #     handle_reset()
            #     break

        # Emit reward to the frontend
        socketio.emit('reward_update', {'reward': reward})
        # Update generating next level status
        socketio.emit('generating_next_level', {'generating': GENERATING_NEXT_LEVEL})

        # Check if the level is successfully completed
        SUCCESS = ENV.get_success() or SUCCESS
        if SUCCESS:
            socketio.emit('level_complete', {'message': 'Level completed!'})

        frame = ENV.render3p(height=180, width=320)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert frame from BGR to RGB
        record_action_logic(action_do_nothing, frame.tolist())  # record the last empty action
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# TODO: Add an ID from user socket io so that the feeds are different for different users
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    global TASK_DESC
    user_id = request.sid  # Get the ID of the connected user
    # print(f"User {user_id} connected")  # Print the ID of the connected user
    socketio.emit('connected_message', {'message': 'You are connected!'}, room=request.sid)  # Send a connected message to the connected user
    join_room(request.sid)  # Join a room with the user's ID
    socketio.emit('env_description', {'description': TASK_DESC})  # Update env description

def record_action_logic(action, frame=None):
    global IS_RECORDING, RECORDED_ACTIONS, RECORDED_FRAMES, MAX_RECORD_STEPS
    if IS_RECORDING:
        if len(RECORDED_ACTIONS) < MAX_RECORD_STEPS:
            RECORDED_ACTIONS.append(action)
            RECORDED_FRAMES.append(frame)  # NOTE: comment this out to not record frames, save space when debugging
        else:
            stop_recording()  # Automatically stop recording if max steps reached

def check_valid_user():
    global ACCESS_USER_ID
    user_id = request.sid
    if ACCESS_USER_ID is None or user_id != ACCESS_USER_ID:
        # print(f"User {user_id} is not authorized")
        return False
    return True

@socketio.on('action')
def handle_action(json):
    global ENV, ACCESS_USER_ID

    if not check_valid_user():
        return

    action = int(json['action'])
    repeat_num = 5 if action in [1, 2] else 2
    for _ in range(repeat_num):
        observation, reward, terminated, truncated, info = ENV.step(action)
        record_action_logic(action, None)

@socketio.on('reset')
def handle_reset():
    global ENV, IS_RECORDING, SUCCESS

    if not check_valid_user():
        return

    logging.info("Resetting the game")
    ENV.reset()
    SUCCESS = False
    socketio.emit('reset_message', {'message': 'Game reset! Start again!'})
    if IS_RECORDING:
        stop_recording()
    socketio.emit('env_description', {'description': TASK_DESC})

@socketio.on('prev_level')
def handle_prev_level():
    global ARCHIVE, ARCHIVE_INDEX, ENV, ENV_PATH, TASK_DESC, \
           IS_RECORDING, SUCCESS, TASKGEN_CHOOSE_PROBS, ITERATE_SAME_TASK_COUNT, \
           LOOP_ARCHIVE, GENERATING_NEXT_LEVEL

    if not check_valid_user():
        return

    logging.info("Loading previous level")

    # Stop recording
    if IS_RECORDING:
        stop_recording()

    # Load the previous env in the archive
    if ARCHIVE_INDEX > 0:
        ARCHIVE_INDEX -= 1

    # Load the new env
    ENV_PATH = ARCHIVE[ARCHIVE_INDEX]
    ENV.close()  # Close the current env
    ENV = PyBullet(env_path=ENV_PATH, vision=False)._env
    TASK_DESC = dedent(ENV.__doc__).strip().split('\n')[0]
    ENV.reset()
    SUCCESS = False
    socketio.emit('next_level_message', {'message': 'Next level started!'})
    socketio.emit('env_description', {'description': TASK_DESC})

@socketio.on('next_level')
def handle_next_level():
    global ARCHIVE, ARCHIVE_INDEX, ENV, ENV_PATH, TASK_DESC, \
           IS_RECORDING, SUCCESS, TASKGEN_CHOOSE_PROBS, ITERATE_SAME_TASK_COUNT, \
           LOOP_ARCHIVE, GENERATING_NEXT_LEVEL

    if not check_valid_user():
        return

    logging.info("Generating next level")
    GENERATING_NEXT_LEVEL = True

    # Stop recording
    if IS_RECORDING:
        stop_recording()

    # Check if we need to generate a new level
    if ARCHIVE_INDEX >= len(ARCHIVE) - 1:
        if LOOP_ARCHIVE:
            # Loop through the archive
            ARCHIVE_INDEX = 0
        else:  # Generate a new level with OMNI-EPIC
            # Load OMNI-EPIC config
            config = OmegaConf.load(f"/workspace/src/configs/omni_epic.yaml")

            # Current env failed
            if not SUCCESS:
                TASKGEN_CHOOSE_PROBS = np.delete(TASKGEN_CHOOSE_PROBS, -1)
                # Update archive with the failed env
                with open(os.path.join(LOGDIR, 'archive.jsonl'), 'r') as f:
                    content = f.read()
                    json_str = re.split('(?<=})\n(?={)', content)[-1]
                    json_obj = json.loads(json_str)
                    json_obj["codepaths"] = [x for x in json_obj["codepaths"] if x != ENV_PATH]
                    json_obj["failedtrain"].append(ENV_PATH)
                    with open(os.path.join(LOGDIR, 'archive.jsonl'), 'a') as f:
                        f.write(json.dumps(json_obj, indent=4) + '\n')
                if ITERATE_SAME_TASK_COUNT < config.task_iterator.max_iterations:
                    # Update OMNI-EPIC config to iterate on the same task
                    config.success_detector.use_vision = False  # Only works without vision
                    config.override_vars['iterate_same_task'] = True
                    config.override_vars['task_description'] = dedent(ENV.__doc__).strip()
                    config.override_vars['task_envpath'] = ENV_PATH
                    ITERATE_SAME_TASK_COUNT += 1
                else:
                    ITERATE_SAME_TASK_COUNT = 0
            else:
                ITERATE_SAME_TASK_COUNT = 0

            # Generate next level
            config.logdir = LOGDIR
            config.robot = 'r2d2'
            config.iterate_until_success_gen = True
            config.enable_moi = True
            config.train_agent = False
            config.archive_from_ckpt = os.path.join(LOGDIR, 'archive.jsonl')
            config.add_examples = False
            config.task_generator.num_add_examples = 5
            config.task_iterator.num_examples = 5
            config.model_of_interestingness.num_examples = 5
            config.override_vars['taskgen_choose_probs'] = TASKGEN_CHOOSE_PROBS.tolist()
            omni_epic_vars = main(config)  # NOTE: comment this out to not gen things when debugging
            TASKGEN_CHOOSE_PROBS = omni_epic_vars['taskgen_choose_probs']

            # Reload archive and env
            ARCHIVE = load_archive(os.path.join(LOGDIR, 'archive.jsonl'))
            ARCHIVE_INDEX = len(ARCHIVE) - 1
    else:
        # Load the next env in the archive
        ARCHIVE_INDEX += 1

    # Load the new env
    ENV_PATH = ARCHIVE[ARCHIVE_INDEX]
    ENV.close()  # Close the current env
    ENV = PyBullet(env_path=ENV_PATH, vision=False)._env
    TASK_DESC = dedent(ENV.__doc__).strip().split('\n')[0]
    ENV.reset()
    SUCCESS = False
    socketio.emit('next_level_message', {'message': 'Next level started!'})
    socketio.emit('env_description', {'description': TASK_DESC})
    GENERATING_NEXT_LEVEL = False

@socketio.on('start_recording')
def start_recording():
    global IS_RECORDING, RECORDED_ACTIONS, RECORDED_FRAMES
    IS_RECORDING = True
    RECORDED_ACTIONS = []
    RECORDED_FRAMES = []
    logging.info("Started recording actions.")

@socketio.on('stop_recording')
def stop_recording():
    global IS_RECORDING, ENV_PATH, RECORDED_ACTIONS, RECOREDED_FRAMES
    IS_RECORDING = False
    data = {
        "env_filepath": ENV_PATH,
        "recorded_actions": RECORDED_ACTIONS,
        "recorded_frames": RECORDED_FRAMES,
    }
    with open(os.path.join(LOGDIR, 'recorded_actions.jsonl'), 'a') as f:
        json.dump(data, f)
        f.write('\n')
    logging.info("Stopped recording actions. Total actions recorded: {}. Data saved to 'recorded_actions.json'".format(len(RECORDED_ACTIONS)))
    socketio.emit('not_recording_status')

@socketio.on('mark_success')
def handle_mark_success():
    global SUCCESS
    if not check_valid_user():
        return
    SUCCESS = True
    socketio.emit('success_marked', {'message': 'Success marked!'})
    logging.info("Environment success marked as True.")

@socketio.on('mark_failure')
def handle_mark_failure():
    global SUCCESS
    if not check_valid_user():
        return
    SUCCESS = False
    socketio.emit('failure_marked', {'message': 'Failure marked!'})
    logging.info("Environment success marked as False.")

@socketio.on('access_code')
def handle_access(access_code):
    global ACCESS_USER_ID
    user_id = request.sid
    if (CURR_ACCESS_CODE != None and access_code == CURR_ACCESS_CODE) or access_code == SECRET_ACESS_CODE:
        ACCESS_USER_ID = user_id
        socketio.emit('access_granted', {'granted': True})
    else:
        socketio.emit('access_granted', {'granted': False})

# Global variable to store the booking status
booking_status = {
    "isBookingOngoing": False,
    "accessCode": ""
}

# URL of the HonoJS endpoint
HONOJS_URL = 'https://api.boopr.xyz/isBookingOngoing'
BEARER_TOKEN = 'api_OAWIJDIUWHJDAWHJIUDOahjwiudhDA9812738712iuahjdwiUDW*&E912hiu4hwtf9g0w78Y'

def fetch_booking_status():
    global booking_status, CURR_ACCESS_CODE, ACCESS_USER_ID
    headers = {
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }
    try:
        response = requests.get(HONOJS_URL, headers=headers)
        if response.status_code == 200:
            booking_status = response.json()
            # print(f"Updated booking status: {booking_status}")
            CURR_ACCESS_CODE = booking_status.get('accessCode')
            if ACCESS_USER_ID == None:
                socketio.emit('show_access_code', {'show': True, 'accessCode': CURR_ACCESS_CODE})
            else:
                socketio.emit('show_access_code', {'show': True, 'accessCode': CURR_ACCESS_CODE})
                # socketio.emit('show_access_code', {'show': False, 'accessCode': CURR_ACCESS_CODE})
        else:
            # print(f"Failed to fetch booking status. HTTP Status: {response.status_code}")
            pass
    except Exception as e:
        # print(f"Error fetching booking status: {e}")
        pass

def update_booking_status():
    global booking_status, CURR_ACCESS_CODE
    while True:
        CURR_ACCESS_CODE = None
        booking_status = {}
        fetch_booking_status()
        time.sleep(30)

@app.route('/check-access-code', methods=['POST'])
def check_access_code():
    code = request.json.get('accessCode')
    if code == booking_status.get('accessCode'):
        return jsonify({"message": "Access code is valid", "isBookingOngoing": booking_status.get('isBookingOngoing')})
    else:
        return jsonify({"message": "Access code is invalid", "isBookingOngoing": booking_status.get('isBookingOngoing')}), 403

@socketio.on('connect')
def handle_connect():
    socketio.emit('booking_status', booking_status)

@app.route('/test')
def test():
    return "Test route is working!"

if __name__ == "__main__":
    threading.Thread(target=update_booking_status, daemon=True).start()
    absl_app.run(lambda argv: socketio.run(app, host='0.0.0.0', port=3005))
