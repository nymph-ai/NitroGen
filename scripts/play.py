import os
import sys
import time
import json
from pathlib import Path
from collections import OrderedDict
import math

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
parser.add_argument("--process", type=str, default="minecraft", help="Game label (unused for ROS2)")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--image-topic", type=str, default="/player/image_raw", help="ROS2 image topic")
parser.add_argument("--joy-topic", type=str, default="/joy", help="ROS2 joy topic")
parser.add_argument("--trigger-mode", type=str, default="0_1", help="Trigger axis mode: 0_1 or neg1_1")
parser.add_argument("--debug-output", action="store_true", help="Write debug frames/videos/actions to disk")
parser.add_argument("--stick-deadzone", type=float, default=0.15, help="Radial deadzone for sticks")
parser.add_argument("--stick-expo", type=float, default=0.3, help="Expo curve for sticks")
parser.add_argument("--log-actions", action="store_true", help="Log stick/button stats periodically")
parser.add_argument("--log-every", type=int, default=60, help="Steps between action logs")

args = parser.parse_args()

policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

DEBUG_OUTPUT = args.debug_output
PATH_DEBUG = None
PATH_OUT = None

if DEBUG_OUTPUT:
    PATH_DEBUG = PATH_REPO / "debug"
    PATH_DEBUG.mkdir(parents=True, exist_ok=True)

    PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
    PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5
STICK_DEADZONE = float(args.stick_deadzone)
STICK_EXPO = float(args.stick_expo)

# Find in path_out the list of existing video files, named 0001.mp4, 0002.mp4, etc.
# If they exist, find the max number and set the next number to be max + 1
PATH_MP4_DEBUG = None
PATH_MP4_CLEAN = None
PATH_ACTIONS = None
if DEBUG_OUTPUT:
    video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
    if video_files:
        existing_numbers = [f.name.split("_")[0] for f in video_files]
        existing_numbers = [int(n) for n in existing_numbers if n.isdigit()]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    PATH_MP4_DEBUG = PATH_OUT / f"{next_number:04d}_DEBUG.mp4"
    PATH_MP4_CLEAN = PATH_OUT / f"{next_number:04d}_CLEAN.mp4"
    PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.json"

class NullRecorder:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_frame(self, _frame):
        return None

def preprocess_img(main_image):
    main_cv = cv2.cvtColor(np.array(main_image), cv2.COLOR_RGB2BGR)
    final_image = cv2.resize(main_cv, (256, 256), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

zero_action = OrderedDict(
    [
        ("WEST", 0),
        ("SOUTH", 0),
        ("BACK", 0),
        ("DPAD_DOWN", 0),
        ("DPAD_LEFT", 0),
        ("DPAD_RIGHT", 0),
        ("DPAD_UP", 0),
        ("GUIDE", 0),
        ("AXIS_LEFTX", 0.0),
        ("AXIS_LEFTY", 0.0),
        ("LEFT_SHOULDER", 0),
        ("LEFT_TRIGGER", 0.0),
        ("AXIS_RIGHTX", 0.0),
        ("AXIS_RIGHTY", 0.0),
        ("LEFT_THUMB", 0),
        ("RIGHT_THUMB", 0),
        ("RIGHT_SHOULDER", 0),
        ("RIGHT_TRIGGER", 0.0),
        ("START", 0),
        ("EAST", 0),
        ("NORTH", 0),
    ]
)

def apply_expo(value, expo):
    return (1.0 - expo) * value + expo * (value ** 3)

def apply_radial_deadzone(x, y, deadzone):
    mag = math.hypot(x, y)
    if mag <= deadzone:
        return 0.0, 0.0
    scaled = (mag - deadzone) / (1.0 - deadzone)
    scale = scaled / mag
    return x * scale, y * scale

def process_stick(x, y):
    x, y = apply_radial_deadzone(float(x), float(y), STICK_DEADZONE)
    x = apply_expo(x, STICK_EXPO)
    y = apply_expo(y, STICK_EXPO)
    return max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))

TOKEN_SET = BUTTON_ACTION_TOKENS

print("Model loaded, starting environment...")
for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

env = GamepadEnv(
    game=args.process,
    game_speed=1.0,
    env_fps=60,
    async_mode=True,
    image_topic=args.image_topic,
    joy_topic=args.joy_topic,
    trigger_mode=args.trigger_mode,
)

env.reset()
env.pause()


# Initial call to get state
obs, reward, terminated, truncated, info = env.step(action=zero_action)

frames = None
step_count = 0

debug_recorder_ctx = (
    VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="medium")
    if DEBUG_OUTPUT
    else NullRecorder()
)
clean_recorder_ctx = (
    VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="medium")
    if DEBUG_OUTPUT
    else NullRecorder()
)

with debug_recorder_ctx as debug_recorder:
    with clean_recorder_ctx as clean_recorder:
        try:
            while True:
                obs = preprocess_img(obs)
                if DEBUG_OUTPUT:
                    obs.save(PATH_DEBUG / f"{step_count:05d}.png")

                pred = policy.predict(obs)

                j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]

                n = len(buttons)
                assert n == len(j_left) == len(j_right), "Mismatch in action lengths"

                if args.log_actions and step_count % max(1, args.log_every) == 0:
                    left_arr = np.array(j_left, dtype=np.float32)
                    right_arr = np.array(j_right, dtype=np.float32)
                    print(
                        "Stick stats: "
                        f"L(min={left_arr.min():.3f}, max={left_arr.max():.3f}, mean={left_arr.mean():.3f}) "
                        f"R(min={right_arr.min():.3f}, max={right_arr.max():.3f}, mean={right_arr.mean():.3f})"
                    )

                env_actions = []

                for i in range(n):
                    move_action = zero_action.copy()

                    xl, yl = j_left[i]
                    xr, yr = j_right[i]
                    xl, yl = process_stick(xl, yl)
                    xr, yr = process_stick(xr, yr)
                    move_action["AXIS_LEFTX"] = xl
                    move_action["AXIS_LEFTY"] = yl
                    move_action["AXIS_RIGHTX"] = xr
                    move_action["AXIS_RIGHTY"] = yr
                    
                    button_vector = buttons[i]
                    assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"

                    
                    for name, value in zip(TOKEN_SET, button_vector):
                        if "TRIGGER" in name:
                            move_action[name] = float(value)
                        else:
                            move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0


                    env_actions.append(move_action)

                print(f"Executing {len(env_actions)} actions, each action will be repeated {action_downsample_ratio} times")

                for i, a in enumerate(env_actions):
                    if NO_MENU:
                        if a["START"]:
                            print("Model predicted start, disabling this action")
                        a["GUIDE"] = 0
                        a["START"] = 0
                        a["BACK"] = 0

                    for _ in range(action_downsample_ratio):
                        obs, reward, terminated, truncated, info = env.step(action=a)

                        # resize obs to 720p
                        obs_viz = np.array(obs).copy()
                        clean_viz = cv2.resize(obs_viz, (1920, 1080), interpolation=cv2.INTER_AREA)
                        debug_viz = create_viz(
                            cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_AREA), # 720p
                            i,
                            j_left,
                            j_right,
                            buttons,
                            token_set=TOKEN_SET
                        )
                        debug_recorder.add_frame(debug_viz)
                        clean_recorder.add_frame(clean_viz)

                # Append env_actions dictionnary to JSONL file
                if DEBUG_OUTPUT:
                    with open(PATH_ACTIONS, "a") as f:
                        for i, a in enumerate(env_actions):
                            # convert numpy arrays to lists for JSON serialization
                            for k, v in a.items():
                                if isinstance(v, np.ndarray):
                                    a[k] = v.tolist()
                            a["step"] = step_count
                            a["substep"] = i
                            json.dump(a, f)
                            f.write("\n")


                step_count += 1
        finally:
            env.unpause()
            env.close()
