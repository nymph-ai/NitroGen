import threading
import time
from typing import Optional

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from PIL import Image as PILImage

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Joy


class _RosBridge(Node):
    def __init__(self, image_topic: str, joy_topic: str, frame_id: str):
        super().__init__("nitrogen_gamepad_env")
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._latest_image: Optional[PILImage.Image] = None
        self._latest_lock = threading.Lock()
        self._image_topic = image_topic
        self._joy_topic = joy_topic
        self._frame_id = frame_id

        self._image_sub = self.create_subscription(
            Image, self._image_topic, self._image_callback, qos_profile
        )
        self._joy_pub = self.create_publisher(
            Joy,
            self._joy_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

    def _image_callback(self, msg: Image) -> None:
        pil_image = self._convert_image(msg)
        if pil_image is None:
            return
        with self._latest_lock:
            self._latest_image = pil_image

    def _convert_image(self, msg: Image) -> Optional[PILImage.Image]:
        if msg.encoding == "jpeg":
            import io

            data = bytes(msg.data)
            return PILImage.open(io.BytesIO(data))

        width = msg.width
        height = msg.height
        step = msg.step if msg.step > 0 else width * 3
        if msg.encoding in ("rgb8", "bgr8"):
            if isinstance(msg.data, list):
                img_data = np.array(msg.data, dtype=np.uint8)
            else:
                img_data = np.frombuffer(msg.data, dtype=np.uint8)

            if step == width * 3:
                img_array = img_data.reshape((height, width, 3))
            else:
                img_array = img_data.reshape((height, step))
                img_array = img_array[:, : width * 3].reshape((height, width, 3))

            if msg.encoding == "bgr8":
                img_array = img_array[:, :, ::-1]
            return PILImage.fromarray(img_array, mode="RGB")

        self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
        return None

    def get_latest_image(self) -> Optional[PILImage.Image]:
        with self._latest_lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

    def publish_joy(self, axes: list, buttons: list) -> None:
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.axes = [float(x) for x in axes]
        msg.buttons = [int(x) for x in buttons]
        self._joy_pub.publish(msg)


class GamepadEnv(Env):
    """
    ROS2-backed environment controlled with a virtual gamepad.

    Observations come from a ROS2 image topic (e.g. /player/image_raw).
    Actions are published as sensor_msgs/Joy to a joystick topic.
    """

    def __init__(
        self,
        game,
        image_height=1440,
        image_width=2560,
        controller_type="xbox",
        game_speed=1.0,
        env_fps=10,
        async_mode=True,
        image_topic="/player/image_raw",
        joy_topic="/joy",
        frame_id="nitrogen",
        image_timeout=2.0,
        trigger_mode="0_1",
    ):
        super().__init__()

        if controller_type not in ["xbox", "ps4"]:
            raise ValueError("controller_type must be either 'xbox' or 'ps4'")

        self.game = game
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.game_speed = game_speed
        self.env_fps = env_fps
        self.step_duration = self.calculate_step_duration()
        self.async_mode = async_mode
        self.image_topic = image_topic
        self.joy_topic = joy_topic
        self.frame_id = frame_id
        self.image_timeout = float(image_timeout)
        self.trigger_mode = trigger_mode

        self.observation_space = Box(
            low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype="uint8"
        )

        self.action_space = Dict(
            {
                "BACK": Discrete(2),
                "GUIDE": Discrete(2),
                "RIGHT_SHOULDER": Discrete(2),
                "RIGHT_TRIGGER": Box(low=0.0, high=1.0, shape=(1,)),
                "LEFT_TRIGGER": Box(low=0.0, high=1.0, shape=(1,)),
                "LEFT_SHOULDER": Discrete(2),
                "AXIS_RIGHTX": Box(low=-32768.0, high=32767, shape=(1,)),
                "AXIS_RIGHTY": Box(low=-32768.0, high=32767, shape=(1,)),
                "AXIS_LEFTX": Box(low=-32768.0, high=32767, shape=(1,)),
                "AXIS_LEFTY": Box(low=-32768.0, high=32767, shape=(1,)),
                "LEFT_THUMB": Discrete(2),
                "RIGHT_THUMB": Discrete(2),
                "DPAD_UP": Discrete(2),
                "DPAD_RIGHT": Discrete(2),
                "DPAD_DOWN": Discrete(2),
                "DPAD_LEFT": Discrete(2),
                "WEST": Discrete(2),
                "SOUTH": Discrete(2),
                "EAST": Discrete(2),
                "NORTH": Discrete(2),
                "START": Discrete(2),
            }
        )

        # Canonical Xbox-style layout for Joy buttons.
        self._button_order = [
            "SOUTH",  # A
            "EAST",   # B
            "WEST",   # X
            "NORTH",  # Y
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "BACK",
            "START",
            "GUIDE",
            "LEFT_THUMB",
            "RIGHT_THUMB",
            "DPAD_UP",
            "DPAD_RIGHT",
            "DPAD_DOWN",
            "DPAD_LEFT",
        ]
        self._axis_order = [
            "AXIS_LEFTX",
            "AXIS_LEFTY",
            "AXIS_RIGHTX",
            "AXIS_RIGHTY",
            "LEFT_TRIGGER",
            "RIGHT_TRIGGER",
        ]

        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self._node = _RosBridge(self.image_topic, self.joy_topic, self.frame_id)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    def calculate_step_duration(self):
        return 1.0 / (self.env_fps * self.game_speed)

    def _normalize_axis(self, value) -> float:
        if isinstance(value, np.ndarray):
            value = float(value.flatten()[0])
        else:
            value = float(value)
        if abs(value) <= 1.0:
            return max(-1.0, min(1.0, value))
        return max(-1.0, min(1.0, value / 32767.0))

    def _normalize_trigger(self, value) -> float:
        if isinstance(value, np.ndarray):
            value = float(value.flatten()[0])
        else:
            value = float(value)
        if value > 1.0:
            value = value / 255.0
        value = max(0.0, min(1.0, value))
        if self.trigger_mode == "neg1_1":
            return value * 2.0 - 1.0
        return value

    def _publish_action(self, action) -> None:
        axes = []
        for axis_name in self._axis_order:
            if "TRIGGER" in axis_name:
                axes.append(self._normalize_trigger(action.get(axis_name, 0.0)))
            else:
                axes.append(self._normalize_axis(action.get(axis_name, 0.0)))

        buttons = [int(bool(action.get(name, 0))) for name in self._button_order]
        self._node.publish_joy(axes, buttons)

    def perform_action(self, action, duration):
        self._publish_action(action)
        if self.async_mode:
            time.sleep(duration)
        else:
            time.sleep(duration)

    def step(self, action, step_duration=None):
        duration = step_duration if step_duration is not None else self.step_duration
        self.perform_action(action, duration)

        obs = self.render()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return None

    def close(self):
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def pause(self):
        pass

    def unpause(self):
        pass

    def _wait_for_image(self) -> Optional[PILImage.Image]:
        start = time.time()
        while True:
            image = self._node.get_latest_image()
            if image is not None:
                return image
            if time.time() - start >= self.image_timeout:
                return None
            time.sleep(0.01)

    def render(self):
        image = self._node.get_latest_image()
        if image is None:
            image = self._wait_for_image()

        if image is None:
            return PILImage.new("RGB", (self.image_width, self.image_height), (0, 0, 0))

        return image.resize((self.image_width, self.image_height))
