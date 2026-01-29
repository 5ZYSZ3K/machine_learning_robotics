import sys
import time
import math
import random
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import carla
import torch.nn.functional as F
# Make sure CARLA PythonAPI is on the path (tweak to where you put CARLA)
sys.path.append("E:/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner


SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 160
WIDTH = 240
N_ACTIONS_PER_DIM = 9
N_ACTION_DIMS = 3

FIXED_DELTA_SECONDS = 0.2

SHOW_PREVIEW = True

class CarEnvironment(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4
    PREFERRED_SPEED = 20
    SPEED_THRESHOLD = 2

    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()

        # Action space: 9 discrete steering values (mapped to [-0.9, 0.9])
        # We keep MultiDiscrete for compatibility with existing mapping
        self.action_space = spaces.MultiDiscrete([9, 3, 3])

        # Observation space: dict with image and angle to next waypoint
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(HEIGHT, WIDTH, N_CHANNELS),
                    dtype=np.float32,
                ),
                "angle": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = True
        self.settings.synchronous_mode = False
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.route = None

    def select_random_route(self):
        """
        Returns a random route for the car/vehicle
        where distance is longer than 100 waypoints.
        """
        point_a = self.vehicle.get_transform().location
        sampling_resolution = 1
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)

        min_distance = 100
        route_list = []
        for loc in self.world.get_map().get_spawn_points():
            cur_route = grp.trace_route(point_a, loc.location)
            if len(cur_route) > min_distance:
                route_list.append(cur_route)

        if not route_list:
            # Fallback: just use shortest route to any spawn point
            for loc in self.world.get_map().get_spawn_points():
                cur_route = grp.trace_route(point_a, loc.location)
                if cur_route:
                    route_list.append(cur_route)

        return random.choice(route_list)

    def get_closest_wp_forward(self):
        """
        Find closest waypoint looking forward.
        Returns (angle_normalized, distance).
        """
        points_ahead = []
        points_behind = []
        for i, wp in enumerate(self.route):
            vehicle_transform = self.vehicle.get_transform()
            wp_transform = wp[0].transform
            distance = (
                (wp_transform.location.y - vehicle_transform.location.y) ** 2
                + (wp_transform.location.x - vehicle_transform.location.x) ** 2
            ) ** 0.5
            angle = math.degrees(
                math.atan2(
                    wp_transform.location.y - vehicle_transform.location.y,
                    wp_transform.location.x - vehicle_transform.location.x,
                )
            ) - vehicle_transform.rotation.yaw

            if angle > 360:
                angle = angle - 360
            elif angle < -360:
                angle = angle + 360

            if angle > 180:
                angle = -360 + angle
            elif angle < -180:
                angle = 360 - angle

            if abs(angle) <= 90:
                points_ahead.append([i, distance, angle])
            else:
                points_behind.append([i, distance, angle])

        if len(points_ahead) == 0:
            closest = min(points_behind, key=lambda x: x[1])
            if closest[2] > 0:
                closest = [closest[0], closest[1], 90]
            else:
                closest = [closest[0], closest[1], -90]
        else:
            closest = min(points_ahead, key=lambda x: x[1])
            # move forward if too close
            for point in points_ahead:
                if 10 <= point[1] < 20:
                    closest = point
                    break

        return closest[2] / 90.0, closest[1]

    def cleanup(self):
        for sensor in self.world.get_actors().filter("*sensor*"):
            sensor.destroy()
        for actor in self.world.get_actors().filter("*vehicle*"):
            actor.destroy()
        cv2.destroyAllWindows()

    def _map_action_to_steer(self, action):
        # Mapping from discrete id to steering value
        mapping = {
            0: -0.9,
            1: -0.25,
            2: -0.1,
            3: -0.05,
            4: 0.0,
            5: 0.05,
            6: 0.1,
            7: 0.25,
            8: 0.9,
        }
        return mapping.get(int(action), 0.0)

    def step(self, action):
        self.step_counter += 1

        # Action can be:
        # - legacy: single steer_idx (int) or [steer_idx]
        # - extended: [steer_idx, throttle, brake]
        steer_idx = None
        throttle = None
        brake = None

        steer_idx = action[0]
        throttle = float(action[1])/3
        brake = float(action[2])/3

        # Map steering index to steering value
        steer = self._map_action_to_steer(steer_idx)

        # Apply control using steer, throttle and brake
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        )

        if self.step_counter % 50 == 0:
            print("steer input from model:", steer)

        distance_travelled = self.initial_location.distance(self.vehicle.get_location())
        step_distance_gain = 0.0
        if self.distance_travelled_last < distance_travelled:
            step_distance_gain = distance_travelled - self.distance_travelled_last
            self.distance_travelled_last = distance_travelled

        cam = self.front_camera
        if self.SHOW_CAM and cam is not None:
            cv2.imshow("Sem Camera", cam)
            cv2.waitKey(1)

        # get angle and distance to the navigation route
        angle, distance = None, None
        while angle is None:
            try:
                angle, distance = self.get_closest_wp_forward()
            except Exception:
                pass

        reward = 0.0
        terminated = False
        truncated = False

        # punish for collision
        if len(self.collision_hist) != 0:
            terminated = True
            reward -= 200.0
            self.cleanup()

        # punish for deviating from the route
        route_loss = distance - self.last_distance_to_route
        if route_loss < -0.1:
            reward += 10.0  # reward for getting closer to the route
        elif route_loss < 0.1:
            reward += 1.0
        else:
            reward -= 2.0
        if distance > 20:
            reward -= 100.0

        # reward for making distance
        reward += float(int(round(step_distance_gain * 3, 0)))

        # check for episode duration
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            truncated = True
            self.cleanup()

        obs = {"image": self.front_camera / 255.0, "angle": np.array([angle], dtype=np.float32)}
        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.collision_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except Exception:
                self.vehicle = None

        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()

        self.sem_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", "90")

        camera_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X)
        )
        self.sensor = self.world.spawn_actor(
            self.sem_cam, camera_init_trans, attach_to=self.vehicle
        )
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        if self.SHOW_CAM and self.front_camera is not None:
            cv2.namedWindow("Sem Camera", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Sem Camera", self.front_camera)
            cv2.waitKey(1)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            colsensor, camera_init_trans, attach_to=self.vehicle
        )
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None
        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.distance_travelled_last = 0.0
        self.route = self.select_random_route()
        angle, distance_to_route = self.get_closest_wp_forward()
        self.last_distance_to_route = distance_to_route

        obs = {
            "image": self.front_camera / 255.0,
            "angle": np.array([angle], dtype=np.float32),
        }
        return obs, {}

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]
        self.front_camera = i.astype(np.uint8)

    def collision_data(self, event):
        self.collision_hist.append(event)

class ActorCritic(nn.Module):
    def __init__(self, lr=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNELS, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # Spatial size after convs: (H,W) -> (39,59) -> (18,28) -> (16,26) for 160x240 input
        self.conv_out_size = 64 * 16 * 26  # 26624 for HEIGHT=160, WIDTH=240
        self.fc_features = nn.Linear(self.conv_out_size + 1, 256)

        self.policy_heads = nn.ModuleList([
            nn.Linear(256, N_ACTIONS_PER_DIM) for _ in range(N_ACTION_DIMS)
        ])
        self.value_head = nn.Linear(256, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _features(self, image, angle):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, angle], dim=1)
        x = F.relu(self.fc_features(x))
        return x

    def forward(self, image, angle):
        features = self._features(image, angle)
        logits = [head(features) for head in self.policy_heads]
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, image, angle, action=None):
        logits, value = self.forward(image, angle)
        dists = [torch.distributions.Categorical(logits=lg) for lg in logits]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=1)
        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        return action, log_prob, entropy, value

    def get_value(self, image, angle):
        _, value = self.forward(image, angle)
        return value