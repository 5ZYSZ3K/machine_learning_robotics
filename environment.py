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
# Make sure CARLA PythonAPI is on the path (tweak to where you put CARLA)
sys.path.append("E:/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner


SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 160
WIDTH = 240

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
        self.action_space = spaces.MultiDiscrete([9])

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

    def maintain_speed(self, s: float):
        """
        Very simple function to maintain desired speed.

        s arg is actual current speed in km/h.
        """
        if s >= self.PREFERRED_SPEED:
            return 0.0
        elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
            return 0.7  # think of it as % of "full gas"
        else:
            return 0.3  # tweak this if the car is way over or under preferred speed

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

        # MultiDiscrete([9]) -> take first component
        steer_idx = action[0] if isinstance(action, (list, np.ndarray)) else action
        steer = self._map_action_to_steer(steer_idx)

        # map throttle to maintain speed and apply steer and throttle
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        estimated_throttle = self.maintain_speed(kmh)
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=estimated_throttle, steer=steer, brake=0.0)
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
    def __init__(self, num_actions):
        super().__init__()

        # CNN for image
        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute conv output size lazily
        with torch.no_grad():
            dummy = torch.zeros(1, N_CHANNELS, HEIGHT, WIDTH)
            conv_out_size = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, image: torch.Tensor, angle: torch.Tensor):
        # image: (B, C, H, W), angle: (B, 1)
        x = self.cnn(image)
        x = torch.cat([x, angle], dim=1)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

    def act(self, obs, device):
        image = obs["image"]
        angle = obs["angle"]

        if isinstance(image, np.ndarray):
            image_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        else:
            raise TypeError("Expected image as numpy array")

        if isinstance(angle, np.ndarray):
            angle_t = torch.from_numpy(angle.astype(np.float32)).view(1, -1)
        else:
            angle_t = torch.tensor([[float(angle)]], dtype=torch.float32)

        image_t = image_t.to(device)
        angle_t = angle_t.to(device)

        logits, value = self.forward(image_t, angle_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)

        return (
            action.cpu().numpy()[0],
            logprob.detach().cpu().numpy()[0],
            value.detach().cpu().numpy()[0, 0],
        )

    def evaluate_actions(
        self, images: torch.Tensor, angles: torch.Tensor, actions: torch.Tensor
    ):
        logits, values = self.forward(images, angles)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprobs, torch.squeeze(values, -1), entropy