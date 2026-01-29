import random
import time
import numpy as np
import math 
import cv2
import gymnasium as gym
from gymnasium import spaces
import carla
import sys
sys.path.append('E:/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 180
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
	
	def __init__(self):
		super(CarEnvironment, self).__init__()

		self.action_space = spaces.MultiDiscrete([9, 9, 9])
		self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0,shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32),
            'angle': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })


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
		'''
		retruns a random route for the car/veh
		out of the list of possible locations locs
		where distance is longer than 100 waypoints
		'''    
		current_location = self.vehicle.get_transform().location 
		sampling_resolution = 1
		route_planner = GlobalRoutePlanner(self.world.get_map(), sampling_resolution)
		min_waypoints_distance = 100
		candidate_routes = []
		for spawn_point in self.world.get_map().get_spawn_points(): 
			route_candidate = route_planner.trace_route(current_location, spawn_point.location)
			if len(route_candidate) > min_waypoints_distance:
				candidate_routes.append(route_candidate)
		selected_route = random.choice(candidate_routes)
		return selected_route

	def get_closest_wp_forward(self):
		forward_waypoints = []
		behind_waypoints = []
		for waypoint_index, route_waypoint in enumerate(self.route):
			vehicle_transform = self.vehicle.get_transform()
			waypoint_transform = route_waypoint[0].transform
			distance_to_waypoint = ((waypoint_transform.location.y - vehicle_transform.location.y)**2 + (waypoint_transform.location.x - vehicle_transform.location.x)**2)**0.5
			angle_to_waypoint = math.degrees(math.atan2(waypoint_transform.location.y - vehicle_transform.location.y,
								waypoint_transform.location.x - vehicle_transform.location.x)) -  vehicle_transform.rotation.yaw
			if angle_to_waypoint>360:
				angle_to_waypoint = angle_to_waypoint - 360
			elif angle_to_waypoint <-360:
				angle_to_waypoint = angle_to_waypoint + 360

			if angle_to_waypoint>180:
				angle_to_waypoint = -360 + angle_to_waypoint
			elif angle_to_waypoint <-180:
				angle_to_waypoint = 360 - angle_to_waypoint 
			if abs(angle_to_waypoint)<=90:
				forward_waypoints.append([waypoint_index,distance_to_waypoint,angle_to_waypoint])
			else:
				behind_waypoints.append([waypoint_index,distance_to_waypoint,angle_to_waypoint])

		if len(forward_waypoints)==0:
			closest_waypoint = min(behind_waypoints, key=lambda x: x[1])
			if closest_waypoint[2]>0:
				closest_waypoint = [closest_waypoint[0],closest_waypoint[1],90]
			else:
				closest_waypoint = [closest_waypoint[0],closest_waypoint[1],-90] 
		else:
			closest_waypoint = min(forward_waypoints, key=lambda x: x[1])

			for waypoint_index, candidate_waypoint in enumerate(forward_waypoints):
				if candidate_waypoint[1]>=10 and candidate_waypoint[1]<20:
					closest_waypoint = candidate_waypoint
					break
			return closest_waypoint[2]/90.0, closest_waypoint[1] 
			
	def cleanup(self):
		for sensor in self.world.get_actors().filter('*sensor*'):
			sensor.destroy()
		for actor in self.world.get_actors().filter('*vehicle*'):
			actor.destroy()
		cv2.destroyAllWindows()
	
	def step(self, action):
		self.step_counter +=1
		steer = action[0]
		
		if steer ==0:
			steer = - 0.9
		elif steer ==1:
			steer = -0.25
		elif steer ==2:
			steer = -0.1
		elif steer ==3:
			steer = -0.05
		elif steer ==4:
			steer = 0.0 
		elif steer ==5:
			steer = 0.05
		elif steer ==6:
			steer = 0.1
		elif steer ==7:
			steer = 0.25
		elif steer ==8:
			steer = 0.9
		# map throttle to maintain speed and apply steer and throttle	

		throttle = action[1] / 9
		brake = action[2] / 9
		self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
		
		total_distance_travelled = self.initial_location.distance(self.vehicle.get_location())
		step_distance_gain = 0
		if self.distance_travelled_last < total_distance_travelled:
			step_distance_gain = total_distance_travelled - self.distance_travelled_last
			self.distance_travelled_last < total_distance_travelled

		camera_image = self.front_camera

		if self.SHOW_CAM:
			cv2.imshow('Sem Camera', camera_image)
			cv2.waitKey(1)
		
		angle_to_route, distance_to_route = None, None
		while angle_to_route is None:
			try:
				angle_to_route, distance_to_route = self.get_closest_wp_forward()
			except:
				pass
			
		reward = 0
		done = False
		if len(self.collision_hist) != 0:
			done = True
			reward = reward-200
			self.cleanup()
			
		# punish for deviating from the route
		route_loss =  distance_to_route - self.last_distance_to_route 
		if route_loss<-0.1:
			reward = reward + 10 #reward for getting closer to the route
		elif route_loss < 0.1:
			reward = reward + 1
		else:
			reward = reward - 2
		if distance_to_route > 20:
			reward = reward - 100

		reward = reward + int(round(step_distance_gain*3,0))
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True
			self.cleanup()
		return  {'image': self.front_camera/255.0, 'angle': angle_to_route}, reward, done, False, {}

	def reset(self, seed=None):
		self.collision_hist = []
		self.actor_list = []
		self.transform = random.choice(self.world.get_map().get_spawn_points())
		if seed:
			self.seed = seed
		self.vehicle = None
		while self.vehicle is None:
			try:
				self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
			except:
				pass
		self.actor_list.append(self.vehicle)
		self.initial_location = self.vehicle.get_location()
		self.semantic_segmentation_camera_blueprint = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.semantic_segmentation_camera_blueprint.set_attribute("image_size_x", f"{self.im_width}")
		self.semantic_segmentation_camera_blueprint.set_attribute("image_size_y", f"{self.im_height}")
		self.semantic_segmentation_camera_blueprint.set_attribute("fov", f"90")

		
		camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
		self.sensor = self.world.spawn_actor(self.semantic_segmentation_camera_blueprint, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(2)
		if self.SHOW_CAM:
			cv2.namedWindow('Sem Camera',cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Sem Camera', self.front_camera)
			cv2.waitKey(1)
		collision_sensor_blueprint = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(collision_sensor_blueprint, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		while self.front_camera is None:
			time.sleep(0.01)
		
		self.episode_start = time.time()
		self.steering_lock = False
		self.step_counter = 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		self.distance_travelled_last = 0
		self.route = self.select_random_route()
		angle_to_route, distance_to_route = self.get_closest_wp_forward()
		self.last_distance_to_route = distance_to_route
		return {'image': (self.front_camera/255.0), 'angle': angle_to_route}, {}

	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)
		image_array = np.array(image.raw_data)
		image_array = image_array.reshape((self.im_height, self.im_width, 4))[:, :, :3]
		self.front_camera = image_array

	def collision_data(self, collision_event):
		self.collision_hist.append(collision_event)
	