import datetime
from environment import CarEnvironment
from stable_baselines3 import PPO #PPO
import os


models_directory = f"models/{datetime.datetime.now().strftime("%I-%M%p-%B-%d-%Y")}/"
logs_directory = f"logs/{datetime.datetime.now().strftime("%I-%M%p-%B-%d-%Y")}/"

if not os.path.exists(models_directory):
	os.makedirs(models_directory)

if not os.path.exists(logs_directory):
	os.makedirs(logs_directory)

print('connecting to env..')

environment = CarEnvironment()
seed = 2025 # Ensure replicability
environment.reset(seed=seed)
model = PPO('MultiInputPolicy', environment, verbose=True, learning_rate=0.001, tensorboard_log=logs_directory)

TIMESTEPS_PER_ITERATION = 100
training_iteration = 0
while training_iteration<4: 
	training_iteration += 1
	print('Iteration ', training_iteration,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS_PER_ITERATION, reset_num_timesteps=False, tb_log_name=f"PPO" )
	print('Iteration ', training_iteration,' has been trained')
	model.save(f"{models_directory}/{TIMESTEPS_PER_ITERATION*training_iteration}")