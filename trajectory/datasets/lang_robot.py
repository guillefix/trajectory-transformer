import sys
from constants import *
sys.path.append(root_folder)

from src.envs.envList import ExtendedUR5PlayAbsRPY1Obj

#env_name='LangRobot-v0'
env_name='LangRobot-v1'

def build_env_name(task):
    """Construct the env name from parameters."""
    return env_name

import gym
from gym import spaces
from gym.envs import registration
import numpy as np

import pickle
obs_mod = "obs_cont_single_nocol_noarm_trim_scaled"
acts_mod = "acts_trim_scaled"
obs_scaler = pickle.load(open(processed_data_folder+obs_mod+"_scaler.pkl", "rb"))
acts_scaler = pickle.load(open(processed_data_folder+acts_mod+"_scaler.pkl", "rb"))

class LangRobotEnv(ExtendedUR5PlayAbsRPY1Obj):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LangRobotEnv, self).__init__(simple_obs=True, obs_scaler=obs_scaler, acts_scaler=acts_scaler, desc_max_len=10, obs_mod="obs_cont_single_nocol_noarm_trim_scaled")

    def get_metrics(self, num_episodes):
        metrics = []
        success_metric = None
        return metrics, success_metric

    def _get_lang_goal(self):
        return self.tokens[:,0]

    def reset(self, goal_str=None):
        return super().reset(description=goal_str)

    lang_goal = property(fget=_get_lang_goal)

    def get_dataset(self):
        dataset = {}
        root_folder = processed_data_folder

        #obs_all = np.load(root_folder+"obs_all_augmented_smol.npy")
        #acts_all = np.load(root_folder+"acts_all_smol.npy")
        #terminals_all = np.load(root_folder+"terminals_all_smol.npy")
        #disc_cond_all = np.load(root_folder+"disc_cond_all_smol.npy")

        obs_all = np.load(root_folder+"obs_all_augmented.npy")
        acts_all = np.load(root_folder+"acts_all.npy")
        terminals_all = np.load(root_folder+"terminals_all.npy")
        disc_cond_all = np.load(root_folder+"disc_cond_all.npy")
        rewards_all = np.load(root_folder+"rewards_all.npy")
        n=acts_all.shape[0]
        dataset['actions'] = acts_all
        dataset['discrete_conds'] = disc_cond_all
        dataset['observations'] = obs_all
        dataset['rewards'] = rewards_all
        dataset['terminals'] = np.full((n,),False)
        dataset['timeouts'] = terminals_all
        return dataset

if env_name in registration.registry.env_specs:
    del registration.registry.env_specs[env_name]
registration.register(
    id=env_name,
    entry_point=LangRobotEnv,
    max_episode_steps=1000)
