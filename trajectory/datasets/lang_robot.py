import sys
from os.path import dirname
#root_dir = "/home/guillefix/code/inria/captionRLenv/"
root_dir = "/gpfswork/rech/imi/usc19dv/captionRLenv/"
#DATA_ROOT_DIR = "/home/guillefix/code/inria/UR5_processed/"
DATA_ROOT_DIR = "/gpfsscratch/rech/imi/usc19dv/data/UR5_processed/"
sys.path.append(root_dir)

import pickle
import collections
from src.envs.envList import *
from src.envs.descriptions import generate_all_descriptions
from src.envs.env_params import get_env_params
import numpy as np
import pybullet as p
import pickle as pk

def add_xyz_rpy_controls(env):
    controls = []
    orn = env.instance.default_arm_orn_RPY
    controls.append(env.p.addUserDebugParameter("X", -1, 1, 0))
    controls.append(env.p.addUserDebugParameter("Y", -1, 1, 0.00))
    controls.append(env.p.addUserDebugParameter("Z", -1, 1, 0.2))
    controls.append(env.p.addUserDebugParameter("R", -4, 4, orn[0]))
    controls.append(env.p.addUserDebugParameter("P", -4, 4, orn[1]))
    controls.append(env.p.addUserDebugParameter("Y", -4,4, orn[2]))
    controls.append(env.p.addUserDebugParameter("grip", env.action_space.low[-1], env.action_space.high[-1], 0))
    return controls

def add_joint_controls(env):
    for i, obj in enumerate(env.instance.restJointPositions):
        env.p.addUserDebugParameter(str(i), -2*np.pi, 2*np.pi, obj)

#env_name='LangRobot-v0'
env_name='LangRobot-v1'

def build_env_name(task):
    """Construct the env name from parameters."""
    return env_name

import gym
from gym import spaces
from gym.envs import registration
import numpy as np

class LangRobotEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LangRobotEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=-10, high=10, shape=(125+384,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(low=-10, high=10, shape=(125,), dtype=np.float32),
            'annotation_emb': spaces.Box(low=-10, high=10, shape=(384,), dtype=np.float32)
        })

        self.joint_control = False # Toggle this flag to control joints or ABS RPY Space
        self.env = env = UR5PlayAbsRPY1Obj()
        self.annotation_emb = None

        self.ex_data = None

    def step(self, action):
        # Execute one time step within the environment
        env = self.env
        if self.joint_control:
            poses  = []
            for i in range(len(env.instance.restJointPositions)):
                poses.append(env.p.readUserDebugParameter(i))
            # Uses a hard reset of the arm joints so that we can quickly debug without worrying about forces
            env.instance.reset_arm_joints(env.instance.arm, poses)

        else:
            # print(action)
            # state = env.instance.calc_actor_state()
            obs, r, done, info = env.step(np.array(action))

        observation = collections.OrderedDict(
            obs=obs,
            annotation_emb=self.annotation_emb
        )

        return observation, r, done, info
    def reset(self):
        # Reset the state of the environment to an initial state
        env = self.env
        object_types = pickle.load(open(root_dir+"object_types.pkl","rb"))
        env.env_params['types'] = object_types

        env.render(mode='human')
        env.reset(o=self.ex_data["obs"][0], info_reset=None, description=self.ex_data["goal_str"][0], joint_poses=self.ex_data["joint_poses"][0], objects=self.ex_data['obj_stuff'][0])
        print([o for o in env.instance.objects])
        if self.joint_control:
            add_joint_controls(env)
        else:
            self.controls = add_xyz_rpy_controls(env)
        return self.observation_space.sample()
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    def get_metrics(self, num_episodes):
        metrics = []
        success_metric = None
        return metrics, success_metric

    def get_dataset(self):
        dataset = {}
        root_folder = DATA_ROOT_DIR
        #obs_all = np.load(root_folder+"obs_all_augmented.npy")
        #acts_all = np.load(root_folder+"acts_all.npy")
        #terminals_all = np.load(root_folder+"terminals_all.npy")
        #disc_cond_all = np.load(root_folder+"disc_cond_all.npy")

        obs_all = np.load(root_folder+"obs_all_augmented_smol.npy")
        acts_all = np.load(root_folder+"acts_all_smol.npy")
        terminals_all = np.load(root_folder+"terminals_all_smol.npy")
        disc_cond_all = np.load(root_folder+"disc_cond_all_smol.npy")
        n=acts_all.shape[0]
        dataset['actions'] = acts_all
        dataset['discrete_conds'] = disc_cond_all
        dataset['observations'] = obs_all
        dataset['rewards'] = np.zeros((n,))
        dataset['terminals'] = np.full((n,),False)
        dataset['timeouts'] = terminals_all
        return dataset

if env_name in registration.registry.env_specs:
    del registration.registry.env_specs[env_name]
registration.register(
    id=env_name,
    entry_point=LangRobotEnv,
    max_episode_steps=1000)
