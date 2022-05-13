import sys
from os.path import dirname
root_dir = "/home/guillefix/code/inria/captionRLenv/"
DATA_ROOT_DIR = "/home/guillefix/code/inria/UR5_processed/"
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

def build_env_name(task):
    """Construct the env name from parameters."""
    return "LangRobot-v0"

import gym
from gym import spaces
from gym.envs import registration
import numpy as np

from src.envs.color_generation import infer_color

def one_hot(x,n):
    a = np.zeros(n)
    a[x]=1
    return a

color_list = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']

def fix_quaternions(rot_stream):
    prev_rot = None
    for i, rot in enumerate(rot_stream):
        if prev_rot is None:
            prev_rot = rot
        if np.any(np.abs(rot-prev_rot) >= prev_rot):
            rot_stream[i:] = -rot_stream[i:]

    return rot_stream

def process_obs(obs):
    obs_color1 = obs[37:40].astype(np.float64)
    obs_color2 = obs[72:75].astype(np.float64)
    obs_color3 = obs[107:110].astype(np.float64)
    n=len(color_list)
    obs_color1 = one_hot(color_list.index(infer_color(obs_color1)),n)
    obs_color2 = one_hot(color_list.index(infer_color(obs_color2)),n)
    obs_color3 = one_hot(color_list.index(infer_color(obs_color3)),n)
    obs_cont = np.concatenate([obs[:14], obs_color1, obs[40:49], obs_color2, obs[75:84], obs_color3, obs[110:]])
    obs_cont[3:7] = fix_quaternions(obs_cont[3:7])
    return obs_cont


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
        self.observation_space = spaces.Box(low=-10, high=10, shape=(71,), dtype=np.float32)

        self.joint_control = False # Toggle this flag to control joints or ABS RPY Space
        self.env = env = UR5PlayAbsRPY1Obj()
        # self.annotation_emb = None

        human_data = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/8/data.npz", allow_pickle=True)
        self.ex_data = human_data

        goal_str = human_data['goal_str'][0]

        import json

        d=json.load(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json", "r"))

        tokens = []
        words = goal_str.split(" ")
        for i in range(11):
            if i < len(words):
                word = words[i]
                tokens.append(d[word])
            else:
                tokens.append(66)

        import pickle
        object_types = pickle.load(open("/home/guillefix/code/inria/captionRLenv/object_types.pkl","rb"))
        # object_types

        import json
        a = human_data
        vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json","r").read())
        obss_disc1 = np.argmax(a['obs'][:,14:37], axis=1)
        obss_disc2 = np.argmax(a['obs'][:,49:72], axis=1)
        obss_disc3 = np.argmax(a['obs'][:,84:107], axis=1)
        assert np.all(obss_disc1 == obss_disc1[0])
        assert np.all(obss_disc2 == obss_disc2[0])
        assert np.all(obss_disc3 == obss_disc3[0])
        obss_disc1 = obss_disc1[0]
        obss_disc2 = obss_disc2[0]
        obss_disc3 = obss_disc3[0]
        obss_disc1 = vocab[object_types[obss_disc1]]
        obss_disc2 = vocab[object_types[obss_disc2]]
        obss_disc3 = vocab[object_types[obss_disc3]]

        tokens += [obss_disc1, obss_disc2, obss_disc3]

        self.lang_goal = np.array(tokens)
        # self.ex_data = None

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
            print(action.shape)
            acts = action.tolist()
            action = [acts[0],acts[1],acts[2]] + list(p.getEulerFromQuaternion(acts[3:7])) + [acts[7]]
            obs, r, done, info = env.step(np.array(action))

        obs = process_obs(obs)
        return obs, r, done, info
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
        # obs = process_obs(obs)
        # return obs
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
        obs_all = np.load(root_folder+"obs_all_augmented.npy")
        acts_all = np.load(root_folder+"acts_all.npy")
        terminals_all = np.load(root_folder+"terminals_all.npy")
        disc_cond_all = np.load(root_folder+"disc_cond_all.npy")
        n=acts_all.shape[0]
        dataset['actions'] = acts_all
        dataset['discrete_conds'] = disc_cond_all
        dataset['observations'] = obs_all
        dataset['rewards'] = np.zeros((n,))
        dataset['terminals'] = np.full((n,),False)
        dataset['timeouts'] = terminals_all
        return dataset

if 'LangRobot-v0' in registration.registry.env_specs:
    del registration.registry.env_specs['LangRobot-v0']
registration.register(
    id='LangRobot-v0',
    entry_point=LangRobotEnv,
    max_episode_steps=1000)
