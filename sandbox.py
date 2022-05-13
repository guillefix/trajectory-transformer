import sys
root_dir = "/home/guillefix/code/inria/captionRLenv/"
sys.path.append(root_dir)
env_name = "halfcheetah-medium-expert-v2"

from trajectory.datasets.d4rl import load_environment

import gym

#%%

wrapped_env = gym.make(env_name)

env = wrapped_env.unwrapped

d=env.get_dataset()

type(d)
d.keys()
d['actions'].shape
d['observations'].shape
d['next_observations'].shape
d['rewards'].shape
d['timeouts'].shape
d['terminals'].shape
d['infos/action_log_probs'].shape
d['infos/qpos'].shape
d['infos/qvel'].shape

d['observations'][1]
d['next_observations'][0]
d
#%%


root_folder = "/home/guillefix/code/inria/UR5_processed/"
filenames=[x[:-1] for x in open("/home/guillefix/code/inria/UR5_processed/base_filenames.txt","r").readlines()]
len(filenames)
get_ann = lambda x: open(root_folder+x+".annotation.txt","r").read()
filenames_filtered = [x for x in filenames if get_ann(x).split(" ")[0] == "Paint"]
# with open(root_folder+"base_filenames_paint.txt", "w") as f:
#     f.writelines([x+"\n" for x in filenames_filtered])
len(filenames_filtered)
filenames = filenames_filtered
# filenames = ["UR5_Guillermo_obs_act_etc_8_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_16_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_22_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_26_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_27_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_28_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_29_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_30_data"]
# filenames += ["UR5_Guillermo_obs_act_etc_31_data"]

filename = filenames[0]
# obss.shape
import numpy as np
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

obs_all = None
acts_all = None
terminals_all = None
disc_cond_all = None
import pickle
object_types = pickle.load(open("/home/guillefix/code/inria/captionRLenv/object_types.pkl","rb"))
# object_types

import json
vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json","r").read())
len(vocab)

from src.envs.color_generation import infer_color

def one_hot(x,n):
    a = np.zeros(n)
    a[x]=1
    return a

color_list = ['yellow', 'magenta', 'blue', 'green', 'red', 'cyan', 'black', 'white']

for filename in filenames:
    obss = np.load(root_folder+filename+".npz.obs.npy")
    actss = np.load(root_folder+filename+".npz.acts.npy")
    ann = np.load(root_folder+filename+".annotation.npy")
    # obss_cont = np.concatenate([obss[:,:14], obss[:,37:49], obss[:,72:84], obss[:,107:]],axis=1)
    # obss[:,14:37][3]
    obss_color1 = obss[:,37:40]
    obss_color2 = obss[:,72:75]
    obss_color3 = obss[:,107:110]
    # obss_color1 = np.array([vocab[infer_color(x)] for x in obss_color1])
    # obss_color2 = np.array([vocab[infer_color(x)] for x in obss_color2])
    # obss_color3 = np.array([vocab[infer_color(x)] for x in obss_color3])
    n=len(color_list)
    # color_list.index('cyan')
    obss_color1 = np.stack([one_hot(color_list.index(infer_color(x)),n) for x in obss_color1])
    obss_color2 = np.stack([one_hot(color_list.index(infer_color(x)),n) for x in obss_color2])
    obss_color3 = np.stack([one_hot(color_list.index(infer_color(x)),n) for x in obss_color3])
    obss_disc1 = np.argmax(obss[:,14:37], axis=1)
    obss_disc2 = np.argmax(obss[:,49:72], axis=1)
    obss_disc3 = np.argmax(obss[:,84:107], axis=1)
    obss_cont = np.concatenate([obss[:,:14], obss_color1, obss[:,40:49], obss_color2, obss[:,75:84], obss_color3, obss[:,110:]],axis=1)
    np.save(root_folder+filename+".obs_cont.npy", obss_cont)
    obss_cont.shape

    assert np.all(obss_disc1 == obss_disc1[0])
    assert np.all(obss_disc2 == obss_disc2[0])
    assert np.all(obss_disc3 == obss_disc3[0])
    # assert np.all(obss_color1 == obss_color1[0])
    # assert np.all(obss_color2 == obss_color2[0])
    # assert np.all(obss_color3 == obss_color3[0])
    obss_disc1 = obss_disc1[0]
    obss_disc2 = obss_disc2[0]
    obss_disc3 = obss_disc3[0]
    # obss_color1 = obss_color1[0]
    # obss_color2 = obss_color2[0]
    # obss_color3 = obss_color3[0]

    obss_disc1 = vocab[object_types[obss_disc1]]
    obss_disc2 = vocab[object_types[obss_disc2]]
    obss_disc3 = vocab[object_types[obss_disc3]]

    ann
    # disc_cond = np.concatenate([ann, [obss_disc1, obss_color1, obss_disc2, obss_color2, obss_disc3, obss_color3]])
    disc_cond = np.concatenate([ann, [obss_disc1, obss_disc2, obss_disc3]])
    np.save(root_folder+filename+".disc_cond.npy", disc_cond)

    # with open(root_folder+filename+".annotation.txt","r") as f:
    #     annotation = f.read()
    # ann_emb = model.encode(annotation)
    # obss2 = np.empty((obss.shape[0], obss.shape[1]+ann_emb.shape[0]))
    terminals = np.empty((obss.shape[0],))
    for i,obs in enumerate(obss):
        # obss2[i] = np.concatenate([obs,ann_emb])
        terminals[i] = False
    terminals[-1] = True

    if obs_all is None:
        # obs_all = obss2
        obs_all = obss_cont
    else:
        # obs_all = np.concatenate([obs_all, obss2])
        obs_all = np.concatenate([obs_all, obss_cont])

    if acts_all is None:
        acts_all = actss
    else:
        acts_all = np.concatenate([acts_all, actss])

    if terminals_all is None:
        terminals_all = terminals
    else:
        terminals_all = np.concatenate([terminals_all, terminals])

    if disc_cond_all is None:
        disc_cond_all = np.expand_dims(disc_cond,0)
    else:
        disc_cond_all = np.concatenate([disc_cond_all, np.expand_dims(disc_cond,0)])

obs_all.shape
acts_all.shape
terminals_all.shape
disc_cond_all.shape

# np.save(root_folder+"obs_all_augmented.npy", obs_all)
# np.save(root_folder+"acts_all.npy", acts_all)
# np.save(root_folder+"terminals_all.npy", terminals_all)
# np.save(root_folder+"disc_cond_all.npy", disc_cond_all)

np.save(root_folder+"obs_all_augmented_smol.npy", obs_all)
np.save(root_folder+"acts_all_smol.npy", acts_all)
np.save(root_folder+"terminals_all_smol.npy", terminals_all)
np.save(root_folder+"disc_cond_all_smol.npy", disc_cond_all)
