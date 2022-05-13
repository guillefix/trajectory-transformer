env_name = "halfcheetah-medium-expert-v2"

from trajectory.datasets.d4rl import load_environment

import gym

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


root_folder = "/home/guillefix/code/inria/UR5_processed/"
filenames=[x[:-1] for x in open("/home/guillefix/code/inria/UR5_processed/base_filenames.txt","r").readlines()]

filename = filenames[0]
obss.shape
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

for filename in filenames:
    obss = np.load(root_folder+filename+".npz.obs.npy")
    actss = np.load(root_folder+filename+".npz.acts.npy")
    ann = np.load(root_folder+filename+".annotation.npy")
    obss_cont = np.concatenate([obss[:,:14], obss[:,37:49], obss[:,72:84], obss[:,107:]],axis=1)
    np.save(root_folder+filename+".obs_cont.npy", obss_cont)
    obss_cont.shape
    # obss[:,14:37][3]
    obss_disc1 = np.argmax(np.abs(obss[:,14:37]), axis=1) # why are some negative??
    obss_disc2 = np.argmax(np.abs(obss[:,49:72]), axis=1)
    obss_disc3 = np.argmax(np.abs(obss[:,84:107]), axis=1)

    assert np.all(obss_disc1 == obss_disc1[0])
    assert np.all(obss_disc2 == obss_disc2[0])
    assert np.all(obss_disc3 == obss_disc3[0])
    obss_disc1 = obss_disc1[0]
    obss_disc2 = obss_disc2[0]
    obss_disc3 = obss_disc3[0]

    obss_disc1 = vocab[object_types[obss_disc1]]
    obss_disc2 = vocab[object_types[obss_disc2]]
    obss_disc3 = vocab[object_types[obss_disc3]]

    ann
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

np.save(root_folder+"obs_all_augmented.npy", obs_all)
np.save(root_folder+"acts_all.npy", acts_all)
np.save(root_folder+"terminals_all.npy", terminals_all)
np.save(root_folder+"disc_cond_all.npy", disc_cond_all)
