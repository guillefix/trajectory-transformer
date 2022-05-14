from constants import *
import gym
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Evaluate LangGoalRobot environment')
parser.add_argument('--base_filename', default="base_filenames_single_objs_filtered.txt", help='specify the base filename to use to get the list of files')
parser.add_argument('--obs_mod', default="obs_cont_single_nocol_noarm_trim_scaled", help='specify the base filename to use to get the list of files')
parser.add_argument('--disc_mod', default="annotation_simp_wnoun", help='specify the base filename to use to get the list of files')
parser.add_argument('--act_mod', default="acts_trim_scaled", help='specify the base filename to use to get the list of files')

def run(args):

    filenames=[x[:-1] for x in open(processed_data_folder+args.base_filename,"r").readlines()]
    # len(filenames)
    # get_ann = lambda x: open(root_folder+x+".annotation.txt","r").read()
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    obs_all = None
    acts_all = None
    terminals_all = None
    disc_cond_all = None

    for filename in filenames:
        actss = np.load(processed_data_folder+filename+"."+args.act_mod+".npy")
        obss_cont = np.load(processed_data_folder+filename+"."+args.obs_mod+".npy")
        # obss_cont = np.concatenate([obss_cont[:,:11], obss_cont[:,12:]], axis=1)
        disc_cond = np.load(processed_data_folder+filename+"."+args.disc_mod+".npy")
        if len(disc_cond.shape) == 2:
            disc_cond = disc_cond[0]
        if len(disc_cond.shape) == 3:
            disc_cond = disc_cond[0,:,0]
        # import pdb; pdb.set_trace()

        print(obss_cont.shape[0])
        # terminals = np.empty((obss_cont.shape[0],)).fill(False)
        terminals = np.full((obss_cont.shape[0],),False)
        terminals[-1] = True

        if obs_all is None:
            obs_all = obss_cont
        else:
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

    print(obs_all.shape)
    print(acts_all.shape)
    print(terminals_all.shape)
    print(disc_cond_all.shape)

    np.save(processed_data_folder+"obs_all_augmented.npy", obs_all)
    np.save(processed_data_folder+"acts_all.npy", acts_all)
    np.save(processed_data_folder+"terminals_all.npy", terminals_all)
    np.save(processed_data_folder+"disc_cond_all.npy", disc_cond_all)

    # np.save(root_folder+"obs_all_augmented_smol.npy", obs_all)
    # np.save(root_folder+"acts_all_smol.npy", acts_all)
    # np.save(root_folder+"terminals_all_smol.npy", terminals_all)
    # np.save(root_folder+"disc_cond_all_smol.npy", disc_cond_all)

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
