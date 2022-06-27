import json
import pdb
from os.path import join
import os
from constants import *
from pathlib import Path
import numpy as np
import torch

import trajectory.utils as utils
import trajectory.datasets as datasets
import trajectory.datasets.lang_robot
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)
from trajectory.utils.arrays import to_torch


def run(args):

    #######################
    ####### models ########
    #######################

    dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
            'data_config.pkl')

    gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
            epoch=args.gpt_epoch, device=args.device)

    #######################
    ####### demo data #######
    #######################
    goal_str = args.goal_str
    traj_data = None
    if args.session_id is not None and args.rec_id is not None:
        if os.path.exists(data_folder+args.session_id+"/obs_act_etc/"+args.rec_id+"/data.npz"):
            traj_data = np.load(data_folder+args.session_id+"/obs_act_etc/"+args.rec_id+"/data.npz", allow_pickle=True)
        elif os.path.exists(data_folder+args.session_id+"/"+args.rec_id+"/data.npz"):
            traj_data = np.load(data_folder+args.session_id+"/"+args.rec_id+"/data.npz", allow_pickle=True)
            print(traj_data)
        if args.goal_str is None:
            goal_str = str(traj_data['goal_str'][0])

        # init_obss = np.load(processed_data_folder+"UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data.obs_cont_single_nocol_noarm_trim_scaled.npy")[0]
        # traj_data_actss = np.load(processed_data_folder+"UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data.acts_trim_scaled.npy")
        traj_data_actss = np.load(processed_data_folder+"acts_all.npy")
        traj_data_obss = np.load(processed_data_folder+"obs_all_augmented.npy")
        init_obss = traj_data_obss[0]
        init_actss = traj_data_actss[0:1]
    print(goal_str)

    #######################
    ####### dataset #######
    #######################

    env = datasets.load_environment(args.dataset) #TODO: set max_len from the dataset
    if args.render:
        env.render()
    # renderer = utils.make_renderer(args)
    timer = utils.timer.Timer()

    discretizer = dataset.discretizer
    discount = dataset.discount
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    value_fn = lambda x: discretizer.value_fn(x, args.percentile)
    preprocess_fn = datasets.get_preprocess_fn(env.name)

    #######################
    ###### main loop ######
    #######################

    # observation = env.reset()
    objects = traj_data['obj_stuff'][0] if args.restore_objects else None
    if traj_data is not None:
        observation = env.reset(o=traj_data["obs"][0], info_reset=None, description=goal_str, joint_poses=traj_data["joint_poses"][0], objects=objects, restore_objs=args.restore_objects)
        observation = init_obss
    else:
        observation = env.reset(o=None, info_reset=None, description=goal_str, joint_poses=None, objects=objects, restore_objs=args.restore_objects)
    total_reward = 0

    if goal_str is None:
        goal_str = env.goal_str

    ## observations for rendering
    rollout = [observation.copy()]

    ## previous (tokenized) transitions for conditioning transformer
    # print(init_actss.shape)
    # acts_dim = init_actss.size
    # print(acts_dim)
    # if args.session_id is not None and args.rec_id is not None:
    #     acts_discrete = discretizer.discretize(init_actss, subslice=[0, action_dim])
    #     acts_discrete = to_torch(acts_discrete, dtype=torch.long)
    #     context = [acts_discrete]
    # else:
    context = []
    context = update_context(context, discretizer, preprocess_fn(observation), init_actss[0], 0.0, args.max_context_transitions)

    # T = env.max_episode_steps
    assert args.max_episode_length < env.max_episode_length
    T = args.max_episode_length
    for t in range(T):

        # import pdb; pdb.set_trace()
        observation = preprocess_fn(observation)
        # print(observation)

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            # import pdb; pdb.set_trace()
            # print(observation)
            # og_observation = observation
            # if t<10:
            # print(observation)
            # observation=traj_data_obss[t]
            # # observation += 1e-2*np.random.randn(*observation.shape)
            # print(observation)
            # print(observation-og_observation)
            prefix, lang_goal = make_prefix(discretizer, context, observation, args.prefix_context, lang_goal=env.lang_goal)
            # observation += 1e-7*np.random.randn(*observation.shape)
            # print(prefix)
            # prefix, lang_goal = make_prefix(discretizer, context, observation, args.prefix_context, lang_goal=env.lang_goal)
            # print(prefix)
            # print(prefix)
            # print(prefix.shape)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                gpt, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                lang_goal=lang_goal, temperature=args.temperature
            )

        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)

        ## execute action in environment
        print(action)
        # # # if t == 0:
        # action = traj_data_actss[t]
        # # action = discretizer.discretize(action, subslice=[observation_dim,observation_dim+action_dim])
        # # action = discretizer.reconstruct(action, subslice=[observation_dim, observation_dim+action_dim])[0]
        # print(action)
        next_observation, reward, terminal, _ = env.step(action)

        success = reward > 0

        print(goal_str+": ",success)
        achieved_goal_end = success

        ## update return
        total_reward += reward
        # score = env.get_normalized_score(total_reward)
        score=reward

        ## update rollout observations and context transitions
        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

        print(
            f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
            f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
        )

        ## visualization
        # if t % args.vis_freq == 0 or terminal or t == T:
        #
        #     ## save current plan
        #     renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), sequence_recon, env.state_vector())
        #
        #     ## save rollout thus far
        #     renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        if terminal: break

        observation = next_observation
        # observation = init_obss

    ## save result as a json file
    json_path = join(args.savepath, 'rollout.json')
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    if args.save_eval_results:
        varying_args = args.varying_args.split(",")
        if not Path(args.savepath+"/results").is_dir():
            os.mkdir(args.savepath+"/results")
        filename = args.savepath+"/results/"+"eval_"
        args_dict = vars(args)
        for k in varying_args:
            filename += str(args_dict[k])+"_"
        filename += "_".join(goal_str.split(" "))+".txt"
        if os.path.exists(filename):
            with open(filename, "a") as f:
                f.write(str(achieved_goal_end)+","+str(t)+"\n")
        else:
            with open(filename, "w") as f:
                f.write("achieved_goal_end,num_steps"+"\n")
                f.write(str(achieved_goal_end)+","+str(t)+"\n")

if __name__=="__main__":
    class Parser(utils.Parser):
        dataset: str = 'halfcheetah-medium-expert-v2'
        config: str = 'config.offline'
        goal_str: str = None
        save_eval_results: bool = False
        restore_objects: bool = False
        render: bool = False
        session_id: str = None
        rec_id: str = None
        varying_args: str = 'session_id,rec_id'
        max_episode_length: int = 3000
        temperature: float = 1.0

    #######################
    ######## setup ########
    #######################

    args = Parser().parse_args('plan')

    run(args)
