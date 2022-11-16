import pybullet as p
from extra_utils.data_utils import get_obs_cont, fix_quaternions, one_hot, get_tokens
from create_simple_dataset import has_concrete_object_ann, check_if_exact_one_object_from_obs, get_new_obs_from_obs
from pathlib import Path
from constants import *
from src.envs.utils import save_traj as save_traj_inner
import json

import torch
import numpy as np

from src.envs.env_params import get_env_params
from src.envs.descriptions import generate_all_descriptions
import uuid
env_params = get_env_params()
_, _, all_descriptions = generate_all_descriptions(env_params)

def generate_goal(single_obj=False, use_train_set=False):
    if use_train_set:
        filenames = [x[:-1] for x in open(processed_data_folder+"/base_filenames_single_objs_filtered.txt", "r").readlines()]
        # descriptions = [open(processed_data_folder+"/"+x+".npz.annotation.txt","r").read()[:-1] for x in filenames]
        descriptions = [open(processed_data_folder+"/"+x+".npz.annotation.txt","r").read().rstrip() for x in filenames]
        # print(descriptions)
    else:
        descriptions = list(all_descriptions)
    if single_obj:
        ann = np.random.choice(descriptions)
        print(ann)
        has_conc_obj, adj, obj = has_concrete_object_ann(ann)
        while not has_conc_obj:
            ann = np.random.choice(descriptions)
            has_conc_obj, adj, obj = has_concrete_object_ann(ann)
            descriptions.remove(ann)
        return ann
    else:
        return np.random.choice(descriptions)

def scale_inputs(obs_scaler, acts_scaler, prev_obs, prev_acts, noarm=True, add_batch_dim = False):
    # print(prev_obs.shape)
    if prev_obs is not None and obs_scaler is not None:
        if not noarm:
            if len(prev_obs.shape) == 1:
                prev_obs[3:7] = fix_quaternions(prev_obs[3:7])
            else:
                prev_obs[:,3:7] = fix_quaternions(prev_obs[:,3:7])
        prev_obs = obs_scaler.transform(prev_obs)
    if prev_acts is not None and acts_scaler is not None:
        if len(prev_acts.shape) == 1:
            prev_acts[3:7] = fix_quaternions(prev_acts[3:7])
        else:
            prev_acts[:,3:7] = fix_quaternions(prev_acts[:,3:7])
        prev_acts = acts_scaler.transform(prev_acts)
    return prev_obs, prev_acts

def package_inputs(tokens, obs, acts, times_to_go=None, n_tiles=1):
    # print(n_tiles)
    # return [torch.from_numpy(tokens.copy()).unsqueeze(1).unsqueeze(1).cuda(), torch.from_numpy(prev_obs.copy()).unsqueeze(1).float().cuda(), torch.from_numpy(prev_acts.copy()).unsqueeze(1).float().cuda()]
    tokens = torch.from_numpy(tokens)
    # tokens = F.one_hot(tokens,num_classes=67)
    # n_tiles = 1
    if len(tokens.shape) == 3:
        tokens = tokens.long().to(device)
        n_tiles = tokens.shape[1]
    elif len(tokens.shape) == 2:
        if add_batch_dim:
            tokens = tokens.unsqueeze(1).long().to(device)
        else:
            tokens = tokens.long().to(device)
    else:
        raise NotImplementedError
    if len(obs.shape) == 3:
        obs = torch.from_numpy(obs).float().to(device)
        if n_tiles > 1:
            assert obs.shape[1] == n_tiles
        n_tiles = obs.shape[1]
    elif len(obs.shape) == 2:
        if add_batch_dim:
            obs = torch.from_numpy(obs).unsqueeze(1).float().to(device)
        else:
            obs = torch.from_numpy(obs).float().to(device)
    else:
        raise NotImplementedError
    if len(acts.shape) == 3:
        acts = torch.from_numpy(acts).float().to(device)
        if n_tiles > 1:
            assert acts.shape[1] == n_tiles
        n_tiles = acts.shape[1]
    elif len(acts.shape) == 2:
        if add_batch_dim:
            acts = torch.from_numpy(acts).unsqueeze(1).float().to(device)
        else:
            acts = torch.from_numpy(acts).float().to(device)
    else:
        raise NotImplementedError
    if times_to_go is not None:
        if len(times_to_go.shape) == 3:
            times_to_go = torch.from_numpy(times_to_go).float().to(device)
            if n_tiles > 1:
                assert times_to_go.shape[1] == n_tiles
            n_tiles = times_to_go.shape[1]
            # print(times_to_go)
            # print(times_to_go.shape)
        elif len(times_to_go.shape) == 2:
            if add_batch_dim:
                times_to_go = torch.from_numpy(times_to_go).unsqueeze(1).float().to(device)
            else:
                times_to_go = torch.from_numpy(times_to_go).float().to(device)
        else:
            raise NotImplementedError

    if tokens.shape[1] == 1:
        tokens = torch.from_numpy(np.tile(tokens.numpy(), (1,n_tiles,1))) #torch.tile not availble in torch 1.7.0
        # print(n_tiles)
        #tokens = torch.tile(tokens, (1,n_tiles,1))
    if obs.shape[1] == 1:
        obs = torch.from_numpy(np.tile(obs.numpy(), (1,n_tiles,1))) #torch.tile not availble in torch 1.7.0
    if acts.shape[1] == 1:
        acts = torch.from_numpy(np.tile(acts.numpy(), (1,n_tiles,1))) #torch.tile not availble in torch 1.7.0
        #acts = torch.tile(acts, (1,n_tiles,1))
    if times_to_go is not None:
        if times_to_go.shape[1] == 1:
            times_to_go = torch.tile(times_to_go, (1,n_tiles,1))
    # tokens = tokens.unsqueeze(1).cuda()

    if times_to_go is not None:
        return [times_to_go, tokens, obs, acts]
    else:
        return [tokens, obs, acts]
    # return [torch.from_numpy(tokens).unsqueeze(1).unsqueeze(1).cpu(), torch.from_numpy(prev_obs).unsqueeze(1).float().cpu(), torch.from_numpy(prev_acts).unsqueeze(1).float().cpu()]

def process_obs(obs, obj_index, obs_mod):
    new_obs = obs
    if obs_mod[:8] == "obs_cont":
        new_obs = get_obs_cont(obs[None])
        if "single" in obs_mod:
            nocol = "nocol" in obs_mod
            noarm = "noarm" in obs_mod
            include_size = "incsize" in obs_mod
            # print("include_size", include_size)
            new_obs = get_new_obs_from_obs(new_obs, obj_index, nocol=nocol, noarm=noarm, include_size=include_size)
    return new_obs[0]

def make_inputs(obs_scaler, acts_scaler, obs, action_scaled, prev_obs, prev_acts, times_to_go, tokens, obj_index, obs_mod, convert_to_torch=True, n_tiles=1):
    # import pdb; pdb.set_trace()
    new_obs = process_obs(obs, obj_index, obs_mod)
    new_obs, acts = scale_inputs(obs_scaler, None, new_obs[None], action_scaled, "noarm" in obs_mod)
    if prev_obs is not None:
        prev_obs = np.concatenate([prev_obs[1:],new_obs])
    else:
        prev_obs = new_obs
    if times_to_go is not None:
        new_times_to_go = times_to_go[-1]-1
        if new_times_to_go == 0:
            new_times_to_go = [times_to_go_start]
        new_times_to_go = np.array([new_times_to_go])
        times_to_go = np.concatenate([times_to_go[1:],new_times_to_go])
    # print(new_obs)
    if prev_acts is not None:
        prev_acts = np.concatenate([prev_acts[1:],acts])
    else:
        prev_acts = acts
    if convert_to_torch:
        inputs = package_inputs(tokens, prev_obs, prev_acts, times_to_go, n_tiles=n_tiles)
    else:
        if times_to_go is not None:
            inputs = (times_to_go, tokens, prev_obs, prev_acts)
        else:
            inputs = (tokens, prev_obs, prev_acts)
    return inputs

def scale_outputs(acts_scaler, scaled_acts):
    if acts_scaler is not None:
        acts = acts_scaler.inverse_transform(scaled_acts)
    else:
        acts = scaled_acts
    #print(acts)
    if len(acts.shape) == 2:
        acts = acts[0]
    act_pos = [acts[0],acts[1],acts[2]]
    act_gripper = [acts[7]]
    acts_euler = list(p.getEulerFromQuaternion(acts[3:7]))
    action = act_pos + acts_euler + act_gripper
    return np.array(action)

def compute_relabelled_logPs(obs_scaler, acts_scaler, t, new_descriptions, env, input_lengths, ann_mod_idx, prev_obs_ext, prev_acts_ext):
    obj_stuff = env.instance.get_stuff_to_save()
    tokenss = []
    good_descs = []
    obss_temp = []
    for jj, desc in enumerate(new_descriptions):
        if obs_mod in ["obs_cont_single_nocol_noarm_trim_scaled", "obs_cont_single_nocol_noarm_scaled"]:
            has_conc_obj, color_temp, object_type_temp = has_concrete_object_ann(desc)
            if not has_conc_obj:
                continue
            objects = env.instance.objects_added
            matches = 0
            obj_index_tmp = -1
            for i, obj in enumerate(objects):
                if obj['type'] == object_type and obj['color'] == color:
                    matches += 1
                    obj_index_tmp = i
        tokens = get_tokens(desc, max_length=input_lengths[ann_mod_idx], obj_stuff=obj_stuff)
        tokenss.append(tokens)
        good_descs.append(desc)
        new_obs_temp = prev_obs_ext
        if obs_mod[:8] == "obs_cont":
            new_obs_temp = get_obs_cont(new_obs_temp)
            if "single" in obs_mod:
                nocol = "nocol" in obs_mod
                noarm = "noarm" in obs_mod
                new_obs_temp = get_new_obs_from_obs(new_obs_temp, obj_index_tmp, nocol=nocol, noarm=noarm)
        print(new_obs.shape)
        scaled_new_obs_temp, _ = scale_inputs(obs_scaler, acts_scaler, new_obs_temp, None, "noarm" in obs_mod)
        obss.append(scaled_new_obs_temp)

    if len(good_descs) > 0:
        print("Good descriptions: "+", ".join(good_descs))
        tokenss = np.stack(tokenss, axis=1)
        obss = np.stack(obss, axis=1)
        # print(tokenss.shape)
        print(obss.shape)
        logPs_temp = None
        for j in range(save_chunk_size-1):
            if ttg_mod is not None:
                times_to_go_temp = np.array(range(t+save_chunk_size-j,t+save_chunk_size-j+context_size_ttg,-1))
                inputs = package_inputs(tokenss, obss[j:j+context_size_obs], prev_acts_ext[j:j+context_size_acts], times_to_go_temp, add_batch_dim=True)
            else:
                inputs = package_inputs(tokenss, obss[j:j+context_size_obs], prev_acts_ext[j:j+context_size_acts], add_batch_dim=True)
            prepared_acts = np.expand_dims(prev_acts_ext[j+context_size_acts],0)
            prepared_acts = np.tile(prepared_acts, (inputs[0].shape[1],1,1))
            prepared_acts = torch.from_numpy(prepared_acts).float().to(device)
            logP = model.training_step({**{"in_"+input_mods[j]: inputs[j].permute(1,0,2) for j in range(len(input_mods))}, "out_"+output_mods[0]: prepared_acts}, batch_idx=0, reduce_loss=False)
            logP = logP.cpu().numpy()
            # print(logP)
            if logPs_temp is None:
                logPs_temp = logP[None]
            else:
                logPs_temp = np.concatenate([logPs_temp, logP[None]])
        mean_logPs = np.mean(logPs_temp, axis=0)
        i = np.argmin(mean_logPs)
        print("mean logP achieved goal: "+str(mean_logPs))
        print("min mean logP achieved goal: "+str(mean_logPs[i])+", for goal "+good_descs[i])

def save_traj(descriptions, args, traj_data, obj_stuff):
    #root_folder_generated_data="/gpfsscratch/rech/imi/usc19dv/data/"
    # description = descriptions[-1]
    experiment_name = args["experiment_name"]
    new_session_id = experiment_name
    new_rec_id = str(uuid.uuid4())
    if not Path(root_folder_generated_data+"generated_data").is_dir():
        os.mkdir(root_folder_generated_data+"generated_data")
    if not Path(root_folder_generated_data+"generated_data/"+new_session_id).is_dir():
        os.mkdir(root_folder_generated_data+"generated_data/"+new_session_id)
    if not Path(root_folder_generated_data+"generated_data/"+new_session_id+"/"+new_rec_id).is_dir():
        os.mkdir(root_folder_generated_data+"generated_data/"+new_session_id+"/"+new_rec_id)
    npz_path = root_folder_generated_data+"generated_data/"+new_session_id+"/"+new_rec_id
    actss, obss, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception = traj_data
    save_traj_inner(npz_path, actss, obss, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, gripper_proprioception, descriptions, obj_stuff)
    args_file = root_folder_generated_data+"generated_data/"+new_session_id+"/"+new_rec_id+"/args.json"
    json_string = json.dumps(args)
    with open(args_file, "w") as f:
        f.write(json_string)
    descriptions_file = root_folder_generated_data+"generated_data/"+new_session_id+"/"+new_rec_id+"/descriptions.txt"
    with open(descriptions_file, "w") as f:
        f.write(",".join(descriptions))

    return new_rec_id
