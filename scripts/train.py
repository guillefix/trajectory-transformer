import os
import numpy as np
import torch
import pdb

import trajectory.utils as utils
import trajectory.datasets as datasets
import trajectory.datasets.lang_robot
from trajectory.models.transformers import GPT, LanguageConditionalGPT


class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    dataset_size: int = -1
    num_workers: int = 0
    config: str = 'config.offline'
    continue_train: bool = False
    not_predict_obs: bool = False
    weight_decay: float = 0.1
    obs_noise: float = 0.0

#######################
######## setup ########
#######################

args = Parser().parse_args('train')

#######################
####### dataset #######
#######################

env = datasets.load_environment(args.dataset)

sequence_length = args.subsampled_sequence_length * args.step

dataset_config = utils.Config(
    datasets.LanguageGoalDataset,
    savepath=(args.savepath, 'data_config.pkl'),
    env=args.dataset,
    N=args.N,
    penalty=args.termination_penalty,
    sequence_length=sequence_length,
    max_path_length=3000,
    step=args.step,
    discount=args.discount,
    discretizer=args.discretizer,
    dataset_size=args.dataset_size,
    not_predict_obs=args.not_predict_obs,
    obs_noise=args.obs_noise,
)

dataset = dataset_config()
obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
disc_cond_dim = dataset.disc_cond_dim
transition_dim = dataset.joined_dim

#######################
######## model ########
#######################

block_size = args.subsampled_sequence_length * transition_dim - 1
print(
    f'Dataset size: {len(dataset)} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)

if args.continue_train:
    model, starting_epoch = utils.load_model(args.logbase, args.dataset, args.exp_name,
            epoch=args.gpt_epoch, device=args.device)
else:
    model_config = utils.Config(
        LanguageConditionalGPT,
        savepath=(args.savepath, 'model_config.pkl'),
        ## discretization
        vocab_size=args.N, block_size=block_size, lang_vocab_size=73,
        ## architecture
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd*args.n_head,
        ## dimensions
        observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim, lang_len=disc_cond_dim,
        ## loss weighting
        action_weight=args.action_weight, reward_weight=args.reward_weight, value_weight=args.value_weight,
        ## dropout probabilities
        embd_pdrop=args.embd_pdrop, resid_pdrop=args.resid_pdrop, attn_pdrop=args.attn_pdrop,
    )
    model = model_config()
    starting_epoch = 0
model.to(args.device)

#######################
####### trainer #######
#######################

warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
# final_tokens = 20 * warmup_tokens
n_epochs = int(1e6 / len(dataset) * args.n_epochs_ref)
final_tokens = len(dataset) * n_epochs * args.batch_size
starting_tokens = len(dataset) * starting_epoch * args.batch_size

trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=args.weight_decay, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args.lr_decay,
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    ## dataloader
    num_workers=args.num_workers,
    device=args.device,
    starting_epoch=starting_epoch,
    starting_tokens=starting_tokens
)

trainer = trainer_config()

#######################
###### main loop ######
#######################

## scale number of epochs to keep number of updates constant
save_freq = max(1,int(n_epochs // args.n_saves))
# save_freq = 1
# from torch.utils.data import Subset
# dataset_old = dataset
# dataset = Subset(dataset, np.arange(100))
# dataset.N = dataset_old.N

for epoch in range(starting_epoch, starting_epoch+n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args.dataset} | {args.exp_name}')

    trainer.train(model, dataset)

    ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
    #save_epoch = (epoch + 1) // save_freq * save_freq
    if epoch % save_freq == 0:
        save_epoch = epoch
        statepath = os.path.join(args.savepath, f'state_{save_epoch}.pt')
        print(f'Saving model to {statepath}')

        ## save state to disk
        state = model.state_dict()
        torch.save(state, statepath)
