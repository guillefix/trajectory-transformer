from run_inference import run
from extra_utils import distribute_tasks
import os
import trajectory.utils as utils
from constants import *
from scripts.plan import run

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.offline'
    base_filenames_file: str = 'base_filenames_single_objs_filtered.txt' #file listing demo sequence ids
    sample_goals: bool = False #file listing demo sequence ids
    num_tasks: int = 1 #number of tasks (overriden by number of sequence ids if base_filenames_file is not None)
    num_repeats: int = 1 #number of times each demo should be used

#######################
######## setup ########
#######################

args = Parser().parse_args('train')

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

#TODO: add task where we sample a random instruction

tasks = []

common_args = vars(args).copy()
del common_args["base_filenames_file"]
del common_args["num_repeats"]
del common_args["sample_goals"]
del common_args["num_tasks"]
if args.base_filenames_file is not None:
    with open(args.base_filenames_file, "r") as f:
        filenames = [x[:-1] for x in f.readlines()] # to remove new lines
    num_tasks = len(filenames)
    tasks = args.num_repeats*list(map(lambda x: {**common_args, "session_id": x.split("_")[1], "rec_id": x.split("_")[5]}, filenames))
elif args.sample_goals:
    from extra_utils.run_utils import generate_goal
    tasks = args.num_repeats*[{**common_args, "goal_str": generate_goal()} for i in range(args.num_tasks)]

    #filenames = filenames[:2]

#common_args = {"restore_objects": True}
tasks = distribute_tasks(tasks, rank, size)
#print(tasks)
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

for task in tasks:
    args = Struct(**task)
    run(args)
print("Finished. Rank: "+str(rank))
