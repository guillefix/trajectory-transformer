def distribute_tasks(tasks, rank, size, residue_in_last=True):
    num_tasks = len(tasks)
    num_tasks_per_job = num_tasks//size
    tasks_ids = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
    if residue_in_last:
        if rank == size - 1:
            tasks_ids += list(range(size*num_tasks_per_job, num_tasks))
    else:
        if rank < num_tasks%size:
            tasks_ids.append(size*num_tasks_per_job+rank)
    tasks = [tasks[i] for i in tasks_ids]
    return tasks
