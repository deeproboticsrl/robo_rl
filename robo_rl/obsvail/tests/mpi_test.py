import sys

from mpi4py import MPI

num_processes = 2

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['../mpi_worker.py'],
                           maxprocs=num_processes)

status = MPI.Status()

trajectory_length = 10
comm.bcast(trajectory_length, root=MPI.ROOT)

num_iterations = 2

finished = False
for _ in range(num_iterations):
    comm.bcast(finished, root=MPI.ROOT)
    reward = None
    dones = [False] * num_processes
    episode_rewards = [0] * num_processes

    all_done = all(dones)

    # Receive initial observation from each worker and send action
    for i in range(num_processes):
        observation = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=MPI.ANY_TAG)
        action = [status.source % 2] * 19  # action = policy(observation)
        comm.send(action, dest=status.source, tag=status.tag)

    # Receive from any worker till all of them finish one episode
    while not all_done:
        observation = comm.recv(source=MPI.ANY_SOURCE, status=status, tag=MPI.ANY_TAG)
        reward = comm.recv(source=status.source, tag=status.tag)
        done = comm.recv(source=status.source, tag=status.tag)
        dones[status.tag] = done
        all_done = all(dones)
        episode_rewards[status.tag] += reward
        if not done:
            action = [status.source % 2] * 19  # action = policy(observation)
            comm.send(action, dest=status.source, tag=status.tag)

    print(episode_rewards)

finished = True
comm.bcast(finished, root=MPI.ROOT)
