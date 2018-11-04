from mpi4py import MPI
from osim.env import ProstheticsEnv
from pros_ai import get_policy_observation

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

env = ProstheticsEnv(visualize=False)

trajectory_length = None
trajectory_length = comm.bcast(trajectory_length, root=0)

finished = False

finished = comm.bcast(finished, root=0)

while not finished:

    observation = env.reset(project=False)
    observation = get_policy_observation(observation)

    comm.send(observation, dest=0, tag=rank)

    done = False
    timestep = 0
    actions = None
    all_done = False

    while not done:
        action = comm.recv(source=0, tag=rank)
        observation, reward, done, _ = env.step(action)
        observation = get_policy_observation(observation)
        comm.send(observation, dest=0, tag=rank)
        comm.send(reward, dest=0, tag=rank)
        timestep += 1
        done = done or (timestep >= trajectory_length)
        comm.send(done, dest=0, tag=rank)
        # print(f"Time step - {timestep},rank - {rank}")

    print(f"Process {rank} finished in {timestep} time steps.")

    finished = comm.bcast(finished, root=0)
