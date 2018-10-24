from robo_rl.common import TrajectoryBuffer
from robo_rl.common import print_heading

trajectory_buffer = TrajectoryBuffer(1000)

observations = [[i + j for j in range(2)] for i in range(5)]
trajectory1 = [observations[0], observations[1]]
trajectory2 = [observations[4], observations[2]]
trajectory_buffer.add(trajectory1)
trajectory_buffer.add(trajectory2)

print_heading("Sample trajectory")
print(trajectory_buffer.sample(batch_size=2))

print_heading("Sample at particular timestep")
print(trajectory_buffer.sample_timestep(batch_size=2, timestep=1))
