import gym

env = gym.make('Humanoid-v2')

episode_count = 0

env.seed(0)
while True:

    env.reset()
    # env.render()

    timestep = 0
    done = False
    episode_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        env.render()
        timestep += 1
        # print(timestep, episode_reward)
    episode_count += 1
    print(episode_count, episode_reward)
