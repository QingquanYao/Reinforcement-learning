import gym

num_episode=500
env=gym.make('our_environments:discrete_singlependulum-v0')
state=env.reset()
env.render()
for i in range(num_episode):
    env.render()
    env.step(env.action_space.sample())

