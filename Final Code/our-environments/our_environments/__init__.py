from gym.envs.registration import register

register(
    id='cartpole-v0',
    entry_point='our_environments.envs:CartPoleEnv',
)

register(
    id='continuous_singlependulum-v0',
    entry_point='our_environments.envs:ContinuousSinglePendulumEnv',
)
register(
    id='discrete_singlependulum-v0',
    entry_point='our_environments.envs:DiscreteSinglePendulumEnv',
)


register(
    id='doublependulum-v0',
    entry_point='our_environments.envs:DoublePendulumEnv',
)

register(
    id='cliffwalking-v0',
    entry_point='our_environments.envs:CliffWalkingEnv',
)

register(
    id='mountaincar-v0',
    entry_point='our_environments.envs:MountainCarEnv',
)

register(
    id='countinuous_mountaincar-v0',
    entry_point='our_environments.envs:Continuous_MountainCarEnv',
)

register(
    id='basic-v2',
    entry_point='our_environments.envs:BasicEnv2',
)