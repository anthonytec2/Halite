from gym.envs.registration import register

register(
    id='halite-v0',
    entry_point='gym_halite.envs:HaliteEnv',
)