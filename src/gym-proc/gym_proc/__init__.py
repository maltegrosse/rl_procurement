from gym.envs.registration import register

register(
    id='Procurement-v0',
    entry_point='gym_proc.envs:ProcurementEnv',
)