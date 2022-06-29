from typing import Callable, Dict, List, Tuple, Type, Union
import gym
from misc.sequoia_envs import MetaDataWrapper 
from misc.ant_direction import AntDirectionEnv
import copy
from gym.wrappers import TimeLimit

EnvFn = Callable[[], gym.Env]


def get_envs(
    env_name: str,
    # version: str,
    nb_tasks: int = 20,
    max_episode_steps: int = 1000,
    train_seed: int = 123,
    eparams = None,
) -> List[EnvFn]:

    assert env_name == 'Ant_direction-v3'

    ## fromt learn2learn
    envs = AntDirectionEnv()

    train_envs: List[gym.Env] = []
    
    for task_id in range(nb_tasks):
        task = envs.sample_tasks(1)[0]
        envs.set_task(task)
        train_env = copy.deepcopy(envs)
        train_env.seed(train_seed)

        train_env = TimeLimit(train_env, max_episode_steps)
        train_env = MetaDataWrapper(train_env, 
                task_id=task_id, nb_tasks=nb_tasks, max_episode_steps=max_episode_steps)
        
        train_envs.append(train_env)

    return train_envs