from typing import Callable, Dict, List, Tuple, Type, Union
import gym
# from misc.sequoia_envs import MetaDataWrapper

from misc.ant_direction import AntDirectionEnv
from misc.quadratic_opt import QuadraticEnv
import copy
from gym.wrappers import TimeLimit

EnvFn = Callable[[], gym.Env]


def get_envs(
    env_name: str,
    nb_tasks: int = 20,
    max_episode_steps: int = 1000,
    train_seed: int = 123,
    eparams=None,
) -> List[EnvFn]:
    assert env_name in ["Ant_direction-v3", "Quadratic_opt"]

    ## from learn2learn
    if env_name == "Ant_direction-v3":
        envs = AntDirectionEnv()
    if env_name == "Quadratic_opt":
        envs = QuadraticEnv(
            obs_dim=eparams.obs_dim,
            train_seed=train_seed,
            max_episode_steps=max_episode_steps,
        )

    train_envs: List[gym.Env] = []

    tasks = envs.sample_tasks(nb_tasks)

    for task_id in range(nb_tasks):
        envs.set_task(tasks[task_id])
        train_env = copy.deepcopy(envs)
        train_env.seed(train_seed)

        train_env = TimeLimit(train_env, max_episode_steps)
        # train_env = MetaDataWrapper(
        #     train_env,
        #     task_id=task_id,
        #     nb_tasks=nb_tasks,
        #     max_episode_steps=max_episode_steps,
        # )
        ## add metadata to the env
        train_env.task_id = task_id
        train_env.nb_tasks = nb_tasks
        train_env.max_episode_steps = max_episode_steps

        train_envs.append(train_env)

    return train_envs
