# 500 for halfcheetah, 600 for hopper, 700 for walker

from functools import partial
from typing import Callable, Dict, List
import gym
from gym.wrappers import TimeLimit

EnvFn = Callable[[], gym.Env]

class SuccessCounter(gym.Wrapper):
    """From Continual World's Codebase"""

    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        if done:
            self.successes.append(self.current_success)
        return obs, reward, done, info

    def pop_successes(self):
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)


def get_envs(
    env_name: str,
    nb_tasks: int = 20,
    max_episode_steps: int = 1000,
    train_seed: int = 123,
    eparams=None,
) -> List[EnvFn]:
    from sequoia.settings.rl.incremental.setting import IncrementalRLSetting

    setting = IncrementalRLSetting(
        dataset=env_name, max_episode_steps=max_episode_steps, nb_tasks=nb_tasks,
    )

    train_envs: List[gym.Env] = []

    task_order = range(nb_tasks)

    for task_id in task_order:

        # Function that will create the env with the given task.
        if nb_tasks == 1:
            task_id = eparams.single_task_id
            print(f"========= single task on task {task_id} ==============")

        base_env_fn = setting.train_envs[task_id]
        env = base_env_fn()
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

        if eparams.monitor_success:
            env = SuccessCounter(env)
        if train_seed is not None:
            env.seed(train_seed)
        
        # Add metadata to env
        env.task_id = task_id
        env.nb_tasks = nb_tasks
        train_envs.append(env)

    return train_envs
