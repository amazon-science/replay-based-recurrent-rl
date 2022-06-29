# 500 for halfcheetah, 600 for hopper, 700 for walker

from functools import partial
from typing import Callable, Dict, List
import gym
from gym.wrappers import TimeLimit

EnvFn = Callable[[], gym.Env]

class MetaDataWrapper(gym.Wrapper):
    def __init__(self, env, task_id, nb_tasks, max_episode_steps):
        super(MetaDataWrapper, self).__init__(env)
        self.task_id = task_id
        self.nb_tasks = nb_tasks
        self._max_episode_steps = max_episode_steps

class SuccessCounter(gym.Wrapper):
    """From Continual World's Codebase"""
    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info.get('success', False):
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


def create_env(
    env: gym.Env,
    kwargs: Dict = None,
    wrappers: List[Callable[[gym.Env], gym.Env]] = None,
    seed: int = None,
) -> gym.Env:
    """
    1. Create an env instance by calling `env_fn`;
    2. Wrap it with the wrappers in `wrappers`, if any;
    3. seed it with `seed` if it is not None.
    """
    wrappers = wrappers or []
    for wrapper in wrappers:
        env = wrapper(env)
    if seed is not None:
        env.seed(seed)
    return env


def get_envs(
    env_name: str,
    nb_tasks: int = 20,
    max_episode_steps: int = 1000,
    train_seed: int = 123,
    eparams = None,
) -> List[EnvFn]:
    
    from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
    setting = IncrementalRLSetting(
        dataset=env_name,
        max_episode_steps=max_episode_steps,
        nb_tasks=nb_tasks,
)
    
    train_envs: List[gym.Env] = []

    task_order = range(nb_tasks)
    
    for task_id in task_order:

        print(task_id)

        # Function that will create the env with the given task.
        if nb_tasks==1:
            task_id=eparams.single_task_id
            print(f'========= single task on task {task_id} ==============')

        base_env_fn = setting.train_envs[task_id]()

        wrappers = []
        wrappers += [partial(TimeLimit, max_episode_steps=max_episode_steps)]
        if eparams.monitor_success:
            wrappers += [SuccessCounter]
        wrappers += [partial(MetaDataWrapper, task_id=task_id,
                                              nb_tasks=nb_tasks,
                                              max_episode_steps=max_episode_steps)]

        train_env = partial(create_env, base_env_fn, wrappers=wrappers, seed=train_seed)()
        train_envs.append(train_env)
    
    return train_envs
