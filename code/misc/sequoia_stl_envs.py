from functools import partial
from typing import Callable, Dict, List
import gym
from gym.wrappers import TimeLimit

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


def get_env(
    env_name: str = 'CW10',
    max_episode_steps: int = 200,
    train_seed: int = 123,
    task_id: int = 0,
    monitor_success: bool = True,
) -> gym.Env:
    
    from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
    setting = IncrementalRLSetting(
        dataset=env_name,
        max_episode_steps=max_episode_steps,
    )
    
    base_env_fn = setting.train_envs[task_id]()

    wrappers = [partial(TimeLimit, max_episode_steps=max_episode_steps)]
    if monitor_success:
        wrappers += [SuccessCounter]

    train_env = partial(create_env, base_env_fn, wrappers=wrappers, seed=train_seed)()
    
    return train_env

if __name__ == "__main__":

    env = get_env()