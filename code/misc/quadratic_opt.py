#!/usr/bin/env python3

import gym
import numpy as np

# from gym.envs.mujoco.mujoco_env import MujocoEnv


from gym.core import Env
from gym import spaces
from scipy.optimize import minimize


class MetaEnv(Env):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/envs/meta_env.py)
    **Description**
    Interface for l2l envs. Environments have a certain number of task specific parameters that uniquely
    identify the environment. Tasks are then a dictionary with the names of these parameters as keys and the
    values of these parameters as values. Environments must then implement functions to get, set and sample tasks.
    The flow is then
    ```
    env = EnvClass()
    tasks = env.sample_tasks(num_tasks)
    for task in tasks:
        env.set_task(task)
        *training code here*
        ...
    ```
    **Credit**
    Adapted from Tristan Deleu and Jonas Rothfuss' implementations.
    """

    def __init__(self, task=None):
        super(Env, self).__init__()
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    def sample_tasks(self, num_tasks):
        """
        **Description**
        Samples num_tasks tasks for training or evaluation.
        How the tasks are sampled determines the task distribution.
        **Arguments**
        num_tasks (int) - number of tasks to sample
        **Returns**
        tasks ([dict]) - returns a list of num_tasks tasks. Tasks are
        dictionaries of task specific parameters. A
        minimal example for num_tasks = 1 is [{'goal': value}].
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        **Description**
        Sets the task specific parameters defined in task.
        **Arguments**
        task (dict) - A dictionary of task specific parameters and
        their values.
        **Returns**
        None.
        """
        self._task = task

    def get_task(self):
        """
        **Description**
        Returns the current task.
        **Arguments**
        None.
        **Returns**
        (task) - Dictionary of task specific parameters and their
        current values.
        """
        return self._task


class QuadraticEnv(MetaEnv):
    def __init__(
        self,
        obs_dim=1,
        train_seed=1,
        max_episode_steps=10,
        task=None,
    ):
        self.obs_dim = obs_dim
        self.train_seed = train_seed
        self.max_rew = 1_000
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(obs_dim,))
        # MetaEnv.__init__(self, task)

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        MetaEnv.set_task(self, task)
        self.A = task["A"]
        self.b = task["b"]
        self.c = task["c"]

    def sample_tasks(
        self, num_tasks,
    ):
        """task: r(x) = xAx + bx + c"""
        out = []
        A = None

        np.random.seed(self.train_seed)

        for _ in range(num_tasks):
            # Sample a positive definite matrix P from a normal distribution
            P = np.random.normal(size=(self.obs_dim, self.obs_dim))
            P = P.T @ P

            # Compute the Cholesky decomposition P = LL^T
            L = np.linalg.cholesky(P)

            # Set A = -L^T L
            A = -L.T @ L
            baseA = A

            # assert that A is negative definite
            assert np.all(np.linalg.eigvals(A) < 0)

            # sample b
            b = np.random.normal(size=self.obs_dim)

            # find the maximum value of xAx + b
            def f(x):
                return x.T.dot(A).dot(x) + b.T.dot(x)

            result = minimize(lambda x: -f(x), np.zeros(self.obs_dim))
            max_val = -result.fun
            c = self.max_rew - max_val

            out.append(
                {"A": A, "b": b, "c": c,}
            )

        return out

    # -------- Gym Methods --------
    def step(self, action):
        # apply the action to the observations
        self.x += action
        # compute the reward using the quadratic function
        reward = self.x.T.dot(self.A).dot(self.x) + self.b.T.dot(self.x) + self.c
        # scale the reward to be in [-inf, 1]
        reward = reward.item() / self.max_rew
        # scale the reward such that the maximum reward per episode is 100
        reward *= 100 / self.max_episode_steps
        # TODO: cap reward?
        # return the observations, reward, done flag, and additional info
        return self.x, reward, False, {}

    def reset(self):
        # reset the observations to some initial value
        self.x = np.random.randn(self.obs_dim)
        return self.x


if __name__ == "__main__":
    env = QuadraticEnv()
    for task in [env.get_task(), env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        action = env.action_space.sample()
        env.step(action)
