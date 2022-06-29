import numpy as np
import torch
from collections import deque

class Runner:
    """
      This class generates batches of experiences
    """
    def __init__(self,
                 env,
                 model,
                 replay_buffer = None,
                 burn_in = 1e4,
                 total_timesteps = 1e6,
                 history_length = 1,
                 multi_task=False,
                 nb_tasks=None,
                 device = 'cpu'):
        '''
            nsteps: number of steps 
        '''
        self.model = model
        self.burn_in = burn_in
        self.device = device
        self.episode_rewards = deque(maxlen=10)
        self.episode_lens = deque(maxlen=10)
        self.replay_buffer = replay_buffer
        self.total_timesteps = total_timesteps
        self.hist_len = history_length
        self.multi_task = multi_task
        self.nb_tasks = nb_tasks

        if self.multi_task:
            self.current_task = 0
            self.envs = env
            self.env = self.envs[self.current_task]
        else:
            self.env = env

    def run(self, timestep, episode):
        '''
            This function add transition to replay buffer
            run a single episode / collects a single trajectory
        '''
        # TODO make this function much cleaner

        obs = self.env.reset()
        task_id = self.env.task_id
        done = False
        episode_timestep = 0
        episode_reward = 0
        reward_epinfos = []
        
        ########
        ## create a queue to keep track of past rewards and actions
        ########
        rewards_hist = deque(maxlen=self.hist_len)
        actions_hist = deque(maxlen=self.hist_len)
        obsvs_hist   = deque(maxlen=self.hist_len)

        ## Given batching schema, I need to build a full seq to keep in replay buffer
        ## Add to all zeros.
        zero_action = np.zeros(self.env.action_space.shape[0])
        zero_obs    = np.zeros(obs.shape)
        for _ in range(self.hist_len):
            rewards_hist.append(0)
            actions_hist.append(zero_action.copy())
            obsvs_hist.append(zero_obs.copy())

        rand_action = self.env.action_space.sample()
        
        ######
        ## create a list keeping the full sequences, for buffer storing
        ######
        full_task_id_hist = [task_id] * self.hist_len
        full_episode_hist = [episode] * self.hist_len
        full_rewards_hist = list(rewards_hist)
        full_actions_hist = list(actions_hist)[1:] + [rand_action.copy()] if self.hist_len > 0 else []
        full_obsvs_hist = list(obsvs_hist) 
        full_episode_timesteps_hist = [-1] * self.hist_len
        full_done_hist = [0] * self.hist_len
        full_next_obsvs_hist = list(obsvs_hist)[1:] + [obs.copy()] if self.hist_len > 0 else []
        
        ## now add obs to the seq
        rewards_hist.append(0)
        actions_hist.append(rand_action.copy())
        #NOTE either next lines can work
        # obsvs_hist.append(zero_obs.copy())
        obsvs_hist.append(obs.copy())

        ######
        # Start collecting data
        #####
        #TODO should we use max_path_length like in MQL here?
        while not done:

            #####
            # Convert actions_hist, rewards_hist to np.array and flatten them out
            # for example: hist =7, actin_dim = 11 --> np.asarray(actions_hist(7, 11)) ==> flatten ==> (77,)
            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32) #(hist, )
            np_pre_obsvs = np.asarray(obsvs_hist, dtype=np.float32).flatten()  #(hist, action_dim) => (hist *action_dim,)
            
            # Select action randomly or according to policy
            if timestep < self.burn_in:
                action = self.env.action_space.sample()
            else:
                action = self.model.select_action(
                        np.array([task_id]), np.array(obs),
                        np.array(np_pre_actions), np.array(np_pre_rewards), np.array(np_pre_obsvs))

            # Perform action
            next_obs, reward, done, _ = self.env.step(action) 
            
            # we need this bc we don't done=True when we reach the end of the episode
            if episode_timestep + 1 == self.env._max_episode_steps:
                done_bool = 0
            else:
                done_bool = float(done)

            episode_reward += reward
            reward_epinfos.append(reward)
            
            # for buffer
            full_task_id_hist.append(task_id)
            full_episode_hist.append(episode)
            full_episode_timesteps_hist.append(episode_timestep)
            full_obsvs_hist.append(obs.copy()) 
            full_actions_hist.append(list(action))
            full_rewards_hist.append(reward)
            full_next_obsvs_hist.append(next_obs.copy()) 
            full_done_hist.append(done_bool)
            
            # new becomes old
            rewards_hist.append(reward)
            actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())

            obs = next_obs.copy()
            episode_timestep += 1
            timestep += 1


        #######
        ## Store data in replay buffer
        #######
        self.replay_buffer.add({'task_id': np.array(full_task_id_hist),
                                'episode': np.array(full_episode_hist),
                                'episode_timestep': np.array(full_episode_timesteps_hist),
                                'obs': np.array(full_obsvs_hist),
                                'action': np.array(full_actions_hist),
                                'reward': np.array(full_rewards_hist),
                                'next_obs': np.array(full_next_obsvs_hist),
                                'done': np.array(full_done_hist)
        })
        
        info = {}
        info['episode_timestep'] = episode_timestep
        # info['update_iter'] = update_iter
        info['episode_reward'] = episode_reward
        info['epinfos'] = [{"r": round(sum(reward_epinfos), 6), "l": len(reward_epinfos)}]

        if self.multi_task:
            self.current_task = (self.current_task + 1) % self.nb_tasks
            self.env = self.envs[self.current_task]

        return info