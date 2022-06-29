import numpy as np
from misc.utils import softmax
import math

# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
# and https://github.com/sfujim/TD3/blob/master/utils.py

class Buffer(object):
	def __init__(self, max_size: int=1e6, 
					   experience_replay: bool=False,
					   curr_task_sampl_prob: float=0., 
				 	   history_length: int=1,
					   ):
		
		self.max_size = max_size
		self.hist_len = history_length
		self.experience_replay = experience_replay
		self.curr_task_sampl_prob = curr_task_sampl_prob

		self.reset()

	def reset(self):
		self.curr_storage = {}
		self.past_storage = {} 
		self.curr_task_bfrac = 1.
		self.curr_task = 0

		#NOTE these values are useful because of the RNN padding
		self.curr_timesteps = 0
		self.past_timesteps = 0
		self.curr_valid_indexes = []
		self.past_valid_indexes = []

	
	def reset_curr_buffer(self, curr_task):

		self.curr_task = curr_task

		## push current data in past buffer and reinit
		for key in self.curr_storage:
			if key not in self.past_storage: 
				self.past_storage[key] = self.curr_storage[key]
			else:
				self.past_storage[key] = np.concatenate((self.past_storage[key],
														 self.curr_storage[key]))
		if len(self.past_valid_indexes)==0:
			self.past_valid_indexes = self.curr_valid_indexes
		else:
			next_indexes = self.past_valid_indexes[-1] + 1 + self.curr_valid_indexes 
			self.past_valid_indexes = np.concatenate((self.past_valid_indexes, next_indexes))
		self.past_timesteps += self.curr_timesteps
		self.curr_storage = {}
		self.curr_timesteps = 0
		self.curr_valid_indexes = []
		
		## recompute the proper sampling proportion
		if self.experience_replay:
			self.curr_task_bfrac = max(1./(curr_task+1), self.curr_task_sampl_prob)
		else:
			self.curr_task_bfrac = 1.

	def add(self, data):
		'''
			data ==> (task_id, episode_timestep, state, next_state, action, reward, done, previous_action, previous_reward)
		'''
		
		## first time
		if len(self.curr_storage) == 0:
			for key in data:
				self.curr_storage[key] = data[key]
			# self.curr_valid_indexes = np.where(data['episode_timestep']>=0)[0]
			self.curr_valid_indexes = np.where(self.curr_storage['episode_timestep']>=0)[0]
			self.curr_timesteps += data['episode_timestep'][-1] + 1
			return

		## add data
		for key in data:
			self.curr_storage[key] = np.concatenate((self.curr_storage[key], data[key]))
		self.curr_valid_indexes = np.where(self.curr_storage['episode_timestep']>=0)[0]
		self.curr_timesteps += data['episode_timestep'][-1] + 1
		
		##################
		# if the current buffer is full
		# i.e. in the MTL and STL case
		# FIFO
		while len(self.curr_valid_indexes) > self.max_size:
			
			## delete the oldest episode
			idx_to_delete = np.where(self.curr_storage['episode'] == self.curr_storage['episode'][0])[0]

			for key in self.curr_storage:
				self.curr_storage[key] = self.curr_storage[key][(idx_to_delete[-1]+1):]
		
			self.curr_valid_indexes = np.where(self.curr_storage['episode_timestep']>=0)[0]
		##################
		
		##################
		# if the past buffer is full
		# i.e. in the CL case
		# balanced FIFO to keep uniformly sampling all tasks
		if self.past_storage != {}:
			task_ids, task_id_counts = np.unique(self.past_storage['task_id'], return_counts=True)
		
		while len(self.past_valid_indexes) > self.max_size:

			## find task w/ the most data
			task_to_del = task_ids[np.argmax(task_id_counts)]

			## delete the oldest episode
			idx_task_to_del = np.where(self.past_storage['task_id']==task_to_del)[0]
			ep_to_delete = self.past_storage['episode'][idx_task_to_del][0]
			idx_to_delete = np.where(self.past_storage['episode'][idx_task_to_del] == ep_to_delete)[0]

			for key in self.past_storage:
				self.past_storage[key] = np.delete(self.past_storage[key], idx_to_delete, axis=0)

			self.past_valid_indexes = np.where(self.past_storage['episode_timestep']>=0)[0]
			task_id_counts[np.where(task_ids==task_to_del)[0]] -= len(idx_to_delete)
		##################

		

	def size_rb(self):
		if len(self.past_storage['obs']) + len(self.curr_storagep['obs']) == self.max_size:
			return self.max_size
		else:
			return len(self.past_storage['obs']) + len(self.curr_storage['obs'])

	def get_batch(self, idx, buffer):
		
		out = {}
		out['task_id'] = buffer['task_id'][idx]
		out['obs'] = buffer['obs'][idx] 
		out['action'] = buffer['action'][idx]
		out['reward'] = buffer['reward'][idx]
		out['next_obs'] = buffer['next_obs'][idx]
		out['done'] = buffer['done'][idx]

		if self.hist_len == 0: 
			return out

		#### RNN mode ####
		shift_idx = np.tile(np.arange(-self.hist_len, 0), len(idx))
		rnn_idx = np.tile(idx, self.hist_len)
		rnn_idx.sort()
		rnn_idx += shift_idx
		
		out['previous_actions'] = buffer['action'][rnn_idx].reshape(len(idx), -1)
		out['previous_rewards'] = buffer['reward'][rnn_idx].reshape(len(idx), -1)
		out['previous_obs'] = buffer['obs'][rnn_idx].reshape(len(idx), -1)

		rnn_idx += 1
		out['current_actions'] = buffer['action'][rnn_idx].reshape(len(idx), -1)
		out['current_rewards'] = buffer['reward'][rnn_idx].reshape(len(idx), -1)
		out['current_obs'] = buffer['obs'][rnn_idx].reshape(len(idx), -1)
		
		return out  


	def sample(self, batch_size):
		'''
			Returns tuples of (state, next_state, action, reward, done,
							  )
		'''
		curr_batch_size = math.floor(self.curr_task_bfrac * batch_size)
		past_batch_size = batch_size - curr_batch_size

		curr_idx = np.random.randint(0, len(self.curr_valid_indexes), curr_batch_size)
		### NOTE ### 
		# this sort is important to reproduce the exact paper's results
		curr_idx.sort()
		############
		curr_idx = self.curr_valid_indexes[curr_idx]
		data = self.get_batch(curr_idx, self.curr_storage)
		
		if past_batch_size>0:
			past_idx = np.random.randint(0, len(self.past_valid_indexes), past_batch_size)
			past_idx = self.past_valid_indexes[past_idx]
			past_data = self.get_batch(past_idx, self.past_storage)
			for key in data:
				data[key] = np.concatenate((data[key], past_data[key]))
			
			
		out = {
				'task_id': data['task_id'],
				'obs': data['obs'], 
				'action': data['action'],
				'reward': data['reward'].reshape(-1, 1),
				'next_obs': data['next_obs'],
				'done': data['done'].reshape(-1, 1),
		}
		if self.hist_len>0:
			out.update({
					'previous_actions': data['previous_actions'], 
					'previous_rewards': data['previous_rewards'],
					'previous_obs': data['previous_obs'],
					'current_actions':data['current_actions'],
					'current_rewards':data['current_rewards'],
					'current_obs':data['current_obs']
			})
		
		return out