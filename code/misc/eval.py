from gym.wrappers.monitor import Monitor
from collections import deque
import numpy as np

def evaluate_policy(policy, eparams, eval_env, eps_num=None, itr=None):
    '''
        runs policy for X episodes and returns average reward
        return:
            average reward per episodes
    '''
    if eparams.record_policy:
        #TODO specific folder per task for the policy
        eval_env = Monitor(eval_env, directory="./video/", force=True)
    cum_reward = 0.
    
    task_id = eval_env.task_id

    #TODO move this somewhere smart
    eval_env.action_space.np_random.seed(eparams.seed)
    
    for _ in range(eparams.num_evals):
        eps_reward = 0
        obs = eval_env.reset()
        done = False
        stp = 0
        
        ### history ####
        rewards_hist = deque(maxlen=eparams.history_length) 
        actions_hist = deque(maxlen=eparams.history_length)
        obsvs_hist   = deque(maxlen=eparams.history_length)

        rewards_hist.append(0)
        obsvs_hist.append(obs.copy())

        rand_action = eval_env.action_space.sample()
        actions_hist.append(rand_action.copy())

        while not done and stp < eval_env._max_episode_steps:
            if eparams.record_policy:
                eval_env.render()
            
            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32) #(hist, )
            np_pre_obsvs  = np.asarray(obsvs_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)           
            
            action = policy.select_action(np.array([task_id]), np.array(obs), 
                    np.array(np_pre_actions), np.array(np_pre_rewards), 
                    np.array(np_pre_obsvs), deterministic=True, eval_step=True)
            
            new_obs, reward, done, _ = eval_env.step(action)
            eps_reward += reward
            stp += 1

            ## new becomes old
            rewards_hist.append(reward)
            actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())

            obs = new_obs.copy()

        cum_reward += eps_reward

    out = {}
    out['avg_reward'] = cum_reward / eparams.num_evals
    if eparams.monitor_success:
        out['success_rate'] = np.mean(eval_env.pop_successes())
    
    #TODO somehow, cant close Meta-World...
    if not any(env_name in eparams.env_name for env_name in ["CW10","MT10", "MT50"]):
        eval_env.close()
    
    return out


def evaluate_policy_on_all_tasks_cl(alg, eparams, eval_envs, task_id,  
                                total_timesteps, total_updates,
                                wrt_csv_eval=None, wrt_csv_eval_final=None, 
                                wandb=None):

    global_eprewmean = 0
    nb_tasks = eparams.nb_tasks
    task_eval = []
    task_success = []
    for eval_task_id in range(nb_tasks):
        #FIXME replace me back when bug is solved
        # eval_env = eval_envs[eval_task_id]()
        eval_env = eval_envs[eval_task_id]
        eval_temp = evaluate_policy(alg, eparams=eparams, eval_env=eval_env)
        task_eval.append(eval_temp['avg_reward'])
        if eparams.monitor_success:
            task_success.append(eval_temp['success_rate'])
    global_eprewmean = np.mean(task_eval)
    global_eprewstd = np.std(task_eval)
    past_eprewmean = np.mean(task_eval[:task_id])
    if eparams.monitor_success:
        global_success_mean = np.mean(task_success)
        global_success_std = np.std(task_success)
        past_success_mean = np.mean(task_success[:task_id])

    
    if wrt_csv_eval is not None:
        log_dict = {'current_task':task_id,
                    'total_updates':total_updates,
                    'total_timesteps':total_timesteps,                    
                    'eval/current_eprew_mean':task_eval[task_id],
                    'eval/past_eprew_mean':past_eprewmean,
                    'eval/global_eprew_mean':global_eprewmean,
                    'eval/global_eprew_std':global_eprewstd}
        if eparams.monitor_success:
            log_dict.update({
                        'eval/current_success_mean':task_success[task_id],
                        'eval/past_success_mean':past_success_mean,
                        'eval/global_success_mean':global_success_mean,
                        'eval/global_success_std':global_success_std,
            })
        print("---------------------------------------")
        print(f"Evaluation during task {task_id} after {total_updates} updates:")
        print(f"Episodic reward for the current task {task_eval[task_id]:0.2f}")
        print(f"Average episodic reward over all tasks {global_eprewmean:0.2f} +/- {global_eprewstd:0.2f}")
        if eparams.monitor_success:
            print(f"Success for the current task {task_success[task_id]:0.2f}")
            print(f"Average success over all tasks {global_success_mean:0.2f} +/- {global_success_std:0.2f}")
        print("---------------------------------------")
        wrt_csv_eval.write(log_dict)
        
        if wandb is not None:
            wandb.log(log_dict)
        
        return task_eval[task_id] 

    if wrt_csv_eval_final is not None:
        for eval_task_id in range(nb_tasks):
            log_dict = {'current_task':task_id,
                        'total_updates':total_updates,
                        f'final/eprew_mean_task{str(eval_task_id).zfill(2)}':task_eval[eval_task_id]}
            if eparams.monitor_success:
                log_dict.update({
                        f'final/eprew_success_task{str(eval_task_id).zfill(2)}':task_success[eval_task_id]})
            if wandb is not None:
                wandb.log(log_dict, step=total_updates)
            #TODO(add support for transfer matrix)
        log_dict = {'current_task':task_id,
                    'total_updates':total_updates,
                    'total_timesteps':total_timesteps,
                    'final/current_eprew_mean':task_eval[task_id],
                    'final/past_eprew_mean':past_eprewmean,
                    'final/global_eprew_mean':global_eprewmean,
                    'final/global_eprew_std':global_eprewstd}
        if eparams.monitor_success:
            log_dict.update({
                'final/current_success_mean':task_success[task_id],
                'final/past_success_mean':past_success_mean,
                'final/global_success_mean':global_success_mean,
                'final/global_success_std':global_success_std})
        wrt_csv_eval_final.write(log_dict)
        if wandb is not None:
            wandb.log(log_dict)
    
    out = {'current_eprew_mean':task_eval[task_id]}
    if eparams.monitor_success:
        out.update({'current_success_rate':task_success[task_id]})
    return out



def evaluate_policy_on_all_tasks_mtl(alg, eparams, eval_envs, total_timesteps, 
                                total_updates, wrt_csv_eval=None, wandb=None):

    global_eprewmean = 0
    nb_tasks = eparams.nb_tasks
    task_eval = []
    task_success = []
    for eval_task_id in range(nb_tasks):
        #FIXME replace me back when bug is solved
        # eval_env = eval_envs[eval_task_id]()
        eval_env = eval_envs[eval_task_id]
        eval_temp = evaluate_policy(alg, eparams=eparams, eval_env=eval_env)
        task_eval.append(eval_temp['avg_reward'])
        if eparams.monitor_success:
            task_success.append(eval_temp['success_rate'])
    global_eprewmean = np.mean(task_eval)
    global_eprewstd = np.std(task_eval)
    if eparams.monitor_success:
        global_success_mean = np.mean(task_success)
        global_success_std = np.std(task_success)
    
    if wrt_csv_eval is not None:
        log_dict = {
                    'total_updates':total_updates,
                    'total_timesteps':total_timesteps,
                    'eval/global_eprew_mean':global_eprewmean,
                    'eval/global_eprew_std':global_eprewstd}
        if eparams.monitor_success:
            log_dict.update({
                        'eval/global_success_mean':global_success_mean,
                        'eval/global_success_std':global_success_std,
            })
        print("---------------------------------------")
        print(f"Evaluation after {total_updates} updates:")
        print(f"Average episodic reward over all tasks {global_eprewmean:0.2f} +/- {global_eprewstd:0.2f}")
        if eparams.monitor_success:
            print(f"Average success over all tasks {global_success_mean:0.2f} +/- {global_success_std:0.2f}")
        print("---------------------------------------")
        wrt_csv_eval.write(log_dict)
        
        if wandb is not None:
            for eval_task_id in range(nb_tasks):
                log_dict.update({
                        f'eval_per_task/eprew_mean_task{str(eval_task_id).zfill(2)}':task_eval[eval_task_id]})
                if eparams.monitor_success:
                    log_dict.update({
                        f'eval_per_task/eprew_success_task{str(eval_task_id).zfill(2)}':task_success[eval_task_id]})
            
            wandb.log(log_dict)