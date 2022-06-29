'''
NOTE:
The codebase has been refactored since the version that was used to generate the public checkpoints
as well as the images created by this file.
The arguments of the loaded model are thus different. 
'''

import argparse
import torch
import os
import numpy as np
from collections import deque
import random
from gym.wrappers.monitor import Monitor
import json
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='public_ck/cw10_sac_CW10_full_meta_world-v0_ER-RNN_gc1.0_s7', help = 'path to model')
parser.add_argument('--stop_at_success', type=bool, default=True)
parser.add_argument('--last_task', type=int, default=9)
parser.add_argument('--compression_algo', type=str, default='pca', choices = ['pca', 'tsne'])
parser.add_argument('--compression_dim', type=int, default=2, choices = [2, 3])
parser.add_argument('--fig_path', type=str, default='code/scripts/figs/')

def evaluate_policy(policy, eparams, eval_env):
    '''
        runs policy for X episodes and returns average reward
        return:
            average reward per episodes
    '''
    if args.record_policy:
        #TODO specific folder per task for the policy
        eval_env = Monitor(eval_env, directory="./video/", force=True)
    out = []
    
    task_id = eval_env.task_id

    print(f'evaluating task {task_id}')

    #TODO move this somewhere smart
    env.action_space.np_random.seed(eparams.seed)
    
    for _ in range(eparams.num_evals):
        eps_reward = 0
        obs = eval_env.reset()
        done = False
        stp = 0
        
        ### history ####
        rewards_hist = deque(maxlen=eparams.history_length) 
        actions_hist = deque(maxlen=eparams.history_length)
        obsvs_hist   = deque(maxlen=eparams.history_length)
        context_hist = list()

        rewards_hist.append(0)
        obsvs_hist.append(obs.copy())

        rand_action = env.action_space.sample()
        actions_hist.append(rand_action.copy())

        while not done and stp < env._max_episode_steps:
            if args.record_policy:
                eval_env.render()
            
            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32) #(hist, )
            np_pre_obsvs  = np.asarray(obsvs_hist, dtype=np.float32).flatten() #(hist, action_dim) => (hist *action_dim,)           
            
            action, context = policy.select_action(np.array([task_id]), np.array(obs), 
                    np.array(np_pre_actions), np.array(np_pre_rewards), 
                    np.array(np_pre_obsvs), deterministic=True, eval_step=True, ret_context=True)
            
            new_obs, reward, done, extra = eval_env.step(action)
            eps_reward += reward
            stp += 1

            ## new becomes old
            rewards_hist.append(reward)
            actions_hist.append(action.copy())
            obsvs_hist.append(obs.copy())
            context_hist.append(context.copy())

            obs = new_obs.copy()

            if _args.stop_at_success and extra['success']:
                #NOTE: in this case, the success monitoring (below) wont work
                break
        
        out.append(context_hist)
    if not _args.stop_at_success:
        print(f'success: {np.mean(eval_env.pop_successes())}')

    return out

def evaluate_policy_on_all_tasks(alg, eparams, eval_envs): 

    nb_tasks = eparams.nb_tasks
    task_eval = []
    for eval_task_id in range(nb_tasks):
        eval_env = eval_envs[eval_task_id]
        eval_temp = evaluate_policy(alg, eparams=eparams, eval_env=eval_env)
        task_eval.append(eval_temp)
        
    return task_eval 

if __name__ == "__main__":

    _args = parser.parse_args()
    
    context_path = f'context_success_task{_args.last_task}.pkl' if _args.stop_at_success \
            else f'context_task{_args.last_task}.pkl'
    context_path = os.path.join(_args.model_path, context_path) 

    ## check if the context file exists
    # if os.path.exists(context_path):
    if False:
        with open(context_path, 'rb') as f:
            context_list = pickle.load(f)
    
    ## else, compute them
    else: 
            
        ##############################
        #### Args
        ##############################

        #load old args
        with open(os.path.join(_args.model_path,'args.json' ) , 'rt') as f:
            args = argparse.Namespace()
            args.__dict__.update(json.load(f)['args'])

        print('------------')
        print(args.__dict__)
        print('------------')

        ##############################
        #### Generic setups
        ##############################
        CUDA_AVAL = torch.cuda.is_available()

        if not args.disable_cuda and CUDA_AVAL:
            gpu_id = "cuda:" + str(args.gpu_id)
            device = torch.device(gpu_id)
            print("**** Yayy we use GPU %s ****" % gpu_id)

        else:
            device = torch.device('cpu')
            print("**** No GPU detected or GPU usage is disabled, sorry! ****")

        if not args.disable_cuda and CUDA_AVAL and args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("****** cudnn.deterministic is set ******")

        torch.set_num_threads(args.set_num_threads)

        ##############################
        # Set seeds
        ##############################
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        ##############################
        # Env setup and Seeds
        ##############################
        if any(env_name in args.env_name for env_name in ["CW10", "MT10", "MT50"]):
            import sys
            sys.path.append("code/")
            from misc.sequoia_envs import get_envs
        envs = get_envs(env_name=args.env_name, 
                        nb_tasks=args.nb_tasks,
                        train_seed=args.seed,
                        max_episode_steps=args.max_episode_steps,
                        eparams=args)
        env = envs[0]

        ##############################
        # Build Networks
        ##############################
        max_action = float(env.action_space.high[0])

        ######
        # This part to add context network
        ######
        reward_dim = 1
        dim_others = 0
        input_dim_context = None
        actor_idim = env.observation_space.shape[0]
        args.output_dim_context = 0

        if args.context_id: # means feed task_id as input_dim_context
            actor_idim += args.nb_tasks
            dim_others += args.nb_tasks

        if args.context_rnn: 
            input_dim_context = env.action_space.shape[0] + reward_dim \
                                + env.observation_space.shape[0]
            args.output_dim_context = args.hiddens_context[0]
            if args.context_rnn:
                actor_idim += args.hiddens_context[0]
                dim_others += args.hiddens_context[0]
        
        if args.context_embedding: # means we use the task-id to learn a task embedding
            actor_idim += args.hiddens_context[0]
            dim_others += args.hiddens_context[0]


        ######
        # now we build the actor and critic
        ######
        m_list_p = []
        if len(env.observation_space.shape) == 1:
            if str.lower(args.alg_name) == 'sac':
                # if not args.soft_modularization:
                    from models.networks import ActorSAC, CriticSAC, get_masks_stats
                    actor_args = {
                            'action_space': env.action_space,
                            'hidden_sizes': args.actor_hidden_sizes,
                            'input_dim': actor_idim,
                            'max_action': max_action,
                            'context_id': args.context_id,
                            'context_embedding': args.context_embedding,
                            'context_rnn': args.context_rnn,
                            'nb_tasks': args.nb_tasks,
                            'hiddens_dim_context': args.hiddens_context,
                            'input_dim_context': input_dim_context,
                            'output_context': args.output_dim_context,
                            'history_length': args.history_length,
                            'obsr_dim': env.observation_space.shape[0],
                            'device': device,
                            'eparams': args
                    }
                    actor_net = ActorSAC(**actor_args).to(device)
                    
                    critic_args = {
                            'action_space': env.action_space,
                            'hidden_sizes': args.critic_hidden_sizes,
                            'input_dim': env.observation_space.shape[0],
                            'context_id': args.context_id,
                            'context_embedding': args.context_embedding,
                            'context_rnn': args.context_rnn,
                            'nb_tasks': args.nb_tasks,
                            'hiddens_dim_context': args.hiddens_context,
                            'input_dim_context': input_dim_context,
                            'output_context': args.output_dim_context,
                            'dim_others': dim_others,
                            'history_length': args.history_length,
                            'obsr_dim': env.observation_space.shape[0],
                            'device': device,
                            'eparams': args
                    }
                    critic_net = CriticSAC(**critic_args).to(device)
                

                    m_list_p.append(actor_net)
                    m_list_p.append(critic_net)
            else:
                raise ValueError("%s model is not supported for %s env" % 
                                (args.env_name, env.observation_space.shape))

        print('-----------------------------')
        print("Name of env:", args.env_name)
        print("Observation_space:", env.observation_space )
        print("Action space:", env.action_space )
        print('----------------------------')

        ##############################
        # Algs setup 
        ##############################
        if str.lower(args.alg_name) == 'sac':
            import algs.SAC.sac as malg
            alg_args = {
                    'actor': actor_net,
                    'critic': critic_net,
                    'lr': args.lr,
                    'gamma': args.gamma,
                    'ptau': args.ptau,
                    'batch_size': args.batch_size,
                    'max_action': max_action,
                    'alpha': args.alpha,
                    'use_auto_entropy': args.use_auto_entropy,
                    'action_dims': env.action_space.shape,
                    'history_length': args.history_length,
                    'grad_clip': args.grad_clip,
                    'monitor_grads': args.monitor_grads,
                    'device': device
            }
            alg = malg.SAC(**alg_args)

        else:
            raise ValueError("%s alg is not supported" % args.alg_name)

        ## load model
        ck_timestamp = 999200 * (_args.last_task+1)


        actor_ck = torch.load(os.path.join(_args.model_path, f'ck_update_{ck_timestamp}.pt'),
                map_location=device)['model_states_actor']
        alg.actor.load_state_dict(actor_ck)

        context_list = evaluate_policy_on_all_tasks(alg, args, envs)

        with open(context_path, "wb") as fp:
            pickle.dump(context_list, fp)

    ################# Finish getting the contexts #######################

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from sklearn import decomposition
    from sklearn.manifold import TSNE

    plt.style.use('seaborn-whitegrid')

    ## analysis args
    successful_task_id = [0, 2, 5, 8]
    cMap = {successful_task_id[0]: 'r',
            successful_task_id[1]: 'b',
            successful_task_id[2]: 'g',
            successful_task_id[3]: 'm',}

    ## preprocess the context
    X, y, evals, t = [], [], [], []
    for task_id in successful_task_id:
        for eval_id in range(len(context_list[task_id])):
            X.extend(context_list[task_id][eval_id])
            y.extend(task_id * np.ones(len(context_list[task_id][eval_id])))
            _t = np.arange(len(context_list[task_id][eval_id])) / len(context_list[task_id][eval_id])
            t.extend(_t)
            evals.extend(eval_id * np.ones(len(context_list[task_id][eval_id])))

    X = np.array(X)
    y = np.array(y)
    t = np.array(t)
    evals = np.array(evals)

    ## compression time
    if _args.compression_algo == 'pca':
        pca = decomposition.PCA(n_components=_args.compression_dim)
        pca.fit(X)
        X = pca.transform(X)
        print(pca.explained_variance_ratio_)
    elif _args.compression_algo == 'tsne':
        tsne = TSNE(n_components=_args.compression_dim, learning_rate='auto')
        X = tsne.fit_transform(X) 
    else:
        raise ValueError("%s compression is not supported" % _args.compression_algo)

    
    c = [cMap[i] for i in y]
    # alpha = t * (1-0.3) + 0.3
    alpha = 0.5
    
    ## plot the PCA w/ all the data
    fname = os.path.join(_args.fig_path, f'{_args.compression_algo}_task{_args.last_task}.png')
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    
    plt.scatter(X[:,0], X[:,1], c=c, alpha=alpha,)
    
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False)

    plt.legend(loc='best', frameon=True, framealpha=1)
    fig.savefig(fname, bbox_inches='tight', dpi=100, rasterized=True)
    
    ## plot the PCA w/ all the data w/ Edge
    fname = os.path.join(_args.fig_path, f'{_args.compression_algo}_task{_args.last_task}_wEdge.png')
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    
    plt.scatter(X[:,0], X[:,1], c=c, alpha=alpha, edgecolor="k")
    
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False)

    plt.legend(loc='best', frameon=True, framealpha=1)
    fig.savefig(fname, bbox_inches='tight', dpi=100, rasterized=True)

    ## plot the PCA single trajectory
    #NOTE eval_id 3 for task9 and 7 for task1
    fname = os.path.join(_args.fig_path, f'{_args.compression_algo}_task{_args.last_task}_single.png')
    X = X[evals==3,:]
    y = y[evals==3]
    t = t[evals==3]
    c = [cMap[i] for i in y]
    alpha = 0.5

    ## find starting point
    start = t==0
    marker_map = {True: u'*', False: u'o'}
    m = [marker_map[i] for i in start]
    size_map = {True: 300, False: 50}
    s = [size_map[i] for i in start]
    alpha_map = {True: 1, False: 0.5}
    alpha = [alpha_map[i] for i in start]

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    if _args.compression_dim == 2:
        
        for i in range(len(X)):
            plt.scatter(X[i,0], X[i,1], c=c[i], alpha=alpha[i], marker=m[i], s=s[i], edgecolor="k")
        
        ## adding a line
        for task_id in successful_task_id:
            idx = np.where(y==task_id)[0]
            plt.plot(X[idx,0], X[idx,1], c=cMap[task_id], alpha=0.5, label=f'Task {task_id}')
        
        ## replot the starting point
        for i in range(len(X)):
            if start[i]:
                plt.scatter(X[i,0], X[i,1], c=c[i], alpha=alpha[i], marker=m[i], s=s[i], edgecolor="k")
        
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labelleft=False)

    else:
        raise ValueError("pca_dim should be 2 or 3")

    fig.savefig(fname, bbox_inches='tight', dpi=100, rasterized=True)