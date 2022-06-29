import argparse
import torch
import os
import time
import numpy as np
from collections import deque
import random
from misc.utils import create_dir, CSVWriter, safemean, overwrite_args, \
        str2bool, restricted_float, print_model_info, take_snapshot, \
        setup_logAndCheckpoints
from oailibs import logger
from misc.buffer import Buffer
from misc.runner_offpolicy import Runner
from datetime import date
from misc.eval import evaluate_policy_on_all_tasks_cl

parser = argparse.ArgumentParser()

# Optim params
parser.add_argument('--lr', type=float, default=1e-3, help = 'Learning rate')
parser.add_argument('--replay_size', type=int, default = 1e6, help ='Replay buffer size int(1e6)')
parser.add_argument('--ptau', type=float, default=0.005 , help = 'Interpolation factor in polyak averaging')
parser.add_argument('--gamma', type=float, default=0.99, help = 'Discount factor [0,1]')
parser.add_argument("--burn_in", default=1000, type=int, help = 'How many timesteps purely random policy is run for') 
parser.add_argument("--warm_up", default=100, type=int, help = 'How many timesteps before updating the models') 
parser.add_argument("--batch_size", default=100, type=int, help = 'Batch size for both actor and critic')
parser.add_argument('--actor_hidden_sizes', nargs='+', type=int, default = [64, 64], help = 'indicates hidden size for actor. Set to 0 for linear')
parser.add_argument('--critic_hidden_sizes', nargs='+', type=int, default = [64, 64], help = 'indicates hidden size for critic. Set to 0 for linear')
parser.add_argument('--grad_clip', type=float, default = None, help = 'clip the gradient norm')

# General params
parser.add_argument('--multi_task', type=str2bool, default=False, choices=[False])
parser.add_argument('--env_name', type=str, default='Ant_direction-v3', choices=['Ant_direction-v3', 'MT10', 'MT50', 'CW10'])
parser.add_argument('--monitor_success', type=str2bool, default=False, help='monitor the success rate')
parser.add_argument('--max_episode_steps', type=int, default=100, help='to overwrite the default _max_episode_steps')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--alg_name', type=str, default='sac')
parser.add_argument("--total_timesteps", default=1e4, type=int, help = 'number of timesteps to train on per CL task')
parser.add_argument("--total_updates", default=None, type=int, help = 'if None, will be equal to total_timesteps')

parser.add_argument('--disable_cuda', type=str2bool, default=False)
parser.add_argument('--cuda_deterministic', default=False, action='store_true')
parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument('--set_num_threads', type=int, default=1, help='set_num_threads')

parser.add_argument('--log_id', default='dummy')
parser.add_argument('--record_policy', default=False, help='for recording the policies evaluation (WIP)')
parser.add_argument('--check_point_dir', default='./ck')
parser.add_argument('--log_dir', default='./log_dir')
parser.add_argument('--wandb_project', type=str, default=None, help='name of the WandB workspace. Set to None for no WandB logging')
parser.add_argument('--num_evals', type=int, default = 25, help ='Length eval episode')
parser.add_argument('--log_freq', type=int, default=1e2, help='how often (updates) we log results')
parser.add_argument("--eval_freq", type=int, default = 1e4, help ='How often (updates) we evaluate')
parser.add_argument('--save_freq', type=int, default = 1e5, help="how often (updates) we save")
parser.add_argument('--monitor_grads', type=str2bool, default = True, help="monitor gradient statistics")

## sac parameters
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--use_auto_entropy', default=False, action='store_true')

## context
parser.add_argument('--context_id', type=str2bool, default=False, help='add task_id to the observations')
parser.add_argument('--context_embedding', type=str2bool, default=False, help='learn a task embedding + add to observations')
parser.add_argument('--context_rnn', type=str2bool, default=False, help='add a context RNN to the observations')
parser.add_argument('--hiddens_context', nargs='+', type=int, default = [30, 30], help = 'indicates hidden size of context next') 
parser.add_argument('--history_length', type=int, default = 30)
parser.add_argument('--multi_head', type=str2bool, default=False, help ='use a seperate head for each task')
parser.add_argument('--task_agnostic', type=str2bool, default=False, help ='enforces task agnosticiy (mainly for TAMH)')

## cl setting
parser.add_argument('--nb_tasks', type=int, default=20, help='number of CL tasks')
parser.add_argument('--single_task_id', type=int, default=None, help='task id for single task exp')
parser.add_argument('--episodes_per_task', type=int, default=1000, help='number of allowed episode per task')

## cl method
parser.add_argument('--experience_replay', type=str2bool, default=False, help='if on, the replay_buffer doesnt reset')
parser.add_argument('--curr_task_sampl_prob', type=restricted_float, default=0.0, help='prob to sample current task data in ExpReplay. If none, it stays as uniform sampling')
parser.add_argument('--train_from_scratch', type=str2bool, default=False, help='if on, reset the neural network weight after each tasks')

## for args overwritting
parser.add_argument('--setting_config', type=str, default=None, help='setting name for configs overwritting')
parser.add_argument('--method_config', type=str, default=None, help='method name for configs overwritting')
parser.add_argument('--hparam_config', type=str, default=None, help='hparam configuration name for configs overwritting')


if __name__ == "__main__":

    ##############################
    #### Args
    ##############################
    args = parser.parse_args()
    if args.setting_config:
        args = overwrite_args(args, 'settings', args.setting_config) 
    if args.method_config:
        args = overwrite_args(args, 'methods', args.method_config) 
    if args.hparam_config:
        args = overwrite_args(args, 'hparams', args.hparam_config) 

    ## other checks
    if not args.context_rnn and args.history_length>0:
        args.history_length = 0
        print('======\nsetting history_length to 0, as there is no RNN\n=======')
    
    assert args.burn_in < args.total_timesteps
    assert args.warm_up < args.total_timesteps
    
    if args.train_from_scratch: assert not args.multi_head

    args.total_updates = args.total_timesteps if args.total_updates is None else args.total_updates
    
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
    # Create log folders, etc.
    ##############################
    create_dir(args.log_dir, cleanup = True)
    ## create folder for save checkpoints
    ck_fname_part, log_file_dir, fname_csv_eval = setup_logAndCheckpoints(args)
    fname_csv_eval_final = os.path.join(log_file_dir, 'eval_final.csv')
    logger.configure(dir = log_file_dir)
    wrt_csv_eval = None

    ##############################
    # Wandb setup 
    ##############################
    if args.wandb_project is not None:
        #NOTE wandb needs to be turned on
        os.environ['WANDB_MODE'] = 'dryrun'
        import wandb
        #TODO nasty bug here when launching jobs through ssh
        wandb.init(project=args.wandb_project, name=args.log_id)
        wandb.config.update(args)
        today = date.today()
        wandb.log({'launch_date': int(today.strftime("%Y%m%d"))})
    else:
        wandb=None

    ##############################
    # Set seeds
    ##############################
    # build_env already calls set seed,
    # Set seed the RNG for all devices (both CPU and CUDA)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    ##############################
    # Env setup and Seeds
    ##############################
    if any(env_name in args.env_name for env_name in ["CW10", "MT10", "MT50"]):
        from misc.sequoia_envs import get_envs
    else:
        from misc.learn2learn_env import get_envs
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
            from models.networks import ActorSAC, CriticSAC
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
        
        else:
            raise ValueError("%s model is not supported for %s env" % 
                            (args.env_name, env.observation_space.shape))

        m_list_p.append(actor_net)
        m_list_p.append(critic_net)

    total_params, _ = print_model_info(m_list_p)
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

    ## init replay buffer
    replay_buffer = Buffer(max_size=args.replay_size, 
            experience_replay=args.experience_replay,
            curr_task_sampl_prob=args.curr_task_sampl_prob, 
            history_length=args.history_length)

    ## log memory consumption
    #FIXME: 
    sample_size = (env.observation_space.shape[0] + env.action_space.shape[0]
                  + 1 + env.observation_space.shape[0]) # SARS'
    replay_buffer_size = args.replay_size * sample_size

    if wandb is not None:
        log_dict = {'total_params': total_params,
                    'replay_buffer_size': replay_buffer_size,
                    'total_memory': total_params + replay_buffer_size}
        wandb.log(log_dict)

    ##############################
    # Train and eval
    ##############################
    # take snapshot
    take_snapshot(args, ck_fname_part, alg, 0)

    ## define some vars
    total_timesteps = 0
    total_episodes = 0
    total_updates = 0
    cumulative_rewards = 0
    avg_current_eprew = []
    if args.monitor_success:
        avg_current_success = []
    tot_time_sampling = 0.
    tot_time_buffer_sampling = 0.
    tot_time_transfer = 0.
    tot_time_updating = 0.
    updates_multiplier = args.total_updates / args.total_timesteps
    timesteps_per_task = int(args.total_timesteps/args.nb_tasks)
    updates_per_task = int(args.total_updates/args.nb_tasks)

    # init loggers
    log_dict = {'current_task':-1,
                'total_updates':-1,
                'total_episodes':-1,
                'total_timesteps':-1,
                'eval/past_eprew_mean':-1,
                'eval/current_eprew_mean':-1,
                'eval/global_eprew_mean':-1,
                'eval/global_eprew_std':-1
                }
    if args.monitor_success:
        log_dict.update({
                'eval/past_success_mean':-1,
                'eval/current_success_mean':-1,
                'eval/global_success_mean':-1,
                'eval/global_success_std':-1
        })
    wrt_csv_eval = CSVWriter(fname_csv_eval, log_dict)
    log_dict = {'current_task':-1,
                'total_updates':-1,
                'total_timesteps':-1,
                'final/current_eprew_mean':-1,
                'final/past_eprew_mean':-1,
                'final/global_eprew_mean':-1,
                'final/global_eprew_std':-1}
    if args.monitor_success:
        log_dict.update({
                'final/current_success_mean':-1,
                'final/past_success_mean':-1,
                'final/global_success_mean':-1,
                'final/global_success_std':-1
        })
    wrt_csv_eval_final = CSVWriter(fname_csv_eval_final, log_dict)

    ############################################################################
    # Main loop train
    ############################################################################
    # Start total timer
    tstart = time.time()    
    print("Start main loop ...")

    for task_id in range(args.nb_tasks):

        ##############################
        # task loop before training
        ##############################
        print(f"Starting task {task_id} ...")
        env = envs[task_id]
        env.reset()

        replay_buffer.reset_curr_buffer(task_id)

        ## variables
        episode_num = 0
        update_num = 0
        timestep_num = 0
        updates_since_eval, updates_since_saving, updates_since_logging = 3*[0]
        warm_up_passed = False
        epinfobuf = deque(maxlen=100)
        epinfobuf_v2 = deque(maxlen=args.num_evals)

        ## training from train_from_scratch
        if args.train_from_scratch:
            print('re-initializing the neural network weights\n')
            del actor_net, critic_net, alg, alg_args['actor'], alg_args['critic']
            actor_net = ActorSAC(**actor_args).to(device)
            critic_net = CriticSAC(**critic_args).to(device)
            alg_args['actor'], alg_args['critic'] = actor_net, critic_net
            alg = malg.SAC(**alg_args)

        ## only reset buffer if ER isn't enabled
        if not args.experience_replay or args.train_from_scratch:
            print('resetting the replay buffer')
            replay_buffer.reset()


        ## rollout/batch generator
        rollouts = Runner(env=env,
                          model=alg,
                          replay_buffer=replay_buffer,
                          burn_in=args.burn_in,
                          total_timesteps=int(args.total_timesteps/args.nb_tasks),
                          history_length=args.history_length,
                          device=device)

        # Evaluate starting policy
        evaluate_policy_on_all_tasks_cl(alg, args, envs, task_id, total_timesteps, total_updates, 
                                     wrt_csv_eval=wrt_csv_eval, wandb=wandb)

        ##############################
        # task train loop
        ##############################
        # start task timer
        task_start = time.time()   
        
        while timestep_num < timesteps_per_task:

            #######
            # Interact and collect data until reset
            #######
            sampling_start = time.time() 
            data = rollouts.run(timestep_num, episode_num)
            tot_time_sampling += time.time() - sampling_start
            updates_to_run = data['episode_timestep'] 
            cumulative_rewards += data['episode_reward']
            epinfobuf.extend(data['epinfos'])
            epinfobuf_v2.extend(data['epinfos'])
            total_episodes += 1
            episode_num += 1
            total_timesteps += data['episode_timestep']
            timestep_num += data['episode_timestep']

            if timestep_num >= args.warm_up:
                warm_up_passed = True
            else:
                ## still in warmup
                continue

            #######
            # run training to calculate loss, run backward, and update params
            #######
            updates_to_run = int(updates_to_run * updates_multiplier)
            alg_stats, extra_alg_stats = alg.train(replay_buffer = replay_buffer, 
                                  iterations = updates_to_run,  
                                  )
            tot_time_buffer_sampling += sum(extra_alg_stats['time_buffer_sampling'])
            tot_time_transfer += sum(extra_alg_stats['time_transfer'])
            tot_time_updating += sum(extra_alg_stats['time_updating'])
            total_updates += updates_to_run
            update_num += updates_to_run
            updates_since_eval += updates_to_run
            updates_since_logging += updates_to_run
            updates_since_saving += updates_to_run


            #######
            # logging
            #######
            nseconds = time.time() - task_start
            # Calculate the fps (frame per second)
            fps = int(( timestep_num) / nseconds)
            
            if updates_since_logging >= args.log_freq:
                updates_since_logging %= args.log_freq
                log_dict= {
                           "current_task": task_id,
                           "total_updates": total_updates,
                           "total_timesteps": total_timesteps,
                           "total_episodes": total_episodes,
                           "train/episode_reward": float(data['episode_reward']),
                           'train/eprewmean': float(safemean([epinfo['r'] for epinfo in epinfobuf])),
                           'train/eplenmean': float(safemean([epinfo['l'] for epinfo in epinfobuf])),
                           'train/eprewmeanV2': float(safemean([epinfo['r'] for epinfo in epinfobuf_v2])),
                           'train/eplenmeanV2': float(safemean([epinfo['l'] for epinfo in epinfobuf_v2])),
                           'train/cumul_rew': cumulative_rewards,
                }

                for key in alg_stats:
                    log_dict['train/'+key] = np.mean(alg_stats[key])

                if args.monitor_grads:
                    for key in extra_alg_stats['critic_grad_summary']:
                        log_dict['gradients/critic_'+key] = \
                                extra_alg_stats['critic_grad_summary'][key]
                    
                    for key in extra_alg_stats['actor_grad_summary']:
                        log_dict['gradients/actor_'+key] = \
                                extra_alg_stats['actor_grad_summary'][key]

                
                ## add monitoring of code performance
                log_dict['code/fps'] = fps
                log_dict['code/tot_time_sampling'] = tot_time_sampling
                log_dict['code/time_sampling'] = tot_time_sampling / episode_num
                log_dict['code/tot_time_buffer_sampling'] = tot_time_buffer_sampling
                log_dict['code/time_buffer_sampling'] = tot_time_buffer_sampling / total_updates 
                log_dict['code/tot_time_transfer'] = tot_time_transfer
                log_dict['code/time_transfer'] = tot_time_transfer / total_updates
                log_dict['code/tot_time_updating'] = tot_time_updating
                log_dict['code/time_updating'] = tot_time_updating / total_updates

                for key, value in log_dict.items():
                    logger.record_tabular(key, value)

                if wandb is not None:
                    wandb.log(log_dict)
                
                logger.dump_tabular()
                print(("Current Task: %d Total updates: %d Total Episode: %d Total Timesteps %d Episode T: %d Reward: %f") %
                        (task_id, total_updates, total_episodes, total_timesteps, data['episode_timestep'],
                        data['episode_reward']))

            #######
            # run eval
            #######
            if updates_since_eval >= args.eval_freq:
                updates_since_eval %= args.eval_freq
                evaluate_policy_on_all_tasks_cl(alg, args, envs, task_id, total_timesteps, total_updates, 
                                             wrt_csv_eval=wrt_csv_eval, wandb=wandb)
            
            #######
            # save for every interval-th update or for the last epoch
            #######
            if (updates_since_saving >= args.save_freq or update_num > updates_per_task - 1):
                updates_since_saving %= args.save_freq
                take_snapshot(args, ck_fname_part, alg, total_updates)
            
        ###############
        # Eval at task completion
        ###############
        print(f"Training on task {task_id} done. Eval starts... ")
        current_performance = evaluate_policy_on_all_tasks_cl(alg, args, envs, task_id, total_timesteps, 
                total_updates, wrt_csv_eval_final=wrt_csv_eval_final, wandb=wandb)
        avg_current_eprew.append(current_performance['current_eprew_mean'])
        if args.monitor_success:
            avg_current_success.append(current_performance['current_success_rate'])
        
    ###############
    # All done
    ###############
    wrt_csv_eval.close()
    wrt_csv_eval_final.close()
    total_runtime = time.time() - tstart
    if wandb is not None:
        wandb.log({'total_runtime': total_runtime})
        wandb.log({'final/avg_current_eprew': np.mean(avg_current_eprew)})
        if args.monitor_success:
            wandb.log({'final/avg_current_success': np.mean(avg_current_success)})

    print(f'All done. total elapsed time {total_runtime}')