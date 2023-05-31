import argparse
import torch
import os
import numpy as np
import random
from misc.utils import (
    create_dir,
    overwrite_args,
    str2bool,
    restricted_float,
    print_model_info,
    take_snapshot,
    setup_logAndCheckpoints,
)
from oailibs import logger
from misc.buffer import Buffer
from datetime import date
from train_and_eval.train_cl import train_cl
from train_and_eval.train_mtl import train_mtl

parser = argparse.ArgumentParser()

# Optim params
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--replay_size", type=int, default=1e6, help="Replay buffer size int(1e6)"
)
parser.add_argument(
    "--ptau", type=float, default=0.005, help="Interpolation factor in polyak averaging"
)
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor [0,1]")
parser.add_argument(
    "--burn_in",
    default=1000,
    type=int,
    help="How many timesteps purely random policy is run for",
)
parser.add_argument(
    "--warm_up",
    default=100,
    type=int,
    help="How many timesteps before updating the models",
)
parser.add_argument(
    "--batch_size", default=100, type=int, help="Batch size for both actor and critic"
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=64,
    help="indicates hidden size for actor and critics",
)
parser.add_argument(
    "--num_hidden_layers",
    type=int,
    default=2,
    help="indicates the number of hidden layers. set to 0 for linear",
)

parser.add_argument(
    "--grad_clip", type=float, default=None, help="clip the gradient norm"
)

# General params
parser.add_argument(
    "--train_mode",
    type=str,
    default="cl",
    choices=["cl", "mtl"],
    help="continual learning (cl) or multi-task learning (mtl)",
)
parser.add_argument(
    "--env_name",
    type=str,
    default="Ant_direction-v3",
    choices=["Quadratic_opt", "Ant_direction-v3", "MT10", "MT50", "CW10"],
)
parser.add_argument(
    "--obs_dim",
    type=int,
    default=2,
    help="number of dimension in the quadratic opt problem",
)
parser.add_argument(
    "--monitor_success", type=str2bool, default=False, help="monitor the success rate"
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=100,
    help="to overwrite the default _max_episode_steps",
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--alg_name", type=str, default="sac")
parser.add_argument(
    "--total_timesteps",
    default=1e4,
    type=int,
    help="number of timesteps to train on per CL task",
)
parser.add_argument(
    "--total_updates",
    default=None,
    type=int,
    help="if None, will be equal to total_timesteps",
)

parser.add_argument("--disable_cuda", type=str2bool, default=False)
parser.add_argument("--cuda_deterministic", default=False, action="store_true")
parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--set_num_threads", type=int, default=1, help="set_num_threads")

parser.add_argument("--log_id", default="dummy")
parser.add_argument(
    "--record_policy", default=False, help="for recording the policies evaluation (WIP)"
)
parser.add_argument("--check_point_dir", default="./ck")
parser.add_argument("--log_dir", default="./log_dir")
parser.add_argument(
    "--wandb_project",
    type=str,
    default=None,
    help="name of the WandB workspace. Set to None for no WandB logging",
)
parser.add_argument("--num_evals", type=int, default=25, help="Length eval episode")
parser.add_argument(
    "--log_freq", type=int, default=1e2, help="how often (updates) we log results"
)
parser.add_argument(
    "--eval_freq", type=int, default=1e4, help="How often (updates) we evaluate"
)
parser.add_argument(
    "--save_freq", type=int, default=1e5, help="how often (updates) we save"
)
parser.add_argument(
    "--monitor_grads", type=str2bool, default=True, help="monitor gradient statistics"
)

## sac parameters
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--use_auto_entropy", type=str2bool, default=False)

## context
parser.add_argument(
    "--context_id", type=str2bool, default=False, help="add task_id to the observations"
)
parser.add_argument(
    "--context_embedding",
    type=str2bool,
    default=False,
    help="learn a task embedding + add to observations",
)
parser.add_argument(
    "--context_rnn",
    type=str2bool,
    default=False,
    help="add a context RNN to the observations",
)
parser.add_argument(
    "--context_transformer",
    type=str2bool,
    default=False,
    help="add a context transformer to the observations",
)
parser.add_argument(
    "--context_dim",
    type=int,
    default=30,
    help="indicates the context dimension (for embedding, rnn and transformer)",
)
parser.add_argument(
    "--tx_hidden_dim",
    type=int,
    default=10,
    help="indicates the hidden size of the context transformer",
)
parser.add_argument(
    "--tx_nb_heads",
    type=int,
    default=2,
    help="nb of heads for the context transformer",
)
parser.add_argument(
    "--tx_token_emb",
    type=bool,
    default=True,
    help="learn token embeddings in the context transformer",
)
parser.add_argument(
    "--tx_pos_enc",
    type=str,
    default="sinusoidal",
    choices=["sinusoidal", "learned"],
    help="either learn the positional embeddings or use the sinusoidal encoding",
)

parser.add_argument("--history_length", type=int, default=30)
parser.add_argument(
    "--multi_head",
    type=str2bool,
    default=False,
    help="use a seperate head for each task",
)
parser.add_argument(
    "--task_agnostic",
    type=str2bool,
    default=False,
    help="enforces task agnosticiy (mainly for TAMH)",
)

## cl setting
parser.add_argument("--nb_tasks", type=int, default=20, help="number of CL tasks")
parser.add_argument(
    "--single_task_id", type=int, default=None, help="task id for single task exp"
)
parser.add_argument(
    "--episodes_per_task",
    type=int,
    default=1000,
    help="number of allowed episode per task",
)

## cl method
parser.add_argument(
    "--experience_replay",
    type=str2bool,
    default=False,
    help="if on, the replay_buffer doesnt reset",
)
parser.add_argument(
    "--curr_task_sampl_prob",
    type=restricted_float,
    default=0.0,
    help="prob to sample current task data in ExpReplay. If none, it stays as uniform sampling",
)
parser.add_argument(
    "--train_from_scratch",
    type=str2bool,
    default=False,
    help="if on, reset the neural network weight after each tasks",
)

## for args overwritting
parser.add_argument(
    "--setting_config",
    type=str,
    default=None,
    help="setting name for configs overwritting",
)
parser.add_argument(
    "--method_config",
    type=str,
    default=None,
    help="method name for configs overwritting",
)
parser.add_argument(
    "--hparam_config",
    type=str,
    default=None,
    help="hparam configuration name for configs overwritting",
)

if __name__ == "__main__":
    ##############################
    #### Args
    ##############################
    args = parser.parse_args()
    if args.setting_config:
        args = overwrite_args(args, "settings", args.setting_config)
    if args.method_config:
        args = overwrite_args(args, "methods", args.method_config)
    if args.hparam_config:
        args = overwrite_args(args, "hparams", args.hparam_config)

    assert args.burn_in < args.total_timesteps
    assert args.warm_up < args.total_timesteps

    if args.train_from_scratch:
        assert not args.multi_head

    args.total_updates = (
        args.total_timesteps if args.total_updates is None else args.total_updates
    )

    if args.train_mode == "mtl":
        args.experience_replay = False

    if not (args.context_rnn or args.context_transformer) and args.history_length > 0:
        print("WARNING: no RNN, setting history_length to 0")
        args.history_length = 0

    if args.context_transformer and args.tx_nb_heads > 1:
        if args.tx_token_emb:
            assert args.tx_hidden_dim % args.tx_nb_heads == 0
        else:
            print(
                "WARNING: no token encoding, setting nb_heads to 1 as the transformer token might be of an odd length"
            )
            args.tx_nb_heads = 1

    print("------------")
    print(args.__dict__)
    print("------------")

    ##############################
    #### Generic setups
    ##############################
    CUDA_AVAL = torch.cuda.is_available()

    if not args.disable_cuda and CUDA_AVAL:
        gpu_id = "cuda:" + str(args.gpu_id)
        device = torch.device(gpu_id)
        print("**** Yayy we use GPU %s ****" % gpu_id)

    else:
        device = torch.device("cpu")
        print("**** No GPU detected or GPU usage is disabled, sorry! ****")

    if not args.disable_cuda and CUDA_AVAL and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("****** cudnn.deterministic is set ******")

    torch.set_num_threads(args.set_num_threads)

    ##############################
    # Create log folders, etc.
    ##############################
    create_dir(args.log_dir, cleanup=True)
    ## create folder for save checkpoints
    ck_fname_part, log_file_dir, fname_csv_eval = setup_logAndCheckpoints(args)
    fname_csv_eval_final = os.path.join(log_file_dir, "eval_final.csv")
    logger.configure(dir=log_file_dir)
    wrt_csv_eval = None

    ##############################
    # Wandb setup
    ##############################
    if args.wandb_project is not None:
        # NOTE: uncomment for dryrun
        # os.environ["WANDB_MODE"] = "dryrun"
        import wandb
        wandb.init(project=args.wandb_project, name=args.log_id)
        wandb.config.update(args)
        today = date.today()
        wandb.log({"launch_date": int(today.strftime("%Y%m%d"))})
    else:
        wandb = None

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
    envs = get_envs(
        env_name=args.env_name,
        nb_tasks=args.nb_tasks,
        train_seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        eparams=args,
    )
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
    context_input_dim = None
    actor_idim = env.observation_space.shape[0]

    if args.context_id:  # means feed task_id to actor critic
        actor_idim += args.nb_tasks
        dim_others += args.nb_tasks

    if args.context_embedding:  # means we use the task-id to learn a task embedding
        actor_idim += args.context_dim
        dim_others += args.context_dim

    if args.context_rnn or args.context_transformer:
        context_input_dim = (
            env.action_space.shape[0] + reward_dim + env.observation_space.shape[0]
        )
        actor_idim += args.context_dim
        dim_others += args.context_dim

    ######
    # now we build the actor and critic
    ######
    m_list_p = []
    if len(env.observation_space.shape) == 1:
        if str.lower(args.alg_name) == "sac":
            from models.networks import ActorSAC, CriticSAC

            actor_args = {
                "action_space": env.action_space,
                "hidden_dim": args.hidden_dim,
                "num_hidden_layers": args.num_hidden_layers,
                "input_dim": actor_idim,
                "max_action": max_action,
                "context_id": args.context_id,
                "context_embedding": args.context_embedding,
                "context_rnn": args.context_rnn,
                "context_transformer": args.context_transformer,
                "nb_tasks": args.nb_tasks,
                "context_input_dim": context_input_dim,
                "tx_hidden_dim": args.tx_hidden_dim,
                "context_dim": args.context_dim,
                "history_length": args.history_length,
                "obsr_dim": env.observation_space.shape[0],
                "device": device,
                "eparams": args,
            }
            actor_net = ActorSAC(**actor_args).to(device)

            critic_args = {
                "action_space": env.action_space,
                "hidden_dim": args.hidden_dim,
                "num_hidden_layers": args.num_hidden_layers,
                "input_dim": env.observation_space.shape[0],
                "context_id": args.context_id,
                "context_embedding": args.context_embedding,
                "context_rnn": args.context_rnn,
                "context_transformer": args.context_transformer,
                "nb_tasks": args.nb_tasks,
                "context_input_dim": context_input_dim,
                "tx_hidden_dim": args.tx_hidden_dim,
                "context_dim": args.context_dim,
                "dim_others": dim_others,
                "history_length": args.history_length,
                "obsr_dim": env.observation_space.shape[0],
                "device": device,
                "eparams": args,
            }
            critic_net = CriticSAC(**critic_args).to(device)

        else:
            raise ValueError(
                "%s model is not supported for %s env"
                % (args.env_name, env.observation_space.shape)
            )

        m_list_p.append(actor_net)
        m_list_p.append(critic_net)

    total_params, _ = print_model_info(m_list_p)
    print("-----------------------------")
    print("Name of env:", args.env_name)
    print("Observation_space:", env.observation_space)
    print("Action space:", env.action_space)
    print("----------------------------")

    ##############################
    # Algs setup
    ##############################
    if str.lower(args.alg_name) == "sac":
        import algs.SAC.sac as malg

        alg_args = {
            "actor": actor_net,
            "critic": critic_net,
            "lr": args.lr,
            "gamma": args.gamma,
            "ptau": args.ptau,
            "batch_size": args.batch_size,
            "max_action": max_action,
            "alpha": args.alpha,
            "use_auto_entropy": args.use_auto_entropy,
            "action_dims": env.action_space.shape,
            "history_length": args.history_length,
            "grad_clip": args.grad_clip,
            "monitor_grads": args.monitor_grads,
            "device": device,
        }
        alg = malg.SAC(**alg_args)

    else:
        raise ValueError("%s alg is not supported" % args.alg_name)

    ## init replay buffer
    replay_buffer = Buffer(
        max_size=args.replay_size,
        experience_replay=args.experience_replay,
        curr_task_sampl_prob=args.curr_task_sampl_prob,
        history_length=args.history_length,
    )

    ## log memory consumption
    # FIXME:
    sample_size = (
        env.observation_space.shape[0]
        + env.action_space.shape[0]
        + 1
        + env.observation_space.shape[0]
    )  # SARS'
    replay_buffer_size = args.replay_size * sample_size

    if wandb is not None:
        log_dict = {
            "total_params": total_params,
            "replay_buffer_size": replay_buffer_size,
            "total_memory": total_params + replay_buffer_size,
        }
        wandb.log(log_dict)

    take_snapshot(args, ck_fname_part, alg, 0)

    if args.train_mode == "cl":
        train_cl(
            args=args,
            envs=envs,
            alg=alg,
            actor_net=actor_net,
            critic_net=critic_net,
            actor_args=actor_args,
            critic_args=critic_args,
            alg_args=alg_args,
            fname_csv_eval=fname_csv_eval,
            fname_csv_eval_final=fname_csv_eval_final,
            ck_fname_part=ck_fname_part,
            replay_buffer=replay_buffer,
            ActorSAC=ActorSAC,
            CriticSAC=CriticSAC,
            device=device,
            logger=logger,
            wandb=wandb,
        )
    elif args.train_mode == "mtl":
        train_mtl(
            args=args,
            envs=envs,
            alg=alg,
            fname_csv_eval=fname_csv_eval,
            fname_csv_eval_final=fname_csv_eval_final,
            ck_fname_part=ck_fname_part,
            replay_buffer=replay_buffer,
            device=device,
            logger=logger,
            wandb=wandb,
        )
