import numpy as np
import time
from collections import deque
from misc.runner_offpolicy import Runner
from train_and_eval.eval import evaluate_policy_on_all_tasks_cl
from misc.utils import (
    CSVWriter,
    safemean,
    take_snapshot,
)
import algs.SAC.sac as malg


def train_cl(
    args,
    envs,
    alg,
    actor_net,
    critic_net,
    actor_args,
    critic_args,
    alg_args,
    fname_csv_eval,
    fname_csv_eval_final,
    ck_fname_part,
    replay_buffer,
    ActorSAC,
    CriticSAC,
    device,
    logger,
    wandb,
):
    ## define some vars
    total_timesteps = 0
    total_episodes = 0
    total_updates = 0
    cumulative_rewards = 0
    avg_current_eprew = []
    if args.monitor_success:
        avg_current_success = []
    tot_time_sampling = 0.0
    tot_time_buffer_sampling = 0.0
    tot_time_transfer = 0.0
    tot_time_updating = 0.0
    updates_multiplier = args.total_updates / args.total_timesteps
    timesteps_per_task = int(args.total_timesteps / args.nb_tasks)
    updates_per_task = int(args.total_updates / args.nb_tasks)

    # init loggers
    log_dict = {
        "current_task": -1,
        "total_updates": -1,
        "total_episodes": -1,
        "total_timesteps": -1,
        "eval/past_eprew_mean": -1,
        "eval/current_eprew_mean": -1,
        "eval/global_eprew_mean": -1,
        "eval/global_eprew_std": -1,
    }
    if args.monitor_success:
        log_dict.update(
            {
                "eval/past_success_mean": -1,
                "eval/current_success_mean": -1,
                "eval/global_success_mean": -1,
                "eval/global_success_std": -1,
            }
        )
    wrt_csv_eval = CSVWriter(fname_csv_eval, log_dict)

    log_dict = {
        "current_task": -1,
        "total_updates": -1,
        "total_timesteps": -1,
        "final/current_eprew_mean": -1,
        "final/past_eprew_mean": -1,
        "final/global_eprew_mean": -1,
        "final/global_eprew_std": -1,
    }
    if args.monitor_success:
        log_dict.update(
            {
                "final/current_success_mean": -1,
                "final/past_success_mean": -1,
                "final/global_success_mean": -1,
                "final/global_success_std": -1,
            }
        )
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
        updates_since_eval, updates_since_saving, updates_since_logging = 3 * [0]
        warm_up_passed = False
        epinfobuf = deque(maxlen=100)
        epinfobuf_v2 = deque(maxlen=args.num_evals)

        ## training from train_from_scratch
        if args.train_from_scratch:
            print("re-initializing the neural network weights\n")
            del actor_net, critic_net, alg, alg_args["actor"], alg_args["critic"]
            actor_net = ActorSAC(**actor_args).to(device)
            critic_net = CriticSAC(**critic_args).to(device)
            alg_args["actor"], alg_args["critic"] = actor_net, critic_net
            alg = malg.SAC(**alg_args)

        ## only reset buffer if ER isn't enabled
        if not args.experience_replay or args.train_from_scratch:
            print("resetting the replay buffer")
            replay_buffer.reset()

        ## rollout/batch generator
        rollouts = Runner(
            env=env,
            model=alg,
            replay_buffer=replay_buffer,
            burn_in=args.burn_in,
            total_timesteps=int(args.total_timesteps / args.nb_tasks),
            history_length=args.history_length,
            device=device,
        )

        # Evaluate starting policy
        evaluate_policy_on_all_tasks_cl(
            alg,
            args,
            envs,
            task_id,
            total_timesteps,
            total_updates,
            wrt_csv_eval=wrt_csv_eval,
            wandb=wandb,
        )

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
            updates_to_run = data["episode_timestep"]
            cumulative_rewards += data["episode_reward"]
            epinfobuf.extend(data["epinfos"])
            epinfobuf_v2.extend(data["epinfos"])
            total_episodes += 1
            episode_num += 1
            total_timesteps += data["episode_timestep"]
            timestep_num += data["episode_timestep"]

            if timestep_num >= args.warm_up:
                warm_up_passed = True
            else:
                ## still in warmup
                continue

            #######
            # run training to calculate loss, run backward, and update params
            #######
            updates_to_run = int(updates_to_run * updates_multiplier)
            alg_stats, extra_alg_stats = alg.train(
                replay_buffer=replay_buffer, iterations=updates_to_run,
            )
            tot_time_buffer_sampling += sum(extra_alg_stats["time_buffer_sampling"])
            tot_time_transfer += sum(extra_alg_stats["time_transfer"])
            tot_time_updating += sum(extra_alg_stats["time_updating"])
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
            fps = int((timestep_num) / nseconds)

            if updates_since_logging >= args.log_freq:
                updates_since_logging %= args.log_freq
                log_dict = {
                    "current_task": task_id,
                    "total_updates": total_updates,
                    "total_timesteps": total_timesteps,
                    "total_episodes": total_episodes,
                    "train/episode_reward": float(data["episode_reward"]),
                    "train/eprewmean": float(
                        safemean([epinfo["r"] for epinfo in epinfobuf])
                    ),
                    "train/eplenmean": float(
                        safemean([epinfo["l"] for epinfo in epinfobuf])
                    ),
                    "train/eprewmeanV2": float(
                        safemean([epinfo["r"] for epinfo in epinfobuf_v2])
                    ),
                    "train/eplenmeanV2": float(
                        safemean([epinfo["l"] for epinfo in epinfobuf_v2])
                    ),
                    "train/cumul_rew": cumulative_rewards,
                }

                for key in alg_stats:
                    log_dict["train/" + key] = np.mean(alg_stats[key])

                if args.monitor_grads:
                    for key in extra_alg_stats["critic_grad_summary"]:
                        log_dict["gradients/critic_" + key] = extra_alg_stats[
                            "critic_grad_summary"
                        ][key]

                    for key in extra_alg_stats["actor_grad_summary"]:
                        log_dict["gradients/actor_" + key] = extra_alg_stats[
                            "actor_grad_summary"
                        ][key]

                ## add monitoring of code performance
                log_dict["code/fps"] = fps
                log_dict["code/tot_time_sampling"] = tot_time_sampling
                log_dict["code/time_sampling"] = tot_time_sampling / episode_num
                log_dict["code/tot_time_buffer_sampling"] = tot_time_buffer_sampling
                log_dict["code/time_buffer_sampling"] = (
                    tot_time_buffer_sampling / total_updates
                )
                log_dict["code/tot_time_transfer"] = tot_time_transfer
                log_dict["code/time_transfer"] = tot_time_transfer / total_updates
                log_dict["code/tot_time_updating"] = tot_time_updating
                log_dict["code/time_updating"] = tot_time_updating / total_updates

                for key, value in log_dict.items():
                    logger.record_tabular(key, value)

                if wandb is not None:
                    wandb.log(log_dict)

                logger.dump_tabular()
                print(
                    (
                        "Current Task: %d Total updates: %d Total Episode: %d Total Timesteps %d Episode T: %d Reward: %f"
                    )
                    % (
                        task_id,
                        total_updates,
                        total_episodes,
                        total_timesteps,
                        data["episode_timestep"],
                        data["episode_reward"],
                    )
                )

            #######
            # run eval
            #######
            if updates_since_eval >= args.eval_freq:
                updates_since_eval %= args.eval_freq
                evaluate_policy_on_all_tasks_cl(
                    alg,
                    args,
                    envs,
                    task_id,
                    total_timesteps,
                    total_updates,
                    wrt_csv_eval=wrt_csv_eval,
                    wandb=wandb,
                )

            #######
            # save for every interval-th update or for the last epoch
            #######
            if (
                updates_since_saving >= args.save_freq
                or update_num > updates_per_task - 1
            ):
                updates_since_saving %= args.save_freq
                take_snapshot(args, ck_fname_part, alg, total_updates)

        ###############
        # Eval at task completion
        ###############
        print(f"Training on task {task_id} done. Eval starts... ")
        current_performance = evaluate_policy_on_all_tasks_cl(
            alg,
            args,
            envs,
            task_id,
            total_timesteps,
            total_updates,
            wrt_csv_eval_final=wrt_csv_eval_final,
            wandb=wandb,
        )
        avg_current_eprew.append(current_performance["current_eprew_mean"])
        if args.monitor_success:
            avg_current_success.append(current_performance["current_success_rate"])

    ###############
    # All done
    ###############
    wrt_csv_eval.close()
    wrt_csv_eval_final.close()
    total_runtime = time.time() - tstart
    if wandb is not None:
        wandb.log({"total_runtime": total_runtime})
        wandb.log({"final/avg_current_eprew": np.mean(avg_current_eprew)})
        if args.monitor_success:
            wandb.log({"final/avg_current_success": np.mean(avg_current_success)})

    print(f"All done. total elapsed time {total_runtime}")
