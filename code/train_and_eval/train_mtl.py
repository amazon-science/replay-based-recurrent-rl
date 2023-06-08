import time
import numpy as np
from collections import deque
from misc.utils import (
    CSVWriter,
    safemean,
    take_snapshot,
)
from oailibs import logger
from misc.runner_offpolicy import Runner
from train_and_eval.eval import evaluate_policy_on_all_tasks_mtl


def train_mtl(
    args,
    envs,
    alg,
    fname_csv_eval,
    fname_csv_eval_final,
    ck_fname_part,
    replay_buffer,
    device,
    logger,
    wandb,
):
    # define some vars
    total_timesteps = 0
    total_episodes = 0
    total_updates = 0
    cumulative_rewards = 0

    # init loggers
    log_dict = {
        "total_updates": -1,
        "total_timesteps": -1,
        "total_episodes": -1,
        "eval/global_eprew_mean": -1,
        "eval/global_eprew_std": -1,
    }
    if args.monitor_success:
        log_dict.update({"eval/global_success_mean": -1, "eval/global_success_std": -1})
    wrt_csv_eval = CSVWriter(fname_csv_eval, log_dict)

    log_dict = {
        "total_updates": -1,
        "total_timesteps": -1,
        "final/global_eprew_mean": -1,
        "final/global_eprew_std": -1,
    }
    if args.monitor_success:
        log_dict.update(
            {"final/global_success_mean": -1, "final/global_success_std": -1,}
        )
    wrt_csv_eval_final = CSVWriter(fname_csv_eval_final, log_dict)

    ############################################################################
    # Main loop train
    ############################################################################
    # Start total timer
    tstart = time.time()
    print("Start main loop ...")

    current_env_id = 0
    env = envs[current_env_id]
    env.reset()

    #### variables
    total_episodes = 0
    total_updates = 0
    total_timesteps = 0
    updates_since_eval, updates_since_saving, updates_since_logging = 3 * [0]
    warm_up_passed = False
    epinfobuf = deque(maxlen=100)
    epinfobuf_v2 = deque(maxlen=args.num_evals)
    tot_time_sampling = 0.0
    tot_time_buffer_sampling = 0.0
    tot_time_transfer = 0.0
    tot_time_updating = 0.0
    updates_multiplier = args.total_updates / args.total_timesteps

    ## rollout/batch generator
    rollouts = Runner(
        env=envs,
        model=alg,
        replay_buffer=replay_buffer,
        burn_in=args.burn_in,
        total_timesteps=args.total_timesteps,
        history_length=args.history_length,
        nb_tasks=args.nb_tasks,
        multi_task=True,
        device=device,
    )

    # Evaluate starting policy
    evaluate_policy_on_all_tasks_mtl(
        alg,
        args,
        envs,
        total_timesteps,
        total_updates,
        wrt_csv_eval=wrt_csv_eval,
        wandb=wandb,
    )

    ##############################
    # task train loop
    ##############################
    # start task timer
    training_start = time.time()

    while total_timesteps < args.total_timesteps:
        #######
        # Interact and collect data until reset
        #######
        sampling_start = time.time()
        data = rollouts.run(total_timesteps, total_episodes)
        tot_time_sampling += time.time() - sampling_start
        updates_to_run = data["episode_timestep"]
        cumulative_rewards += data["episode_reward"]
        epinfobuf.extend(data["epinfos"])
        epinfobuf_v2.extend(data["epinfos"])
        total_timesteps += data["episode_timestep"]
        total_episodes += 1

        if total_timesteps >= args.warm_up:
            warm_up_passed = True
        else:
            ## still in warm_up
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
        updates_since_eval += updates_to_run
        updates_since_logging += updates_to_run
        updates_since_saving += updates_to_run

        #######
        # logging
        #######
        nseconds = time.time() - training_start
        # Calculate the fps (frame per second)
        fps = int((total_timesteps) / nseconds)

        if updates_since_logging >= args.log_freq:
            updates_since_logging %= args.log_freq
            log_dict = {
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
            log_dict["code/time_sampling"] = tot_time_sampling / total_episodes
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
                ("updates: %d Episode Num: %d  Episode T: %d Reward: %f")
                % (
                    total_updates,
                    total_episodes,
                    data["episode_timestep"],
                    data["episode_reward"],
                )
            )

        #######
        # run eval
        #######
        if updates_since_eval >= args.eval_freq:
            updates_since_eval %= args.eval_freq
            evaluate_policy_on_all_tasks_mtl(
                alg,
                args,
                envs,
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
            or total_updates > args.total_updates - 1
        ):
            updates_since_saving %= args.save_freq
            take_snapshot(args, ck_fname_part, alg, total_updates)

    ###############
    # Eval at training completion
    ###############
    print(f"Training done. Eval starts... ")
    evaluate_policy_on_all_tasks_mtl(
        alg,
        args,
        envs,
        total_episodes,
        total_updates,
        wrt_csv_eval_final=wrt_csv_eval_final,
        wandb=wandb,
    )

    wrt_csv_eval.close()
    total_runtime = time.time() - tstart
    if wandb is not None:
        wandb.log({"total_runtime": total_runtime})

    print(f"All done. total elapsed time {total_runtime}")
