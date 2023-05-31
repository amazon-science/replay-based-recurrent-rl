from __future__ import print_function, division
from ast import Not
from re import I
from tkinter import N
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import copy
import time
from misc.autograd_hacks import compute_grad1, clear_backprops
from misc import autograd_hacks


class SAC:
    def __init__(
        self,
        actor,
        critic,
        lr=1e-3,
        gamma=0.99,
        ptau=0.005,
        batch_size=100,
        optim_method="",
        max_action=None,
        alpha=0.2,
        use_auto_entropy=False,
        action_dims=None,
        history_length=0,
        grad_clip=None,
        monitor_grads=False,
        device="cpu",
    ):
        """
        actor:  actor network
        critic: critic network
        lr:   learning rate for RMSProp
        gamma: reward discounting parameter
        ptau:  Interpolation factor in polyak averaging
        policy_noise: add noise to policy
        noise_clip: clipped noise
        policy_freq: delayed policy updates
        alpha: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)
        """
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.ptau = ptau
        self.max_action = max_action
        self.batch_size = batch_size
        self.hist_len = history_length
        self.grad_clip = grad_clip
        self.monitor_grads = monitor_grads
        self.device = device

        # sac parmas
        self.alpha = alpha

        # load target models.
        self.critic_target = copy.deepcopy(self.critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        ## automatic entropy tuning
        self.use_auto_entropy = use_auto_entropy
        if self.use_auto_entropy == True:
            self.target_entropy = -np.prod(action_dims).item()  # a trick from sac paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        print("-----------------------------")
        print("Optim Params")
        print("Actor:\n ", self.actor_optimizer)
        print("Critic:\n ", self.critic_optimizer)
        if self.use_auto_entropy:
            print("Alpha:\n ", self.alpha_optimizer)
        print("-----------------------------")

        if self.monitor_grads:
            autograd_hacks.add_hooks(self.critic)
            autograd_hacks.add_hooks(self.actor)

    def select_action(
        self,
        task_id,
        obs,
        previous_action,
        previous_reward,
        previous_obs,
        deterministic=False,
        eval_step=False,
        ret_context=False,
    ):
        """
        return action
        """
        with torch.no_grad():
            # obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
            task_id = torch.LongTensor(task_id).to(self.device)
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
            if self.hist_len > 0:
                previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(
                    self.device
                )
                previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(
                    self.device
                )
                previous_obs = torch.FloatTensor(previous_obs.reshape(1, -1)).to(
                    self.device
                )

                # combine all other data here before send them to actor
                # torch.cat([previous_action, previous_reward], dim = -1)
                pre_act_rew = [previous_action, previous_reward, previous_obs]
            else:
                pre_act_rew = None

            out = self.actor(
                task_id,
                obs,
                pre_act_rew,
                deterministic=deterministic,
                with_logprob=False,
                ret_context=ret_context,
            )

            a = out[0].cpu().data.numpy().flatten()

        if not ret_context:
            return a
        else:
            return a, out[2].cpu().data.numpy().flatten()

    def compute_critic_loss(
        self,
        task_id,
        obs,
        next_obs,
        action,
        reward,
        mask,
        pre_act_rew,
        act_rew,
        last_time,
    ):
        """
        Compute loss Q:
        t =  r + gamma * mask *( min_{1,2} Q_target(s', a') - alpha * log pi(s', a'))
        """
        with torch.no_grad():
            ########
            # Target actions come from *current* policy
            ########
            # next_action [B, num_actions], logp_next_actions [B, 1]
            next_action, logp_next_actions, _ = self.actor(task_id, next_obs, act_rew)
            ########
            #  Update critics
            #  1. Compute the target Q value
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the critic
            ########
            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            # target_Qs [B, 1]
            target_Q1, target_Q2, extras1, extras2 = self.critic_target(
                task_id, next_obs, next_action, act_rew
            )
            target_Q = torch.min(target_Q1, target_Q2)

            # backup: [B, 1]
            backup = reward + self.gamma * mask * (
                target_Q - self.alpha * logp_next_actions
            )

        # 2.  Get current Q estimates
        current_Q1, current_Q2, extras1, extras2 = self.critic(
            task_id, obs, action, pre_act_rew
        )

        # 3. Compute critic loss
        # even we picked min Q, we still need to backprob to both Qs
        critic_loss = ((current_Q1 - backup) ** 2).mean() + (
            (current_Q2 - backup) ** 2
        ).mean()

        # Useful info for logging
        q_info = dict(
            Q1Vals=current_Q1.squeeze(-1).detach().cpu().numpy(),
            Q2Vals=current_Q2.squeeze(-1).detach().cpu().numpy(),
        )

        if last_time:
            q_info["current_Q1"] = current_Q1
            q_info["current_Q2"] = current_Q2

        return critic_loss, q_info

    def compute_loss_pi(self, task_id, obs, pre_act_rew, last_time):
        """
        Compute pi loss
           loss = alpha * log_pi - min_{1,2} Q(s, pi)
        """
        pi_action, logp_pi, extras = self.actor(task_id, obs, pre_act_rew)
        q1_pi, q2_pi, _, _ = self.critic(task_id, obs, pi_action, pre_act_rew)
        q_pi = torch.min(q1_pi, q2_pi)

        #####
        # Alpha loss
        #####
        alpha_loss_out = 0
        if self.use_auto_entropy:
            alpha_loss = -(
                self.log_alpha * (logp_pi + self.target_entropy).detach()
            ).mean()
            alpha_loss_out = alpha_loss.item()

            # run optim
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # now get value of alpha
            self.alpha = self.log_alpha.exp()

        # Entropy-regularized policy loss
        # logp_pi,  q_pi: [B, 1]
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.squeeze(-1).detach().cpu().numpy(),)

        if last_time:
            pi_info["logp_pi"] = logp_pi

        return loss_pi, alpha_loss_out, pi_info

    def train(
        self, replay_buffer=None, iterations=None,
    ):
        """
        Runs mutiple update iterations w/ self.batch_size
            inputs:
                replay_buffer
            outputs:
                out (includes results)
        """

        out = {
            "actor_loss": [],
            "critic_loss": [],
            "alpha_loss": [],
            "alpha": [],
            "Q1Vals": [],
            "Q2Vals": [],
            "LogPi": [],
        }
        extras = {
            "time_buffer_sampling": [],
            "time_transfer": [],
            "time_updating": [],
        }

        critic_grads_epoch = None
        actor_grads_epoch = None

        for it in range(iterations):
            last_time = it == iterations - 1

            ########
            # Sample replay buffer
            ########
            start_buffer_sampling = time.time()
            data = replay_buffer.sample(self.batch_size)
            extras["time_buffer_sampling"].append(time.time() - start_buffer_sampling)

            start_transfer = time.time()
            task_id = torch.LongTensor(data["task_id"]).to(self.device)
            obs = torch.FloatTensor(data["obs"]).to(self.device)
            next_obs = torch.FloatTensor(data["next_obs"]).to(self.device)
            action = torch.FloatTensor(data["action"]).to(self.device)
            reward = torch.FloatTensor(data["reward"]).to(self.device)
            mask = torch.FloatTensor(1 - data["done"]).to(self.device)

            if self.hist_len > 0:
                previous_actions = torch.FloatTensor(data["previous_actions"]).to(
                    self.device
                )
                previous_rewards = torch.FloatTensor(data["previous_rewards"]).to(
                    self.device
                )
                previous_obs = torch.FloatTensor(data["previous_obs"]).to(self.device)

                # list of hist_actions and hist_rewards which are one time ahead of previous_ones
                # example:
                # previous_action = [t-3, t-2, t-1]
                # hist_actions    = [t-2, t-1, t]
                hist_actions = torch.FloatTensor(data["current_actions"]).to(
                    self.device
                )
                hist_rewards = torch.FloatTensor(data["current_rewards"]).to(
                    self.device
                )
                hist_obs = torch.FloatTensor(data["current_obs"]).to(self.device)

                # combine actions, rewards and obs
                act_rew = [
                    hist_actions,
                    hist_rewards,
                    hist_obs,
                ]  # torch.cat([action, reward], dim = -1)
                pre_act_rew = [
                    previous_actions,
                    previous_rewards,
                    previous_obs,
                ]  # torch.cat([previous_action, previous_reward], dim = -1)
            else:
                act_rew = None
                pre_act_rew = None
            extras["time_transfer"].append(time.time() - start_transfer)

            start_updating = time.time()

            ########
            # critic updates
            ########
            # 1. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss, q_info = self.compute_critic_loss(
                task_id,
                obs,
                next_obs,
                action,
                reward,
                mask,
                pre_act_rew,
                act_rew,
                last_time,
            )

            def accumulate_grads_epoch(model, grads_epoch):
                grads = None
                for p in model.parameters():
                    grad = torch.clone(p.grad).detach().flatten()
                    if grads is None:
                        grads = grad
                    else:
                        grads = torch.cat((grads, grad), axis=0)

                if grads_epoch is None:
                    grads_epoch = dict(sum=grads, abs_sum=grads.abs(),)
                else:
                    grads_epoch["sum"] += grads
                    grads_epoch["abs_sum"] += grads.abs()

                return grads_epoch

            def gradient_analysis(model, grads_epoch):
                ## be mindfull we are only getting the Linear layers (no RNN, MH and module_select)
                grads = None
                for p in model.parameters():
                    if hasattr(p, "grad1"):
                        grad = torch.clone(p.grad1).detach()
                        grad = grad.view(grad.shape[0], -1)
                        if grads is None:
                            grads = grad

                        else:
                            grads = torch.cat((grads, grad), axis=1)

                ## tested to verify autograd_hacks was working properly
                # assert all(torch.isclose(grads_epoch[-1], grads.mean(0)))
                ## for RNN:
                # assert all(torch.isclose(grads_epoch[-1][:grads.mean(0).shape[0]], grads.mean(0)))

                grad_norm = torch.norm(grads, dim=1)
                grads_normalized1 = grads / (grad_norm + 1e-6).unsqueeze(1)
                grads_normalized2 = grads / (grad_norm.mean() + 1e-6)

                ## analysing throughout the epoch
                grads_epoch_sum = grads_epoch["sum"].abs()
                grads_epoch_sum += 1e-6
                p = grads_epoch_sum / grads_epoch_sum.sum()
                grads_epoch_entropy = -(p * torch.log(p)).sum()

                grads_epoch_abs_sum = grads_epoch["abs_sum"]
                grads_epoch_abs_sum += 1e-6
                p = grads_epoch_abs_sum / grads_epoch_abs_sum.sum()
                grads_epoch_abs_entropy = -(p * torch.log(p)).sum()

                out = dict(
                    ## NOTE: in terms of opt, these are the most important
                    grads_std_tot=grads.std().item(),
                    grads_var_tot=grads.var().item(),
                    grads_std_mean=grads.std(0).mean().item(),
                    grads_var_mean=grads.var(0).mean().item(),
                    ## NOTE: now all gradients are unitary, so it gives you more info about their
                    ## conflict, but less about optimization
                    grads_normed1_std_tot=grads_normalized1.std().item(),
                    grads_normed1_var_tot=grads_normalized1.var().item(),
                    grads_normed1_std_mean=grads_normalized1.std(0).mean().item(),
                    grads_normed1_var_mean=grads_normalized1.var(0).mean().item(),
                    ## NOTE: now they are normalized at a batch level. could be easier to compare 2 methods...
                    ## but again we are losing info about optimizatin
                    grads_normed2_std_tot=grads_normalized2.std().item(),
                    grads_normed2_var_tot=grads_normalized2.var().item(),
                    grads_normed2_std_mean=grads_normalized2.std(0).mean().item(),
                    grads_normed2_var_mean=grads_normalized2.var(0).mean().item(),
                    ## NOTE: this one if for weight movement per epoch
                    grads_epoch_entropy=grads_epoch_entropy.item(),
                    grads_epoch_abs_entropy=grads_epoch_abs_entropy.item(),
                )
                return out

            # 2. Optimize the critic
            clear_backprops(self.critic)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.monitor_grads:
                critic_grads_epoch = accumulate_grads_epoch(
                    self.critic, critic_grads_epoch
                )
                if last_time:
                    compute_grad1(self.critic)
                    critic_grad_summary = gradient_analysis(
                        self.critic, critic_grads_epoch
                    )
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.critic.parameters():
                p.requires_grad = False

            # for logging
            out["critic_loss"].append(critic_loss.item())
            out["Q1Vals"].append(q_info["Q1Vals"].mean())
            out["Q2Vals"].append(q_info["Q2Vals"].mean())

            ########
            # policy updates
            ########
            # Optimize the actor
            actor_loss, alout, pi_info = self.compute_loss_pi(
                task_id, obs, pre_act_rew, last_time
            )

            clear_backprops(self.actor)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.monitor_grads:
                actor_grads_epoch = accumulate_grads_epoch(
                    self.critic, critic_grads_epoch
                )
                if last_time:
                    compute_grad1(self.actor)
                    actor_grad_summary = gradient_analysis(
                        self.actor, actor_grads_epoch
                    )
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.critic.parameters():
                p.requires_grad = True

            # for logging
            out["actor_loss"].append(actor_loss.item())
            out["LogPi"].append(pi_info["LogPi"].mean())
            if self.use_auto_entropy:
                out["alpha_loss"].append(alout)
                out["alpha"].append(self.alpha.item())

            ########
            # target updates
            ########
            with torch.no_grad():
                # Update the frozen target models
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    # Use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    target_param.data.mul_((1 - self.ptau))
                    target_param.data.add_(self.ptau * param.data)

            extras["time_updating"].append(time.time() - start_updating)

        if self.monitor_grads:
            extras["critic_grad_summary"] = critic_grad_summary
            extras["actor_grad_summary"] = actor_grad_summary

        return out, extras
