from __future__ import  print_function, division
from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from misc.utils import get_action_info, atanh
from functools import partial



class ActorSAC(nn.Module):
    """
      This arch is standard for actor
    """
    def __init__(self,
                action_space,
                hidden_sizes = [256, 256],
                input_dim = None,
                max_action = None,
                context_id = False,
                context_rnn = False,
                context_embedding = False,
                nb_tasks = None,
                hiddens_dim_context = [50],
                input_dim_context=None,
                output_context=None,
                history_length = 1,
                obsr_dim = None,
                LOG_STD_MAX = 2,
                LOG_STD_MIN = -20,
                device = 'cpu',
                eparams = None,
                ):

        super(ActorSAC, self).__init__()

        self.action_dim, action_space_type = get_action_info(action_space)

        self.nb_tasks = nb_tasks
        self.multi_head = eparams.multi_head
        self.task_agnostic = eparams.task_agnostic
        self.device = device
        
        
        ################
        # Define actor 
        ################
        hidden_sizes = [input_dim] + hidden_sizes
        self.net = MLPSequential(hidden_sizes=hidden_sizes,
                                    output_dim=2*self.action_dim,
                                    nb_heads = 1 if not self.multi_head else nb_tasks,
                                    task_agnostic=self.task_agnostic)
        
        
        ################
        # Context Network 
        ################
        self.context_id = context_id
        self.context_rnn = context_rnn
        self.context_embedding = context_embedding

        if context_id:
            self.context_id_fn = partial(F.one_hot, num_classes=self.nb_tasks)
        else:
            self.context_id_fn = None

        if context_rnn:
            self.context_rnn_fn = ContextRNN(hidden_sizes = hiddens_dim_context,
                                             input_dim = input_dim_context,
                                             output_dim = output_context,
                                             history_length = history_length,
                                             action_dim = self.action_dim,
                                             obsr_dim = obsr_dim,
                                             device = device
                                             )
        else:
            self.context_rnn_fn = None

        if context_embedding:
            self.context_embedding_fn = nn.Embedding(nb_tasks, hiddens_dim_context[-1])
        else:
            self.context_embedding_fn = None
            
        self.act_limit  = max_action
        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_MIN = LOG_STD_MIN


    def forward(self, task_id, x, pre_act_rew=None, state=None, ret_context=False, 
                deterministic=False, with_logprob=True,  with_log_mle=False, gt_actions=None):
        '''
            input (x  : B * D where B is batch size and D is input_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''

        ############
        ## Context 
        ############
        if self.context_id:
            context_id = self.context_id_fn(task_id)
            context = task_id
            x = torch.cat([x, context_id], dim = -1)
        if self.context_rnn_fn is not None:
            context = self.context_rnn_fn(pre_act_rew)
            if self.context_rnn:
                x = torch.cat([x, context], dim = -1)
        if self.context_embedding_fn is not None:
            context = self.context_embedding_fn(task_id)
            if self.context_embedding:
                x = torch.cat([x, context], dim = -1)

        ####################
        ## Action prediction
        ####################

        extras = {}

        x, extras = self.net(x, task_id)

        ## task agnostic multi-head
        if self.multi_head and self.task_agnostic:
            mu, log_std = x[:, :, :self.action_dim:], x[:, :, self.action_dim:]
            mean_log_std = torch.mean(log_std.detach(), axis=2)
            most_confident_heads = torch.argmax(mean_log_std, axis=0)
            b = torch.arange(mu.shape[1]) 
            mu = mu[most_confident_heads, b, :]
            log_std = log_std[most_confident_heads, b, :]
        else:
            mu, log_std = x[:, :self.action_dim:], x[:, self.action_dim:] 
        
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        ## Pre-squash distribution and sample
        pi_distribution = torch.distributions.Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu

        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # SAC paper (arXiv 1801.01290) appendix C
            # This is a more numerically-stable equivalent to Eq 21.

            # pi_action [B, num_actions], logp_pi: [B]
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi:  [B, num_actions].sum(-1) ==> [B]
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
            # logp_pi: [B] ==> [B, 1]
            logp_pi = logp_pi.unsqueeze(-1)

        else:
            logp_pi = None


        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        if ret_context:
            return pi_action, logp_pi, context, extras
        else:
            return pi_action, logp_pi, extras


class CriticSAC(nn.Module):
    """
      This arch is standard based on https://github.com/sfujim/TD3/blob/master/TD3.py
    """
    def __init__(self,
                action_space,
                hidden_sizes = [256, 256],
                input_dim = None,
                # hidden_activation = F.relu,
                context_id = False,
                context_embedding = False,
                context_rnn = False,
                nb_tasks = None,
                dim_others = 0,
                hiddens_dim_context = [50],
                input_dim_context=None,
                output_context=None,
                history_length = 1,
                obsr_dim = None,
                device = 'cpu',
                eparams = None,
                ):
        super(CriticSAC, self).__init__()
        
        action_dim, action_space_type = get_action_info(action_space)

        self.nb_tasks = nb_tasks
        self.multi_head = eparams.multi_head
        self.task_agnostic = eparams.task_agnostic
        self.device = device
        
        critic_input_dim = input_dim + action_dim + dim_others 
        
        ################
        # Define critic 
        ################
        # It uses two different Q networks
        critic_hidden_sizes = [critic_input_dim, *hidden_sizes]
        self.q1 = MLPSequential(critic_hidden_sizes, output_dim=1, 
                                nb_heads = 1 if not self.multi_head else nb_tasks,
                                task_agnostic=self.task_agnostic)
        self.q2 = MLPSequential(critic_hidden_sizes, output_dim=1,
                                nb_heads = 1 if not self.multi_head else nb_tasks,
                                task_agnostic=self.task_agnostic)
        
        ################
        # Define context 
        ################
        self.context_id = context_id
        self.context_rnn = context_rnn
        self.context_embedding = context_embedding
        
        if context_id:
            self.context_id_fn = partial(F.one_hot, num_classes=self.nb_tasks)
        else:
            self.context_id_fn = None

        if context_rnn:
            self.context_rnn_fn = ContextRNN(hidden_sizes = hiddens_dim_context,
                                             input_dim = input_dim_context,
                                             output_dim = output_context,
                                             history_length = history_length,
                                             action_dim = action_dim,
                                             obsr_dim = obsr_dim,
                                             device = device
                                             )
        else:
            self.context_rnn_fn = None

        if context_embedding:
            self.context_embedding_fn = nn.Embedding(nb_tasks, hiddens_dim_context[0])
        else:
            self.context_embedding_fn = None


    
    def forward(self, task_id, obs, a, pre_act_rew=None, ret_context=False):
        '''
            input (x): B * D where B is batch size and D is input_dim
            input (u): B * A where B is batch size and A is action_dim
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
        '''
        xu = torch.cat([obs, a], dim =-1)
        # combined = None

        #############
        ## Context 
        #############
        if self.context_id:
            context_id = self.context_id_fn(task_id)
            xu = torch.cat([xu, context_id], dim = -1)
        if self.context_rnn_fn is not None:
            context_rnn = self.context_rnn_fn(pre_act_rew)
            if self.context_rnn:
                xu = torch.cat([xu, context_rnn], dim = -1)
        if self.context_embedding_fn is not None:
            context_embedding = self.context_embedding_fn(task_id)
            if self.context_embedding:
                xu = torch.cat([xu, context_embedding], dim = -1)

        #############
        ## Value prediction
        #############
        x1, extras1 = self.q1(xu, task_id)
        x2, extras2 = self.q2(xu, task_id)

        ## task-agnostic multi-head 
        if self.multi_head and self.task_agnostic:
            ## choose most optimistic head
            x1, _ = torch.max(x1, axis=0)
            x2, _ = torch.max(x2, axis=0)
        
        if ret_context == True:
            #TODO rm support for this
            return x1, x2, combined, extras1, extras2

        else:
            return x1, x2, extras1, extras2


class MLPSequential(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 output_dim,
                 nb_heads,
                 task_agnostic,
                ):
        super(MLPSequential, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.nb_heads = nb_heads
        self.task_agnostic = task_agnostic
        self.params = []
        self.output_layer = []

        ## init
        for layer in range(len(hidden_sizes)-1):
            self.params.append(nn.Linear(hidden_sizes[layer], hidden_sizes[layer+1]))
        if nb_heads>1:
            self.output_layer = torch.randn(nb_heads, output_dim, hidden_sizes[-1]+1,
                    requires_grad=True)
        else:
            self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)

        ## register
        self.params = nn.ModuleList(self.params)
        if nb_heads>1:
            #TODO smth so that they appear in the model print
            self.output_layer = nn.Parameter(self.output_layer)
        else:
            self.output_layer = nn.ModuleList([self.output_layer])[0]

    def forward(self, x, task_id=None):

        activations = []
        out = x
        
        for layer in range(len(self.params)):

            out = self.params[layer](out)
            out = torch.relu(out)
            activations.append(out)

        ## apply output layer
        if self.nb_heads>1:
            # add ones for the bias terminal
            out = torch.cat((out, torch.ones_like(task_id).unsqueeze(1)), dim=1)  
            
            if self.task_agnostic:
                ## tensor gymastics
                # self.output_layer.shape = [nb_heads, output_dim, hidden_dim]
                # out.shape = [B, hidden_dim]
                out = out.transpose(0, 1)
                # out.shape = [hidden_dim, B]
                out = torch.matmul(self.output_layer, out)
                # out.shape = [nb_heads, output_dim, B]
                out = out.transpose(1, 2)
                # out.shape = [nb_heads, B, output_dim]
            else:
                ## get output heads
                output_heads = self.output_layer[task_id] 
                ## apply
                out = (output_heads * out.unsqueeze(1)).sum(-1)
        else:
            out = self.output_layer(out)

        return out, {'activations': activations}


class ContextRNN(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """
    def __init__(self,
                 hidden_sizes = [50],
                 output_dim = None,
                 input_dim = None,
                 hidden_activation=F.relu,
                 history_length = 1,
                 action_dim = None,
                 obsr_dim = None,
                 device = 'cpu'
                 ):

        super(ContextRNN, self).__init__()
        self.hid_act = hidden_activation
        self.fcs = [] # list of linear layer
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.output_dim_final = output_dim # count the fact that there is a skip connection
        self.output_dim_last_layer  = output_dim // 2
        self.hist_length = history_length
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim

        #### build LSTM or multi-layers FF
        self.recurrent =nn.GRU(self.input_dim,
                            self.hidden_sizes[0],
                            bidirectional = False,
                            batch_first = True,
                            num_layers = 1)

    def init_recurrent(self, bsize = None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def forward(self, data):
        '''
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        '''
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]
        
        # first prepare data for LSTM
        bsize, dim = previous_action.shape # previous_action is B* (history_len * D)
        pacts = previous_action.view(bsize, -1, self.action_dim) # view(bsize, self.hist_length, -1)
        prews = previous_reward.view(bsize, -1, 1) # reward dim is 1, view(bsize, self.hist_length, 1)
        pxs   = pre_x.view(bsize, -1, self.obsr_dim ) # view(bsize, self.hist_length, -1)
        pre_act_rew = torch.cat([pacts, prews, pxs], dim = -1) # input to LSTM is [action, reward]

        # init lstm/gru
        hidden = self.init_recurrent(bsize=bsize)

        # lstm/gru
        _, hidden = self.recurrent(pre_act_rew, hidden) # hidden is (1, B, hidden_size)
        out = hidden.squeeze(0) # (1, B, hidden_size) ==> (B, hidden_size)

        return out
