import os
import numpy as np
import gym
import glob
import json
from collections import deque, OrderedDict
import psutil
import re
import csv
import pandas as pd
import ntpath
import re
import yaml
import argparse
import torch
from misc.torch_utility import get_state

def hess_analysis(model):
    ## be mindfull we are only getting the Linear layers (no RNN, MH and module_select)
    hesses = None
    for p in model.parameters():
        if hasattr(p, 'hess'):
            hess = torch.clone(p.hess).detach().flatten()
            if hesses is None:
                hesses = hess
            else:
                hesses = torch.cat((hesses, hesses), axis=0)
    out = dict(
        hesses_std = hesses.std().item(),
        hesses_tot = hesses.var().item(),
        hesses_abs_mean = hesses.abs().mean().item(),
        hesses_mean = hesses.mean().item(),
    )
    return out 

def print_model_info(models, enable_print=True):
    '''
        models is [actor, critic]
    '''
    total_params = 0
    total_params_Trainable = 0

    for model in models:
        for i in model.parameters():
            total_params += np.prod(i.size())
            if (i.requires_grad == True):
                total_params_Trainable += np.prod(i.size())
        if enable_print:
            print(model)
    if enable_print:
        # since there are target actor and actor, critic and target crtitc ==> total_params * 2
        print("Total number of ALL parameters: %d" % (total_params * 2))
        print("Total number of TRAINABLE parameters: %d" % (total_params_Trainable * 2))
    
    return total_params, total_params_Trainable


def atanh(x):
    '''
        aratnh = 0.5 * log ((1+ x) / (1-x))
    '''
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x/ one_minus_x)

            
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def softmax(x, T=1.):

    y = x - np.max(x)  # for numerical stability
    f_x = np.exp(y/T) / np.sum(np.exp(y/T))
    return f_x

def create_dir(directory):

    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except OSError as e:
            raise ValueError(e)

def prepare_ec2(machines, user):
    print('Preparing ec2 machines.....')
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
        
    def ck_mc(ip, output, user= 'ubuntu'):
        
        list_cmds = []
        cmd_run =  ' ssh -i ' + SSH_KEY_NAME + ' -T ' + ' -o StrictHostKeyChecking=no ' + user +'@' + ip + " rm ec2-*  "

        if os.system(cmd_run) !=0:
            list_cmds.append(cmd_run)
        output.put(list_cmds)
    
    for mi in chunks(machines, 4):
        
        all_outputs = mp.Queue()
        processes = [mp.Process(target=ck_mc, args=(ip,all_outputs, user)) 
                 for ip in mi]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()
        
        #TODO some weird stuff is happening here. Maybe check w/ Rasool
        print("WZZZZZZZZZZZZ")
        time.sleep(2)
        os.system("ps -ef | grep ssh |cut -d. -f1 | awk '{print $2}' | xargs kill -9")
        print('Done with this step.')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_fname_from_path(f):
    '''
     input:
           '/Users/user/logs/check_points/mmmxm_dummy_B32_H5_D1_best.pt'
     output:
           'mmmxm_dummy_B32_H5_D1_best.pt'
    '''
    return ntpath.basename(f)

def identity(x):
    '''
        identity function 
    '''
    return x

def get_action_info(action_space, obs_space = None):
    '''
        This fucntion returns info about type of actions.
    '''
    space_type = action_space.__class__.__name__

    if action_space.__class__.__name__ == "Discrete":
            num_actions = action_space.n

    elif action_space.__class__.__name__ == "Box":
            num_actions = action_space.shape[0]

    elif action_space.__class__.__name__ == "MultiBinary":
            num_actions = action_space.shape[0]
    
    else:
        raise NotImplementedError
    
    return num_actions, space_type

def create_dir(log_dir, ext = '*.monitor.csv', cleanup = False):

    '''
        Setup checkpoints dir
    '''
    try:
        os.makedirs(log_dir)

    except OSError:
        if cleanup == True:
            files = glob.glob(os.path.join(log_dir, '*.'))

            for f in files:
                os.remove(f)

def dump_to_json(path, data):
    '''
      Write json file
    '''
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json(input_json):
    ## load the json file
    file_info = json.load(open(input_json, 'r'))

    return file_info

class CSVWriter:

    def __init__(self, fname, fieldnames):

        self.fname = fname
        self.fieldnames = fieldnames
        self.csv_file = open(fname, mode='w')
        self.writer = None

    def write(self, data_stats):

        if self.writer == None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(data_stats)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def safemean(xs):
    '''
        Avoid division error when calculate the mean (in our case if
        epinfo is empty returns np.nan, not return an error)
    '''
    return np.nan if len(xs) == 0 else np.mean(xs)


def overwrite_args(args, config_type, config):
    '''
    Overwritting arguments to set the configs for a particular experiment.
    '''
    with open(f'code/configs/{config_type}/{config}.yaml') as f:
        file_args = yaml.safe_load(f, Loader=yaml.FullLoader)
        # overwrite the default values with the values from the file.
        args_dict = vars(args)
        file_args_dict = vars(file_args)
        if not args_dict.keys() >= file_args_dict.keys():
            raise ValueError("the config file to overwrite the args contains a wrong arg")
        args_dict.update(vars(file_args))
        return argparse.Namespace(**args_dict)


def take_snapshot(args, ck_fname_part, model, update):
    '''
        This fucntion just save the current model and save some other info
    '''
    fname_ck =  ck_fname_part + '.pt'
    fname_json =  ck_fname_part + '.json'
    curr_state_actor = get_state(model.actor)
    curr_state_critic = get_state(model.critic)

    print('---------------------------------------')
    print('Saving a checkpoint for iteration %d in %s' % (update, fname_ck))
    print('---------------------------------------')
    checkpoint = {
                    'args': args.__dict__,
                    'model_states_actor': curr_state_actor,
                    'model_states_critic': curr_state_critic,
                 }
    torch.save(checkpoint, fname_ck)

    del checkpoint['model_states_actor']
    del checkpoint['model_states_critic']
    del curr_state_actor
    del curr_state_critic

    dump_to_json(fname_json, checkpoint)

def setup_logAndCheckpoints(args):

    # create folder if not there
    create_dir(args.check_point_dir)

    fname = str.lower(args.env_name) + '_' + args.alg_name + '_' + args.log_id + '_s' + str(args.seed)
    fname_log = os.path.join(args.log_dir, fname)
    fname_eval = os.path.join(fname_log,  'eval.csv')

    return os.path.join(args.check_point_dir, fname), fname_log, fname_eval
