import os

methods = os.listdir('./code/configs/methods')

python = '/Users/massimo/opt/anaconda3/envs/rl_4_cl/bin/python3'

#NOTE cant run in debug mode

for method in methods:
    method = method[:-5] #removes ".yaml"

    if method[0] == '_':
        # skip test
        continue

    cmd = f'{python} code/main_cl.py --method_config {method} --hparam_config test_hparams '
    print(cmd)
    if os.system(cmd) !=0:
        err =f'following test failed: {method}' 
        raise RuntimeError(err)
        
print('congrats! all tests passed')


'''
last printed line for each test:

ERv
---------------------------------------
Evaluation during task 2 after 960 updates:
Episodic reward for the current task 4.54
Average episodic reward over all tasks 5.34 +/- 0.58
---------------------------------------

3RL
---------------------------------------
Evaluation during task 2 after 960 updates:
Episodic reward for the current task 4.57
Average episodic reward over all tasks 4.88 +/- 0.28
---------------------------------------

ER-MH
---------------------------------------
Evaluation during task 2 after 960 updates:
Episodic reward for the current task 1.69
Average episodic reward over all tasks 1.62 +/- 0.83
---------------------------------------

FineTuning
---------------------------------------
Evaluation during task 2 after 960 updates:
Episodic reward for the current task 4.52
Average episodic reward over all tasks 5.27 +/- 0.55
---------------------------------------

'''