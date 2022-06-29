import os

methods = os.listdir('./code/configs/methods')

python = '/Users/caccimou/opt/anaconda3/envs/sequoia/bin/python'

#NOTE cant run in debug mode

for method in methods:
    method = method[:-5] #removes ".yaml"

    if method[0] == '_':
        # skip test
        continue

    cmd = f'{python} code/main_mtl.py --method_config {method} --hparam_config test_hparams '
    print(cmd)
    if os.system(cmd) !=0:
        err =f'following test failed: {method}' 
        raise RuntimeError(err)

print('congrats! all tests passed')

'''
last printed line for each test:

ERv
---------------------------------------
Evaluation after 995 updates:
Average episodic reward over all tasks 5.41 +/- 0.62
---------------------------------------

3RL
---------------------------------------
Evaluation after 995 updates:
Average episodic reward over all tasks 4.27 +/- 0.28
---------------------------------------

ER-MH
---------------------------------------
Evaluation after 995 updates:
Average episodic reward over all tasks 0.80 +/- 1.20
---------------------------------------

'''