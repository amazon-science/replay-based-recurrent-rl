import os

methods = os.listdir('./code/configs/methods')

python = '/Users/caccimou/opt/anaconda3/envs/sequoia/bin/python'

#NOTE cant run in debug mode

for method in methods:
    method = method[:-5] #removes ".yaml"

    if method[0] == '_':
        # skip test
        continue

    if any(x in method for x in ['ID', 'EMB', 'MH']):
        # doesnt make sense in STL
        continue

    cmd = f'{python} code/main_mtl.py --method_config {method} --hparam_config debug_stl '
    print(cmd)
    if os.system(cmd) !=0:
        err =f'following test failed: {method}' 
        raise RuntimeError(err)
        
print('congrats! all tests passed')
