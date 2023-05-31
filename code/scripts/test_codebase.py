import os

methods = os.listdir("./code/configs/methods")

python = "/Users/massimo/opt/anaconda3/envs/rl_4_cl/bin/python3"

# NOTE cant run in debug mode

for method in methods:
    method = method[:-5]  # removes ".yaml"

    if method[0] == "_":
        # skip test
        continue

    for train_mode in ['cl', 'mtl']:

        cmd = f"{python} code/main.py --train_mode {train_mode} --method_config {method} --hparam_config test_hparams "
        print(cmd)
        if os.system(cmd) != 0:
            err = f"following test failed: {method} in {train_mode} mode"
            raise RuntimeError(err)

print("congrats! all tests passed")
