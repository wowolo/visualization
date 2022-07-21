import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_config', type=int)

args = parser.parse_args()

num_config = args.num_config
print(num_config)

if isinstance(num_config, type(None)):
    num_config = 1
num_config -= 1

print(num_config)
