import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, type=str)
args = parser.parse_args()

data_root = args.data_root
sub_folders = [os.path.join(data_root, one, 'label')
               for one in os.listdir(data_root)]

time2intrinsic = {}
all_intrinsics = set()

for one_folder in sub_folders:
    all_txt = [os.path.join(one_folder, one) for one in os.listdir(one_folder)]
    for one_txt in all_txt:
        timestamp = one_txt.split(
            '/')[-1].replace('.txt', '').replace('.json', '')
        print(timestamp)
        try:
            with open(one_txt, 'r') as f:
                data = json.load(f)
            intrin = str(data['params']['intrinsic'])
            time2intrinsic[timestamp] = intrin
            all_intrinsics.add(intrin)
        except:
            continue

print('======================')
print(all_intrinsics)
file_name = data_root.split('/')[-1] + '.json'
with open(file_name, 'w') as f:
    json.dump(time2intrinsic, f)
