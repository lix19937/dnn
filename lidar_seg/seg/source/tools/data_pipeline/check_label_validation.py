import json
import os


invalid_labels = []
root = '/home/igs/Downloads/lidar128_label2'
subs = [os.path.join(root, one_sub, 'label') for one_sub in os.listdir(root)]
for one_sub in subs:
    labels = [os.path.join(one_sub, one_label)
              for one_label in os.listdir(one_sub)]
    for one_label in labels:
        with open(one_label, 'r') as f:
            data = f.readlines()
            data = data[0]
        # Some json is invalid like '\x00\x00...'
        if '{' not in data or '}' not in data or '\x00' in data:
            invalid_labels.append(one_label)
            print(one_label)
print(invalid_labels)
print(len(invalid_labels))
