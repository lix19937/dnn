# coding=utf-8

# Copes with wrong collections of labels and clouds.

import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--cloud_dir', help='e.g.: /Data/zzn/luminar_od/lidarxxx/clouds', required=True, type=str)
args = parser.parse_args()

cloud_dir = args.cloud_dir
sub_dirs = os.listdir(cloud_dir)
for one_sub in sub_dirs:
    one_folder = os.path.join(cloud_dir, one_sub)
    all_bins = [os.path.join(one_folder, one_bin)
                for one_bin in os.listdir(one_folder)]
    wrong_files = []
    for one_bin in all_bins:
        one_json = one_bin.replace('clouds', 'labels').replace('.bin', '.json')
        if not os.path.exists(one_json):
            wrong_files.append(one_json)
    for one_file in wrong_files:
        for ano_sub in sub_dirs:
            if ano_sub == one_sub:
                continue
            tmp_file = one_file.replace(one_sub, ano_sub)
            if os.path.exists(tmp_file):
                cmd = [
                    'mv',
                    tmp_file,
                    one_file
                ]
                subprocess.check_call(cmd)
                print(cmd)
